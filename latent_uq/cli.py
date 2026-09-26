"""Command-line interface: one configuration, explicit overrides, portable checkpoints."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform

import torch
import yaml

from .config import from_dict, save_config
from .data import make_loader, prepare_batch
from .models import build_models
from .sampling import sample
from .training import load_checkpoint, restore_models, train, train_step, validate_checkpoint_config
from .process import Process
from .utils import seed_everything


def apply_overrides(values, overrides):
    for assignment in overrides:
        if "=" not in assignment:
            raise ValueError("Overrides must use section.field=value")
        path, raw = assignment.split("=", 1)
        keys = path.split(".")
        node = values
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise ValueError(f"Unknown configuration mapping: {path}")
            node = node[key]
        # Constructor kwargs are intentionally extensible; other fields are strict.
        if keys[-1] not in node and "kwargs" not in keys:
            raise ValueError(f"Unknown configuration field: {path}")
        node[keys[-1]] = yaml.safe_load(raw)
    return values


def infer(config, checkpoint, output_dir, *, dry_run=False):
    """Run each requested analysis separately: they can use different sampling parameters."""
    from .inference import module_for
    config.validate()
    if checkpoint is not None:
        validate_checkpoint_config(config, checkpoint)
    else:
        if not config.model.backbone.checkpoint:
            raise ValueError("Inference from --config requires model.backbone.checkpoint")
        context = config.model.context_encoder
        if config.mode == "selfcond" and (context is None or not context.checkpoint):
            raise ValueError("Selfcond inference from --config requires context_encoder.checkpoint")
    analyses = config.inference.analyses or [None]
    # Resolve every branch before creating any output. A custom analysis has no
    # reference of its own: this validates the built-in analysis it samples like.
    for analysis in analyses:
        name = analysis or 'metrics'
        custom = config.inference.custom_analyses.get(name)
        module_for(config.framework).reference(config, custom.sampling_analysis if custom else name)
    output = Path(output_dir)
    if not dry_run and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'Inference directory is not empty: {output}; choose a new output directory')
    for analysis in analyses:
        run_config = replace(config,
                             inference=replace(config.inference,
                                               analyses=[] if analysis is None else [analysis]))
        destination = output / analysis if len(analyses) > 1 else output
        _infer_one(run_config, checkpoint, destination, dry_run=dry_run)
    if not dry_run and len(analyses) > 1:
        save_config(config, output / 'config.yaml')
        (output / 'run.json').write_text(
            json.dumps(dict(analyses=analyses, seed=config.seed, independent_analysis_runs=True),
                       indent=2))


def _infer_one(config, checkpoint, output, *, dry_run=False):
    from .inference import module_for
    from .inference.common import resolved
    from .analysis import Reporter
    import monai
    seed_everything(config.seed)
    loader = make_loader(config, training=False)
    models = build_models(config, initialize=checkpoint is None)
    if checkpoint is not None:
        restore_models(models, checkpoint)
    analysis = next(iter(config.inference.analyses), 'metrics')
    custom = config.inference.custom_analyses.get(analysis)
    reference = resolved(
        config,
        module_for(config.framework).reference(config,
                                                custom.sampling_analysis if custom else analysis))
    if dry_run:
        condition, _ = prepare_batch(next(iter(loader)),
                                     config,
                                     require_target=bool(config.inference.analyses))
        result = sample(condition, models, config)
        print(f'Inference dry run passed: {analysis}, prediction {tuple(result.image.shape)}')
        return
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / 'config.yaml')
    with Reporter(output, config) as reporter:
        for batch in loader:
            condition, target = prepare_batch(batch,
                                              config,
                                              require_target=bool(config.inference.analyses))
            reporter.write(condition,
                           target,
                           sample(condition, models, config),
                           batch.get('case_id'),
                           regions=batch.get('regions'),
                           affines=batch.get('affine'))
        count, regions = reporter.index, reporter.regions
    metadata = dict(
        framework=config.framework,
        mode=config.mode,
        analysis=analysis,
        custom_analysis=custom is not None,
        uncertainty=config.uncertainty,
        legacy_function=reference.function,
        weights_source="run_checkpoint" if checkpoint is not None else "component_checkpoints",
        prediction=reference.posthoc_prediction
        if config.uncertainty == 'posthoc' else 'single_trajectory',
        images=count,
        regions=regions,
        tiling=config.inference.tiling if config.inference.patch_size is not None else None,
        prediction_format=config.inference.prediction_format
        if config.inference.save_predictions else None,
        steps=config.steps,
        model_evaluations_per_trajectory=config.steps + (config.mode == 'selfcond'),
        samples=reference.samples if config.uncertainty == 'posthoc' else 1,
        last_k=min(reference.last_k, max(config.steps - 1, 0)),
        decode_samples=reference.decode_samples
        if config.latent and config.uncertainty == 'propagated' else 0,
        flow_dt_squared=reference.flow_dt_squared,
        variance_denominator='N',
        seed=config.seed,
        torch=str(torch.__version__),
        monai=monai.__version__,
        python=platform.python_version())
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {count} predictions/results to {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train or resume DM/FM/LDM/LFM")
    training.add_argument("--config", default="configs/example.yaml")
    training.add_argument("--resume",
                          help="Versioned run checkpoint; epochs denotes the total budget")
    inference = commands.add_parser("infer",
                                    help="Generate images, uncertainty maps and analysis CSVs")
    source = inference.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Versioned run checkpoint")
    source.add_argument("--config",
                        help="Configuration with explicit backbone/context/VAE weight files")
    for command in (training, inference):
        command.add_argument("--output-dir", required=True)
        command.add_argument("--set",
                             action="append",
                             default=[],
                             metavar="SECTION.FIELD=VALUE",
                             help="Typed YAML override; repeat as needed")
        command.add_argument("--dry-run",
                             action="store_true",
                             help="Execute one batch without writing results")
    args = parser.parse_args(argv)
    if args.command == "train":
        values = yaml.safe_load(Path(args.config).read_text()) or {}
        # Defaults are materialized without validating until CLI overrides are applied.
        from .config import Config, _construct
        values = asdict(_construct(Config, values))
        config = from_dict(apply_overrides(values, args.set))
        if args.dry_run:
            seed_everything(config.seed)
            checkpoint = load_checkpoint(args.resume) if args.resume else None
            if checkpoint:
                validate_checkpoint_config(config, checkpoint, resume=True)
            models = build_models(config, initialize=checkpoint is None)
            if checkpoint:
                restore_models(models, checkpoint)
            condition, target = prepare_batch(next(iter(make_loader(config, training=True))),
                                              config,
                                              require_target=True)
            if config.training.patch_size is not None:
                from .data import random_crop_pair
                condition, target = random_crop_pair(condition, target, config.training.patch_size,
                                                     config.training.pad_value)
            with torch.no_grad():
                loss = train_step(condition, target, models, Process(config), config)
            message = f"Training dry run passed: loss={float(loss):.6f}"
            if config.training.plot_every or config.training.sample_every:
                from .monitor import Monitor
                # Checks matplotlib and loads the monitoring examples; writes nothing.
                examples = Monitor(config, args.output_dir).examples
                message += f", {len(examples)} monitoring examples"
            print(message)
        else:
            print(train(config, args.output_dir, resume=args.resume))
    else:
        checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
        if checkpoint is not None:
            # A run checkpoint protects its saved architecture and training process.
            values = json.loads(json.dumps(checkpoint["config"]))
        else:
            from .config import Config, _construct
            values = asdict(_construct(Config, yaml.safe_load(Path(args.config).read_text()) or {}))
        config = from_dict(apply_overrides(values, args.set))
        infer(config, checkpoint, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
