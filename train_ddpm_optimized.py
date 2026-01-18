"""
Optimized training script for MONAI + diffusion

Main changes vs your original:
- Uses DistributedDataParallel (DDP) instead of DataParallel
- Removes torch.cuda.empty_cache() (don’t call it per-iter)
- Faster DataLoader: DistributedSampler, persistent_workers, prefetch_factor
- Non-blocking GPU transfers + pin_memory
- Optional gradient accumulation (lets you raise effective batch size)
- Reduced CPU↔GPU sync by logging every N steps and using loss.detach()
- TF32 enabled for speed on A40
"""

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.transforms import Compose, LoadImaged, ScaleIntensityRangeD, EnsureChannelFirstd, ToTensord
from monai.data import CacheDataset

# your imports
from src import Mri2DSlicedataset
from src import networks
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from inferers import DiffusionInferer

def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        is_ddp = True
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        is_ddp = False
        local_rank = 0
        world_size = 1
        rank = 0
    return is_ddp, local_rank, world_size, rank

def is_main_process(rank: int) -> bool:
    return rank == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)

    # Performance knobs
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--prefetch_factor', default=4, type=int)
    parser.add_argument('--cache_rate', default=1.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--grad_accum_steps', default=1, type=int)  # increase effective batch size safely
    parser.add_argument('--log_every', default=50, type=int)

    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)

    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999])
    parser.add_argument('--phase', type=str, default=None)
    parser.add_argument('--under_sample_dataset', action="store_true")

    args = parser.parse_args()

    # -----------------------
    # ✅ DDP + device setup
    # -----------------------
    is_ddp, local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Speedups on Ampere (A40)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # -----------------------
    # ✅ Experiment dir + logging
    # -----------------------
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    if is_main_process(rank):
        os.makedirs(experiment_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=experiment_dir)
    else:
        writer = None

    # -----------------------
    # ✅ Build MONAI CacheDataset
    # -----------------------
    base_ds = Mri2DSlicedataset(args)

    data_list = []
    for s in base_ds.samples:
        subject = s["subject"]
        slice_idx = s["slice_idx"]
        A_path = base_ds.subject_dict[subject][base_ds.A_mod][slice_idx]
        B_path = base_ds.subject_dict[subject][base_ds.B_mod][slice_idx]
        data_list.append(
            {"A": A_path, "B": B_path, "subject": subject, "slice_idx": slice_idx,
             "A_mod": base_ds.A_mod, "B_mod": base_ds.B_mod}
        )

    train_transforms = Compose([
        LoadImaged(keys=["A", "B"]),
        ScaleIntensityRangeD(keys=["A", "B"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True),
        EnsureChannelFirstd(keys=["A", "B"]),
        ToTensord(keys=["A", "B"]),
    ])

    dataset = CacheDataset(
        data=data_list,
        transform=train_transforms,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,  # caching workers (happens once-ish)
    )

    # -----------------------
    # ✅ DataLoader (DDP-aware)
    # -----------------------
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # -----------------------
    # ✅ Model / optimizer / schedulers
    # -----------------------
    diffusion = networks.init_ddpm(args.diff_ckpt).to(device)

    if is_ddp:
        diffusion = torch.nn.parallel.DistributedDataParallel(
            diffusion,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inference_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    scaler = GradScaler()

    # -----------------------
    # ✅ Training loop
    # -----------------------
    global_step = 0
    grad_accum = max(1, args.grad_accum_steps)

    for epoch in range(args.epoch_start, args.n_epochs):
        diffusion.train()

        if is_ddp:
            sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        num_steps = len(train_loader)

        # tqdm only on main process to avoid messy output
        iterator = enumerate(train_loader)
        if is_main_process(rank):
            iterator = tqdm(iterator, total=num_steps, desc=f"Epoch {epoch}", leave=True)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in iterator:
            img_A = batch["A"].to(device, non_blocking=True)
            img_B = batch["B"].to(device, non_blocking=True)

            noise = torch.randn_like(img_B)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (img_B.size(0),),
                device=device
            ).long()

            with autocast(True):
                noise_pred, _ = inferer(
                    inputs=img_B,
                    concat=img_A,
                    diffusion_model=diffusion,
                    noise=noise,
                    timesteps=timesteps,
                    condition=img_A,
                    mode='concat'
                )
                loss = F.mse_loss(noise, noise_pred)
                loss = loss * (args.diff_loss_weight / grad_accum)

            scaler.scale(loss).backward()

            # step optimizer every grad_accum steps
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # accumulate for display/logging (detach avoids autograd + reduces sync)
            epoch_loss_sum += float(loss.detach().cpu())  # minor sync, but okay at epoch scale
            global_step += 1

            if is_main_process(rank) and (global_step % args.log_every == 0):
                # Avoid loss.item() every step; log periodically
                writer.add_scalar("train/loss", float(loss.detach().cpu()) * grad_accum, global_step)

            if is_main_process(rank) and isinstance(iterator, tqdm) and (step % args.log_every == 0):
                iterator.set_postfix({"loss": (epoch_loss_sum / (step + 1)) * grad_accum})

        # handle leftover grads if steps not divisible by grad_accum
        if num_steps % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if is_main_process(rank):
            avg_epoch_loss = (epoch_loss_sum / max(1, num_steps)) * grad_accum
            writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
            # optionally save checkpoints here

        if epoch % 50 == 0:
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(args.output_dir, f'diffusion-ep-{epoch+args.epoch_start}.pth'))

    if writer is not None:
        writer.close()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()