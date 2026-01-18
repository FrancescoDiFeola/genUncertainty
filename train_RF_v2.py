import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from src import StochasticFlowScheduler
from tqdm import tqdm
from src import LDCTHDCTDataset
from src import networks
import matplotlib.pyplot as plt
from src import T1T2Dataset


# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


# -----------------------
# ✅ Log to tensorboard
# -----------------------


@torch.no_grad()
def run_inference(
    diffusion_model,
    condition_batch,
    gt_batch,
    writer,
    step,
    device,
    scheduler,
    num_inference_steps=30,
    tag="FlowMatching_Sampling",
):
    """
    Stochastic Flow Matching inference via ODE integration.
    Plots [Condition | GT | Prediction | Error] per row.
    """

    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    # Initial sample x_1 ~ N(0, I)
    x = torch.randn_like(condition_batch).to(device)

    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    dt = -1.0 / num_inference_steps

    for i in range(num_inference_steps):
        # Continuous time t in [1, 0]
        t = 1.0 - i / num_inference_steps
        t_tensor = torch.full((B,), t * scheduler.num_train_timesteps, device=device)

        # Model input
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            predicted_velocity = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
            )

        # Euler ODE step
        x = x + predicted_velocity * dt

    final_output = x

    # -------------------------
    # Visualization
    # -------------------------
    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

    cond = condition_batch.cpu()
    gt = gt_batch.cpu()
    pred = final_output.cpu()
    error = norm(torch.abs(pred - gt))

    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(8, 2.5 * num_samples))
    if B == 1:
        axes = [axes]

    for s in range(num_samples):
        images = [cond[s], gt[s], pred[s], error[s]]
        titles = ["Condition", "GT", "Prediction", "Error"]

        for j in range(4):
            ax = axes[s][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).numpy()
            cmap = "hot" if titles[j] == "Error" else "gray"
            ax.imshow(img, cmap=cmap)

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()

# -----------------------
# ✅ Training script
# -----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--epoch_start', default=0, type=float)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)

    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    # Load the LDCT/HDCT dataset
    dataset = T1T2Dataset(
        annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
        annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

    )

    """
    dataset = LDCTHDCTDataset(
        annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
        annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
    )
    """

    # dataset = CTPETDataset(args)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)

    # ----------------------- #
    # ✅ Load diffusion model #
    # ----------------------- #
    diffusion = networks.init_ddpm(2, 1, args.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    scheduler = StochasticFlowScheduler(
        num_train_timesteps=1000,
        sigma_min=0.0,
        sigma_max=1.0,
    )

    scaler = GradScaler()
    writer = SummaryWriter(comment=args.experiment_name)

    global_counter = {'train': 0}

    # -------------------------------
    # ✅ Training loop
    # -------------------------------
    for epoch in range(args.n_epochs):
        diffusion.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            img_A = batch["A"].to(DEVICE)  # condition
            img_B = batch["B"].to(DEVICE)  # target

            batch_size = img_B.shape[0]

            # source distribution
            x0 = torch.randn_like(img_B)

            # stochastic perturbation
            eps = torch.randn_like(img_B)

            # sample continuous time t ∈ (0,1)
            t = scheduler.sample_timesteps(
                batch_size=batch_size,
                device=img_B.device,
            )

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                # stochastic interpolation
                x_t = scheduler.interpolate(
                    x0=x0,
                    x1=img_B,
                    t=t,
                    noise=eps,
                )

                # conditional model input
                model_input = torch.cat([x_t, img_A], dim=1)

                # predict velocity
                predicted_velocity = diffusion(
                    x=model_input,
                    timesteps=t,  # continuous time
                )

                # correct stochastic flow target
                target_v = scheduler.target_velocity(
                    x0=x0,
                    x1=img_B,
                    noise=eps,
                    t=t,
                )

                loss = F.mse_loss(
                    predicted_velocity.float(),
                    target_v.float(),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss", loss.item(), global_counter["train"])
            epoch_loss += loss.item()
            global_counter["train"] += 1

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            torch.cuda.empty_cache()

            if step % 150 == 0:
                run_inference(
                    diffusion_model=diffusion,
                    condition_batch=img_A,
                    gt_batch=img_B,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,  # same scheduler is fine
                    num_inference_steps=30,  # ODE steps
                    tag="Rectified_Flow",
                )

        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if epoch % 50 == 0:
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{epoch}.pth'))

    print("Training complete.")
