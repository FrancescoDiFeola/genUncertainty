import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from torchvision import transforms
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp import networks
from src.VAE.utils.checkpoints_utils import load_checkpoint
from src.inference.inference_LFM import *
from src.inference.utils import initialize_writers
from src.brlp.MR_to_CT import MRCTPaired
from src.brlp.CBCTtoCT_dataset import CBCTCTPaired
from src.brlp.motionArtifact_dataset import MotionT1Dataset
# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str,default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", required=False)
    parser.add_argument('--diff_ckpt', type=str, required=False)
    parser.add_argument('--context_ckpt', default=None, type=str)
    parser.add_argument('--VAE_ckpt', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--motion_level', default=1, type=str)
    parser.add_argument('--ablation', action="store_true")
    parser.add_argument('--analysis', type=str, required=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--spatial_enc_channels', type=int, default=2)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)

    parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999], help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')

    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.task, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    args.diff_ckpt = os.path.join(experiment_dir, f"diffusion-ep-{args.epoch}.pth")
    args.context_ckpt = os.path.join(experiment_dir, f"spatial_encoder-ep-{args.epoch}.pth")
    args.VAE_ckpt = os.path.join(args.output_dir, args.task, "VAE")

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    scaling_factor = 1
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        # dataset = Mri2DSlicedataset(args)
        # scaling_factor = 9.404202

        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',

        )
        scaling_factor = 9.404202

    elif args.task == "T1motion":

        """
        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',
            mode="train",
            motion_range=(0.0, 0.15),
        )
        """

        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
            mode="test",
            fixed_motion_level = float(args.motion_level),
        )  # test_dataset_lvl_0 = T1T2Dataset(..., mode="test", fixed_motion_level=0.0

        scaling_factor = 5.634654

    elif args.task == "CS":
        transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor()
        ])

        dataset = CityscapesColorDataset(
            root=args.dataroot,
            split="train",
            transform=transform,
            target_transform=transform
        )

    elif args.task == "ND":
        transform = transforms.Compose([
            transforms.Resize((272, 480)),
            transforms.ToTensor()
        ])

        dataset = PairedImageDataset(
            csv_path="train.csv",
            root_dir="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/ND_dataset",
            transform_A=transform,
            transform_B=transform
        )

    elif args.task == "CTPET":
        dataset = Mri2DSlicedataset(args)

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            # annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            # annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
        )
        scaling_factor = 7.832608

    elif args.task == "T1T2_Oasis":
        dataset = Mri2DSlicedataset(args)
        scaling_factor = 9.404202

    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_train.csv",
            output_size=256,
        )
        scaling_factor = 6.640712

    elif args.task == "CBCTtoCT":

        dataset = CBCTCTPaired(
            csv_path="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_test.csv",
            output_size=256,
        )
        scaling_factor = 9.744896

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    # -----------------------
    # ✅ Load autoencoder
    # -----------------------
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 128, 256),
        latent_channels=3,
        num_res_blocks=2,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )
    autoencoder = autoencoder.to(DEVICE)

    # **Load Checkpoints from checkpoint.py**
    _ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=args.VAE_ckpt, model_name="autoencoder")
    autoencoder.eval()

    diffusion = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)
    spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)
        autoencoder = torch.nn.DataParallel(autoencoder)
        spatial_encoder = torch.nn.DataParallel(spatial_encoder)

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=64*64,
        spatial_dim=2
    )

    scheduler.set_timesteps(num_inference_steps=30, device=DEVICE, input_img_size_numel=64*64)

    writer = SummaryWriter(comment=args.experiment_name)

    if args.analysis == "sparsification":

        csv_path = os.path.join(experiment_dir, f"sparsification_K_30_epoch_{args.epoch}_motion_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "metrics":

        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_image_uncertainty_k_30_motion_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "uncertainty_eval":
        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_uncertainty_eval_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "uncertainty_cal":
        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_uncertainty_cal_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    for step, batch in enumerate(loader):
        img_A = batch["A"].to(DEVICE)
        img_B = batch["B"].to(DEVICE)

        with torch.no_grad():
            _, img_A_latent, _ = autoencoder(img_A)

        img_A_latent = img_A_latent * scaling_factor

        if args.analysis == "sparsification":

            run_inference_LFM_self_refining_and_log_uncertainty_propagation_sparsification(
                diffusion_model=diffusion,
                autoencoder=autoencoder,
                context_encoder=spatial_encoder,
                condition_batch=img_A_latent,
                gt_batch=batch['B'],
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                scaling=scaling_factor,
                csv_writer=writer_csv,
                K=30,
            )

        elif args.analysis == "metrics":

            if args.ablation:

                run_inference_LFM_self_refining_and_log_uncertainty_propagation_ablation(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    context_encoder=spatial_encoder,
                    writer=writer,
                    condition_batch=img_A_latent,
                    gt_batch=batch['B'],
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    scaling=scaling_factor,
                    csv_writer=writer_csv,
                    K=30,
                )

            else:

                run_inference_LFM_self_refining_and_log_uncertainty_propagation(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    context_encoder=spatial_encoder,
                    writer=writer,
                    condition_batch=img_A_latent,
                    gt_batch=batch['B'],
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    scaling=scaling_factor,
                    csv_writer=writer_csv,
                    K=30,
                )
        elif args.analysis == "uncertainty_eval":
            run_inference_LFM_self_refining_and_log_uncertainty_eval(
                diffusion_model=diffusion,
                autoencoder=autoencoder,
                context_encoder=spatial_encoder,
                writer=writer,
                condition_batch=img_A_latent,
                gt_batch=batch['B'],
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                scaling=scaling_factor,
                csv_writer=writer_csv,
                K=30,
            )
        elif args.analysis == "uncertainty_cal":
            run_inference_LFM_self_refining_and_log_uncertainty_calibration_tail_bins(
                diffusion_model=diffusion,
                autoencoder=autoencoder,
                context_encoder=spatial_encoder,
                writer=writer,
                condition_batch=img_A_latent,
                gt_batch=batch['B'],
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                scaling=scaling_factor,
                csv_writer=writer_csv,
                K=30,
            )

    print(f"✅ Inference complete. Metrics saved to {csv_path}")
