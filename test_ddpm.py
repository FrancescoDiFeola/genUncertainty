import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from generative.networks.schedulers import DDIMScheduler
from torchvision import transforms
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.MR_to_CT import MRCTPaired
from src.brlp import networks
from src.inference.inference_ddpm import *
from src.inference.utils import initialize_writers
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
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--motion_level', default=1, type=str)
    parser.add_argument('--perturbation_type', default=None, required=False, type=str)
    parser.add_argument('--perturbation_level', required=False, type=int)
    parser.add_argument('--analysis', type=str, required=False)
    parser.add_argument('--epoch', default=None, type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)
    parser.add_argument('--MC_sampling', action="store_true")
    parser.add_argument('--n_sampling', default=4, type=int)

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

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
        )
    elif args.task == "T1motion":

        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
            mode="test",
            fixed_motion_level = float(args.motion_level),
        )  # test_dataset_lvl_0 = T1T2Dataset(..., mode="test", fixed_motion_level=0.0)

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
            csv_path="test.csv",
            root_dir="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/ND_dataset",
            transform_A=transform,
            transform_B=transform
        )

    elif args.task == "CTPET":
        dataset = Mri2DSlicedataset(args)

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
            perturbation_type=args.perturbation_type,
            noise_level=args.perturbation_level,
            deterministic_noise=True,
        )

    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_test.csv",
            output_size=256,
        )

    elif args.task == "T1T2_Oasis":
        dataset = Mri2DSlicedataset(args)

    elif args.task == "CBCTtoCT":

        dataset = CBCTCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_test.csv",
            output_size=256,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    diffusion = networks.init_ddpm(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    writer = SummaryWriter(comment=args.experiment_name)

    if args.analysis == "sparsification":

        csv_path = os.path.join(experiment_dir, f"sparsification_S_{args.n_sampling}_epoch_{args.epoch}_motion_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "both":

        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_image_uncertainty_MC_sampling_train.csv")
        csv_path_2 = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_uncertainty_calibration_MC_sampling_train.csv")
        writer_ = initialize_writers(csv_path, csv_path_2, writer_type=args.analysis)
        writer_csv = writer_[2]
        writer_csv_2 = writer_[3]

    elif args.analysis == "metrics_no_uncertainty":

        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_{args.perturbation_type}_level_{args.perturbation_level}_motion_level_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "uncertainty_eval":
        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_uncertainty_eval_{args.motion_level}_N_{args.n_sampling}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    if args.MC_sampling:

            for step, batch in enumerate(loader):

                if args.analysis == "sparsification":

                    run_ddpm_vanilla_inference_and_log_MC_sampling_sparsification(
                        diffusion_model=diffusion,
                        condition_batch=batch['A'],
                        gt_batch=batch['B'],
                        step=step,
                        device=DEVICE,
                        scheduler=scheduler,
                        csv_writer=writer_csv,
                        n_sampling=args.n_sampling,
                    )

                elif args.analysis == "both":
                        run_ddpm_vanilla_inference_and_log_MC_sampling(
                            diffusion_model=diffusion,
                            condition_batch=batch['A'],
                            gt_batch=batch['B'],
                            writer=writer,
                            step=step,
                            device=DEVICE,
                            scheduler=scheduler,
                            csv_writer=writer_csv,
                            csv_writer_2=writer_csv_2
                        )
                elif args.analysis == "uncertainty_eval":
                        run_ddpm_vanilla_inference_and_log_MC_sampling_uncertainty_eval(
                            diffusion_model=diffusion,
                            condition_batch=batch['A'],
                            gt_batch=batch['B'],
                            writer=writer,
                            step=step,
                            device=DEVICE,
                            scheduler=scheduler,
                            csv_writer=writer_csv,
                            n_sampling=args.n_sampling,
                        )

            print(f"✅ Inference complete. Metrics saved to {csv_path}")
    else:

        for step, batch in enumerate(loader):
            run_ddpm_vanilla_inference_and_log(
                diffusion_model=diffusion,
                condition_batch=batch['A'],
                gt_batch=batch['B'],
                writer=writer,
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                csv_writer=writer_csv
            )

        print(f"✅ Inference complete. Metrics saved to {csv_path}")

