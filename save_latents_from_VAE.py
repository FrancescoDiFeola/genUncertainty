import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#from VAE.utils.checkpoints_utils import load_checkpoint
#from VAE.models.autoencoder import Autoencoder
from torchvision import transforms
from src.VAE.utils.checkpoints_utils import load_checkpoint
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.ND_dataset import PairedImageDataset

# --- CONFIGURAZIONE ---
INPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/patches_test/images"
OUTPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/latents_test_masked"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PARAMETRI PRESTAZIONI
BATCH_SIZE = 128
NUM_WORKERS = 8


# --- 1. DEFINIZIONE DATASET ---
class NiftiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []

        print(f"Scansione file in {root_dir}...")
        for r, d, f in os.walk(root_dir):
            for file in f:
                if file.endswith(".nii.gz"):
                    full_path = os.path.join(r, file)
                    rel_path = os.path.relpath(full_path, root_dir)
                    self.files.append((full_path, rel_path))
        print(f"Trovati {len(self.files)} file.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_path, rel_path = self.files[idx]
        try:
            img = nib.load(full_path)
            data = img.get_fdata().astype(np.float32)
            # Da (D, H, W) a (1, D, H, W)
            tensor = torch.from_numpy(data).unsqueeze(0)
            return tensor, rel_path
        except Exception as e:
            print(f"Errore caricamento {full_path}: {e}")
            return torch.zeros(1), ""


# --- 2. SETUP MODELLO ---
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        latent = self.model.encoder(x)
        mu, log_var = torch.chunk(latent, 2, dim=1)
        return mu


print(f"Caricamento modello su {DEVICE}...")
num_channels = [32, 64, 128]

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

checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/src/VAE/checkpoints_masked"
_ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=checkpoint_dir, model_name="autoencoder")
encoder_only = EncoderWrapper(autoencoder).to(DEVICE)
encoder_only.eval()

# --- 3. LOOP OTTIMIZZATO (FLOAT32 + FIX SHAPE) ---
if __name__ == "__main__":
    task ==
    # Load the LDCT/HDCT dataset
    if task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

        )

    elif task == "CS":
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

    elif task == "ND":
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

    elif task == "CTPET":
        dataset = Mri2DSlicedataset(args)

    elif task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
        )


    print(f"Inizio elaborazione (Output Shape [1, 3, ...]) con Batch Size {BATCH_SIZE}...")

    sq_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_tensors, batch_paths in tqdm(loader):

            valid_indices = [i for i, p in enumerate(batch_paths) if p != ""]
            if not valid_indices:
                continue

            inputs = batch_tensors[valid_indices].to(DEVICE, non_blocking=True)
            paths = [batch_paths[i] for i in valid_indices]

            # Inferenza (Float32)
            latents = encoder_only(inputs)  # Output: [Batch, 3, 16, 16, 16]
            print(f"Latents shape: {latents.shape}")

            sq_sum += (latents ** 2).sum().item()
            count += latents.numel()

            # Sposta su CPU
            latents_cpu = latents.cpu()

            for i, rel_path in enumerate(paths):
                dest_path = os.path.join(OUTPUT_ROOT, rel_path.replace("_ct.nii.gz", ".pt"))
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)

                # --- FIX: AGGIUNTA DIMENSIONE BATCH ---
                # latents_cpu[i] è [3, 16, 16, 16]
                # unsqueeze(0) lo fa diventare [1, 3, 16, 16, 16]
                tensor_to_save = latents_cpu[i].unsqueeze(0).clone()


                torch.save(tensor_to_save, dest_path)

    print("\nFinito! Dataset latente ricreato correttamente in:", OUTPUT_ROOT)

    sigma2 = sq_sum / count
    scale = 1.0 / np.sqrt(sigma2)

    print("=================================")
    print(f"Latent variance (sigma^2): {sigma2:.6f}")
    print(f"Scaling factor: {scale:.6f}")
    print("=================================")