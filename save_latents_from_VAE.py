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
from src.VAE.configs.train_options import TrainOptions
from src.VAE.data.dataset_Denoising import LDCTHDCTDataset
from src.VAE.data.dataset_T1T2 import T1T2Dataset
from src.VAE.data.dataset_CTPET import CTPETDataset
from src.VAE.data.dataset_MRtoCT import MRCTSingleImageDataset
import csv

# --- CONFIGURAZIONE ---
# INPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/patches_test/images"
# OUTPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/latents_test_masked"
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

checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/MRtoCT/VAE"
_ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=checkpoint_dir, model_name="autoencoder")
autoencoder.eval()

# --- 3. LOOP OTTIMIZZATO (FLOAT32 + FIX SHAPE) ---
if __name__ == "__main__":

    """
    dataset = LDCTHDCTDataset(
        annotation='/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/src/VAE/csvs/Mayo_total_stacked_shuffled.csv',
    )
    """

    # dataset = T1T2Dataset("/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/src/VAE/csvs/T1T2_train.csv")

    # opt = TrainOptions()

    # dataset = CTPETDataset(opt)

    dataset = MRCTSingleImageDataset(csv_path="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_train.csv")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Inizio elaborazione (Output Shape [1, 3, ...]) con Batch Size {BATCH_SIZE}...")

    sq_sum = 0.0
    count = 0

    # check if there are NaN values inside the loaded images
    """
    nan_paths = []

    with torch.no_grad():
        for batch in tqdm(loader):

            inputs = batch['img'].to(DEVICE)  # shape: [B, C, H, W] (or similar)
            paths = batch['path']  # list of paths (length B)

            # Check NaNs per sample (not just whole batch)
            # Flatten each sample and check if any NaN exists
            B = inputs.shape[0]

            for i in range(B):
                if torch.isnan(inputs[i]).any():
                    nan_paths.append(paths[i])

    # Save CSV at the end
    output_csv = "nan_images_MRtoCT_train.csv"
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for p in nan_paths:
            writer.writerow([p])

    print(f"Saved {len(nan_paths)} problematic paths to {output_csv}")
    """


    with torch.no_grad():
        for batch in tqdm(loader):

            inputs = batch['img'].to(DEVICE)
            path = batch['path']
            # valid_indices = [i for i, p in enumerate(batch_paths) if p != ""]
            # if not valid_indices:
            #    continue

            #inputs = batch_tensors[valid_indices].to(DEVICE, non_blocking=True)
            #paths = [batch_paths[i] for i in valid_indices]

            # Inferenza (Float32)
            print(inputs.shape)
            _, latents, _ = autoencoder(inputs)  # Output: [Batch, 3, 16, 16, 16]
            print(f"Latents shape: {latents.shape}")

            sq_sum += (latents ** 2).sum().item()
            count += latents.numel()

            # Sposta su CPU
            # latents_cpu = latents.cpu()

            
            # for i, rel_path in enumerate(paths):
            #    dest_path = os.path.join(OUTPUT_ROOT, rel_path.replace("_ct.nii.gz", ".pt"))
            #    dest_dir = os.path.dirname(dest_path)
            #    os.makedirs(dest_dir, exist_ok=True)

                # --- FIX: AGGIUNTA DIMENSIONE BATCH ---
                # latents_cpu[i] è [3, 16, 16, 16]
                # unsqueeze(0) lo fa diventare [1, 3, 16, 16, 16]
            #    tensor_to_save = latents_cpu[i].unsqueeze(0).clone()


            #    torch.save(tensor_to_save, dest_path)
            

    # print("\nFinito! Dataset latente ricreato correttamente in:", OUTPUT_ROOT)

    sigma2 = sq_sum / count
    scale = 1.0 / np.sqrt(sigma2)

    print("=================================")
    print(f"Latent variance (sigma^2): {sigma2:.6f}")
    print(f"Scaling factor: {scale:.6f}")
    print("=================================")
