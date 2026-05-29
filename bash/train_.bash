#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
# Output files
#SBATCH --error=./error/job_%J.err
#SBATCH --output=./output/out_%J.out
# Mail me1SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francesco.feola@umu.se

# Load modules
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a


# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/venv2
source bin/activate


# Executes the code 
cd /mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/

# Train HERE YOU RUN YOUR PROGRAM

# python ./prepare_dataset.py
# python ./pretrain_VAE.py
python ./save_latents_from_VAE.py
# python ./csv_utils_2.py