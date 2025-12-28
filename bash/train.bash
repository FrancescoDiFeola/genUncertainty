#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 7-00:00:00
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
cd /mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion

# Train HERE YOU RUN YOUR PROGRAM


# python3 ./autoencoder_doppio_freeze.py --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_doppio_freeze_VT_no_align" --experiment_name "autoKL_doppio_freeze_VT_no_align" --dataset_csv "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/virtual_treatment_NLST.csv" --aekl_diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_VT/autoencoder-ep-350.pth"  # --compute_metrics
# python3 ./test_autoKL.py --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_doppio_freeze_VT" --experiment_name "autoKL_doppio_freeze_VT" --aekl_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_doppio_freeze_VT/autoencoder_init-ep-300.pth" --epoch_checkpoint 300 --compute_metrics # --analyze_distribution
# python3 ./train_autoencoderKL.py --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_VT_attention" --experiment_name "autoKL_VT_attention" --dataset_csv "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/virtual_treatment_NLST.csv" # --compute_metrics
# python3 ./train_ddpm_aleatoric.py --num_workers 32 --epoch_start 450 --experiment_name "aleatoric_uncertainty_b16_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_b16_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_b16_T1T2/diffusion-ep-450.pth"
# python3 ./train_ddpm.py --num_workers 32 --epoch_start 300 --experiment_name "ddpm_b16_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2/diffusion-ep-300.pth"
# python3 ./test_ddpm_aleatoric.py  --num_workers 32 --experiment_name "aleatoric_uncertainty_regularized_continue" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_regularized_continue" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_regularized_continue/diffusion-ep-150.pth"
# python3 ./test_ddpm.py  --num_workers 32 --experiment_name "ddpm_continue" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_continue" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_continue/diffusion-ep-150.pth"
# python3 ./test_ddpm_with_refiner.py  --num_workers 32 --experiment_name "ddpm_with_joint_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/diffusion-ep-450.pth" --refiner_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/joint_refiner-ep-450.pth"


# python3 ./test_ddpm.py  --num_workers 32 --experiment_name "ddpm_b16_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2/diffusion-ep-500.0.pth"

# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 32 --experiment_name "aleatoric_uncertainty_cross_attention_denoising" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_cross_attention_denoising" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_cross_attention_denoising/diffusion-ep-550.pth"

# python3 ./train_ddpm_with_refiner.py --num_workers 32 --experiment_name "ddpm_with_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_refiner_T1T2"  # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_denoising/diffusion-ep-150.pth"

# python3 ./test_ddpm_aleatoric_epistemic.py --num_workers 32 --experiment_name "aleatoric_epistemic_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-300.pth"

# python3 ./train_ddpm_with_refiner.py --num_workers 32 --experiment_name "ddpm_with_joint_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2"  # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-100.pth" # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-100.pth"

# python3 ./train_ddpm_with_double_refiner.py --num_workers 32 --experiment_name "ddpm_with_double_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2"  # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_denoising/diffusion-ep-150.pth"
# python3 ./test_ddpm_with_double_refiner.py  --num_workers 32 --experiment_name "ddpm_with_double_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/diffusion-ep-550.pth" --refiner1_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/error_refiner-ep-550.pth" --refiner2_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/variance_refiner-ep-550.pth"

# python3 ./train_RF.py --num_workers 8 --experiment_name "RF_denoising"
# python3 ./train_RF_v2.py --num_workers 8 --experiment_name "RF_T1T2_v2"
# python3 ./train_RF_aleatoric.py --num_workers 8 --experiment_name "RF_denoising_aleatoric"
# python3 ./train_RF_aleatoric_v2.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric_v2"
# python3 ./train_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_denoising_aleatoric_two_forward" --spatial_enc_channels 1

# python3 ./test_RF_aleatoric_two_forward.py  --spatial_enc_channels 1 --num_workers 8 --experiment_name "RF_T1T2_aleatoric_two_forward"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2_aleatoric_two_forward/diffusion-ep-900.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2_aleatoric_two_forward/spatial_encoder-ep-900.pth"
# python3 ./test_RF_aleatoric.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2_aleatoric/diffusion-ep-900.pth"
# python3 ./test_RF.py --num_workers 8 --experiment_name "RF_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2/diffusion-ep-900.pth"

# python3 ./train_ddpm.py --num_workers 8 --batch_size 64 --experiment_name "ddpm_CTPET"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET

# python3 ./train_ddpm_aleatoric.py --num_workers 8 --experiment_name "ddpm_CTPET_aleatoric"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET
# python3 ./train_ddpm_aleatoric_two_forward.py --num_workers 8 --experiment_name "ddpm_CTPET_aleatoric_two_forward"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET

# python3 ./test_ddpm_aleatoric_two_forward.py  --spatial_enc_channels 1 --num_workers 8 --experiment_name "two_forward_variance_normalized_T1T2"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/diffusion-ep-300.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/spatial_encoder-ep-300.pth"




# Distributed launch (DDP)
# torchrun --nproc_per_node=4 train_ddpm_optimized.py \
#  --num_workers 0 \
#  --batch_size 16 \
#  --grad_accum_steps 1 \
#  --cache_rate 1.0 \
#  --log_every 50 \
#  --experiment_name "ddpm_CTPET_A40x4_bs128" \
#  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" \
#  --phase train \
#  --slice_range 0 10000 \
#  --mri_modalities CT PET