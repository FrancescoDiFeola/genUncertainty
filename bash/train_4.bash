#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 7-0:00:00
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
# python3 ./test_ddpm_aleatoric.py  --num_workers 8 --experiment_name "aleatoric_uncertainty_b16_T1T2" --epoch "300" --task "T1T2"

# python3 ./test_ddpm_with_refiner.py  --num_workers 32 --experiment_name "ddpm_with_joint_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/diffusion-ep-450.pth" --refiner_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/joint_refiner-ep-450.pth"


# python3 ./test_ddpm.py  --num_workers 8 --experiment_name "ddpm_b16_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2/diffusion-ep-200.pth"

# python3 ./train_ddpm.py --num_workers 8 --experiment_name "ddpm" --task "T1T2_Oasis" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_ddpm_aleatoric.py --num_workers 8 --experiment_name "ddpm_aleatoric" --task "T1T2_Oasis" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_ddpm_aleatoric_two_forward.py  --num_workers 8 --experiment_name "aleatoric_two_forward" --task "T1T2_Oasis" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset




# python3 ./train_ddpm_with_refiner.py --num_workers 32 --experiment_name "ddpm_with_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_refiner_T1T2"  # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_denoising/diffusion-ep-150.pth"

# python3 ./test_ddpm_aleatoric_epistemic.py --num_workers 32 --experiment_name "aleatoric_epistemic_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-300.pth"

# python3 ./train_ddpm_with_refiner.py --num_workers 32 --experiment_name "ddpm_with_joint_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2"  # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-100.pth" # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-100.pth"

# python3 ./train_ddpm_with_double_refiner.py --num_workers 32 --experiment_name "ddpm_with_double_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2"  # --annotation_A "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set_segmented_lungs/slices/ct_pet_slice_paths.csv" # --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_denoising/diffusion-ep-150.pth"
# python3 ./test_ddpm_with_double_refiner.py  --num_workers 32 --experiment_name "ddpm_with_double_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/diffusion-ep-550.pth" --refiner1_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/error_refiner-ep-550.pth" --refiner2_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/variance_refiner-ep-550.pth"

# python3 ./train_RF.py --num_workers 8 --experiment_name "RF" --task "T1T2_Oasis" --in_ch 2 --out_ch 1 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_RF_v2.py --num_workers 8 --experiment_name "RF_T1T2_v2"
# python3 ./train_RF_aleatoric.py --num_workers 8 --experiment_name "RF_aleatoric" --task "T1T2_Oasis" --in_ch 2 --out_ch 1 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_RF_aleatoric_v2.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric_v2"
# python3 ./train_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_aleatoric_two_forward" --spatial_enc_channels 1 --task "T1T2_Oasis" --in_ch 2 --out_ch 1 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset

# python3 ./train_LFM.py --num_workers 8 --experiment_name "LFM_CTPET" --task "CTPET" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices"  --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_LFM_aleatoric.py --num_workers 8 --experiment_name "LFM_aleatoric_denoising" --task "denoising" --in_ch 6 --out_ch 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices"  --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_LFM_aleatoric_two_forward.py --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_CTPET" --spatial_enc_channels 3 --task "CTPET" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices"  --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_RF_v2.py --num_workers 8 --experiment_name "RF_T1T2_v2"
# python3 ./train_RF_aleatoric.py --num_workers 8 --experiment_name "RF_aleatoric_MRtoCT" --task "MRtoCT" --in_ch 2 --out_ch 1
# python3 ./train_RF_aleatoric_v2.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric_v2"
# python3 ./train_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_aleatoric_two_forward_denoising" --spatial_enc_channels 1 --task "denoising" --in_ch 2 --out_ch 1

# python3 ./test_RF_aleatoric_two_forward.py  --spatial_enc_channels 1 --num_workers 8 --experiment_name "RF_denoising_aleatoric_two_forward"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_denoising_aleatoric_two_forward/diffusion-ep-300.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_denoising_aleatoric_two_forward/spatial_encoder-ep-300.pth"
# python3 ./test_RF_aleatoric.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2_aleatoric/diffusion-ep-100.pth"
# python3 ./test_RF.py --num_workers 8 --experiment_name "RF_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2/diffusion-ep-100.pth"


# python3 ./train_ddpm.py --num_workers 8 --batch_size 16 --experiment_name "ddpm_MRtoCT" --task "MRtoCT" --in_ch 2 --out_ch 1
# python3 ./train_ddpm_aleatoric.py --num_workers 8 --batch_size 16 --experiment_name "ddpm_aleatoric_MRtoCT" --task "MRtoCT" --in_ch 2 --out_ch 1
# python3 ./train_ddpm_aleatoric_two_forward.py --num_workers 8 --batch_size 16 --task "MRtoCT" --spatial_enc_channels 1 --in_ch 2 --out_ch 1 --experiment_name "ddpm_aleatoric_two_forward_MRtoCT"

# python3 ./train_ddpm.py --num_workers 8 --batch_size 16 --experiment_name "ddpm_CTPET" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_ddpm_aleatoric.py --num_workers 8 --batch_size 16 --backbone "UNet" --task "CTPET" --experiment_name "ddpm_aleatoric"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_ddpm_aleatoric_two_forward.py --num_workers 8 --epoch_start 100 --experiment_name "ddpm_CTPET_aleatoric_two_forward" --spatial_enc_channels 1 --task "CTPET" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./test_ddpm_aleatoric_two_forward.py  --spatial_enc_channels 1 --num_workers 8 --experiment_name "two_forward_variance_normalized_T1T2"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/diffusion-ep-300.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/spatial_encoder-ep-300.pth"

# ND dataset
# python3 ./train_ddpm.py --num_workers 8 --batch_size 16 --task "ND" --experiment_name "ddpm_ND_dataset" --in_ch 6 --out_ch 3
# python3 ./train_ddpm_aleatoric.py --num_workers 8 --batch_size 16 --task "ND" --experiment_name "ddpm_aleatoric_ND_dataset" --in_ch 6 --out_ch 3
# python3 ./train_ddpm_aleatoric_two_forward.py --num_workers 8 --batch_size 16 --task "ND" --spatial_enc_channels 3 --experiment_name "ddpm_aleatoric_two_forward_ND_dataset" --in_ch 6 --out_ch 3



# python3 ./train_UViT.py --num_workers 8 --batch_size 16 --task "denoising" --experiment_name "UViT_denoising" --in_ch 2 --out_ch 1
# python3 ./train_ddpm_aleatoric.py --num_workers 8 --batch_size 16 --experiment_name "UViT_CTPET_aleatoric"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./train_ddpm_aleatoric_two_forward.py --num_workers 8 --experiment_name "ddpm_CTPET_aleatoric_two_forward"  --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/ddpm_CTPET_aleatoric_two_forward/diffusion-ep-100.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/ddpm_CTPET_aleatoric_two_forward/spatial_encoder-ep-100.pth"

# python3 ./train_ddpm_aleatoric.py --num_workers 8 --batch_size 16 --backbone "UViT" --task "denoising" --experiment_name "UViT_denoising_aleatoric" -in_ch 2 --out_ch 1



# python3 ./train_RF.py --num_workers 8  --batch_size 16  --task "ND" --experiment_name "RF_ND_dataset" --in_ch 6 --out_ch 3
# python3 ./train_RF_aleatoric.py --num_workers 8 --batch_size 16  --task "ND" --experiment_name "RF_ND_aleatoric"  --in_ch 6 --out_ch 3
# python3 ./train_RF_aleatoric_two_forward.py --num_workers 8 --batch_size 16 --epoch_start 100  --task "CTPET" --experiment_name "RF_aleatoric_two_forward_CTPET" --spatial_enc_channels 1 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

############ LDM ###################

python3 ./train_LDM.py --num_workers 8 --experiment_name "LDM_MRCT" --task "MRtoCT" --in_ch 6 --out_ch 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_LDM_aleatoric.py --num_workers 8 --experiment_name "LDM_aleatoric_T1T2" --task "T1T2_Oasis" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_LDM_aleatoric_two_forward.py --num_workers 8 --experiment_name "LDM_aleatoric_two_forward" --spatial_enc_channels 3 --task "T1T2_Oasis" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset

####################################

############ LFM ###################
# python3 ./train_LFM.py --num_workers 8 --experiment_name "LFM_MRtoCT" --task "MRtoCT" --in_ch 6 --out_ch 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_LFM_aleatoric.py --num_workers 8 --experiment_name "LFM_aleatoric_MRCT" --task "MRtoCT" --in_ch 6 --out_ch 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./train_LFM_aleatoric_two_forward.py --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_T1T2" --spatial_enc_channels 3 --task "T1T2_Oasis" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
####################################

# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices"  --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_LDM.py  --num_workers 8 --experiment_name "LDM_T1T2" --task "T1T2" --epoch "300" --in_ch 6 --out_ch 3
# python3 ./test_LDM_aleatoric.py  --num_workers 8 --experiment_name "LDM_aleatoric_denoising" --task "denoising" --epoch "50" --in_ch 6 --out_ch 3
# python3 ./test_LDM_aleatoric_two_forward.py  --num_workers 8 --experiment_name "LDM_aleatoric_two_forward_denoising" --task "denoising" --epoch "50" --in_ch 6 --out_ch 3 --spatial_enc_channels 3


# python3 ./utils/prepare_dataset.py