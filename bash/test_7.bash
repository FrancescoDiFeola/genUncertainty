#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 2-00:00:00
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


# python3 ./test_autoKL.py --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_doppio_freeze_VT" --experiment_name "autoKL_doppio_freeze_VT" --aekl_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/checkpoints/autoKL_doppio_freeze_VT/autoencoder_init-ep-300.pth" --epoch_checkpoint 300 --compute_metrics # --analyze_distribution

# python3 ./test_ddpm_aleatoric.py  --num_workers 8 --experiment_name "aleatoric_uncertainty_b16_T1T2" --epoch "300" --task "T1T2"

# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 8 --experiment_name "ddpm_aleatoric_two_forward_MRtoCT" --epoch "300" --task "MRtoCT" --analysis "sparsification" --spatial_enc_channels 1
#--dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_ddpm.py  --num_workers 8 --experiment_name "ddpm_b16_denoising" --epoch "300" --task "denoising" --analysis "sparsification" --MC_sampling

# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_ddpm_with_refiner.py  --num_workers 32 --experiment_name "ddpm_with_joint_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/diffusion-ep-450.pth" --refiner_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_joint_refiner_T1T2/joint_refiner-ep-450.pth"






# python3 ./test_ddpm.py  --num_workers 8 --experiment_name "ddpm_b16_denoising"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_b16_T1T2/diffusion-ep-200.pth"
# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 32 --experiment_name "aleatoric_uncertainty_cross_attention_denoising" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_cross_attention_denoising" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_uncertainty_cross_attention_denoising/diffusion-ep-550.pth"


# python3 ./test_ddpm_aleatoric_epistemic.py --num_workers 32 --experiment_name "aleatoric_epistemic_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/aleatoric_epistemic_T1T2/diffusion-ep-300.pth"

# python3 ./test_ddpm_with_double_refiner.py  --num_workers 32 --experiment_name "ddpm_with_double_refiner_T1T2" --output_dir "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/diffusion-ep-550.pth" --refiner1_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/error_refiner-ep-550.pth" --refiner2_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/uncertainty_diffusion/checkpoints/ddpm_with_double_refiner_T1T2/variance_refiner-ep-550.pth"


###################################################### Metrics ANALYSIS ###########################################################################################

# python3 ./test_ddpm.py  --num_workers 8 --experiment_name "ddpm" --epoch "280" --task "T1T2_Oasis" --analysis "metrics_no_uncertainty" --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase test --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 8 --experiment_name "aleatoric_two_forward" --epoch "40" --task "T1T2_Oasis" --analysis "both" --spatial_enc_channels 1 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase test --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset

python3 ./test_RF.py --num_workers 8 --experiment_name "RF" --epoch "300" --task "CBCTtoCT" --analysis "metrics_no_uncertainty" # --analysis "sparsification" --MC_sampling
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase test --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset
# python3 ./test_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_aleatoric_two_forward_MRtoCT" --task "MRtoCT" --spatial_enc_channels 1 --epoch "300" --analysis "both"
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase train --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset

# python3 ./test_LDM.py  --num_workers 8 --experiment_name "LDM_MRCT" --task "MRtoCT" --epoch "300" --in_ch 6 --out_ch 3 --analysis "metrics_no_uncertainty"
# python3 ./test_LDM_aleatoric_two_forward.py  --num_workers 8 --experiment_name "LDM_aleatoric_two_forward" --task "MRtoCT" --analysis "metrics" --epoch "300" --spatial_enc_channels 3 --in_ch 6 --out_ch 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase test --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset


# python3 ./test_LFM.py  --num_workers 8 --experiment_name "LFM_MRtoCT" --task "MRtoCT" --epoch "300" --in_ch 6 --out_ch 3 --analysis "metrics_no_uncertainty"
# python3 ./test_LFM_two_forward.py  --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_MRCT" --task "MRtoCT" --epoch "300" --in_ch 6 --out_ch 3 --spatial_enc_channels 3 --analysis "metrics"
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/OASIS-3_filtered_slices"  --phase test --slice_range 0 10000 --mri_modalities t1n t2w --under_sample_dataset

###########################################################################################################################################################################


###################################################### SPARSIFICATION ANALYSIS ###########################################################################################

# python3 ./test_ddpm.py  --num_workers 8 --experiment_name "ddpm_b16_denoising" --epoch "300" --task "denoising" --analysis "sparsification" --MC_sampling
# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 8 --experiment_name "ddpm_aleatoric_two_forward_MRtoCT" --epoch "300" --task "MRtoCT" --analysis "sparsification" --spatial_enc_channels 1

# python3 ./test_RF.py --num_workers 8 --experiment_name "RF_denoising" --epoch "50" --task "denoising" --analysis "sparsification" --MC_sampling
# python3 ./test_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_aleatoric_two_forward_denoising" --task "denoising" --spatial_enc_channels 1 --epoch "50" --analysis "sparsification"

# python3 ./test_LDM.py  --num_workers 8 --experiment_name "LDM_denoising" --task "denoising" --epoch "50" --in_ch 6 --out_ch 3 --analysis "sparsification" --MC_sampling
# python3 ./test_LDM_aleatoric_two_forward.py  --num_workers 8 --experiment_name "LDM_aleatoric_two_forward_denoising" --task "denoising" --epoch "50" --spatial_enc_channels 3 --in_ch 6 --out_ch 3 --analysis "sparsification"

# python3 ./test_LFM.py  --num_workers 8 --experiment_name "LFM_denoising" --task "denoising" --epoch "250" --in_ch 6 --out_ch 3 --analysis "sparsification" --MC_sampling
# python3 ./test_LFM_two_forward.py  --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_denoising" --task "denoising" --epoch "250" --in_ch 6 --out_ch 3 --spatial_enc_channels 3 --analysis "sparsification"
###########################################################################################################################################################################

################################################### our w/o (IR ^ SC) #####################################################################################################

# python3 ./test_ddpm_aleatoric.py  --num_workers 8 --experiment_name "aleatoric_uncertainty_b16_denoising" --epoch "300" --task "denoising" --analysis "both"
# python3 ./test_RF_aleatoric.py --num_workers 8 --experiment_name "RF_denoising_aleatoric" --epoch "50" --task "denoising" --analysis "both"

# python3 ./test_LDM_aleatoric.py  --num_workers 8 --experiment_name "LDM_aleatoric_denoising" --task "denoising" --epoch "50" --in_ch 6 --out_ch 3 --analysis "metrics"
# python3 ./test_LFM_aleatoric.py  --num_workers 8 --experiment_name "LFM_aleatoric_denoising" --task "denoising" --epoch "250" --in_ch 6 --out_ch 3 --analysis "metrics"

###########################################################################################################################################################################

################################################### our w/o IR ############################################################################################################

# python3 ./test_ddpm_aleatoric_two_forward.py  --num_workers 8 --experiment_name "two_forward_variance_normalized_denoising" --epoch "300" --task "denoising" --spatial_enc_channels 1 --analysis "both" --ablation
# python3 ./test_RF_aleatoric_two_forward.py --num_workers 8 --experiment_name "RF_aleatoric_two_forward_denoising" --epoch "50" --task "denoising" --spatial_enc_channels 1 --analysis "both" --ablation

# python3 ./test_LDM_aleatoric_two_forward.py  --num_workers 8 --experiment_name "LDM_aleatoric_two_forward_denoising" --task "denoising" --epoch "50" --in_ch 6 --out_ch 3 --spatial_enc_channels 3 --analysis "metrics" --ablation
# python3 ./test_LFM_two_forward.py  --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_denoising" --task "denoising" --epoch "250" --in_ch 6 --out_ch 3 --spatial_enc_channels 3 --analysis "metrics" --ablation

###########################################################################################################################################################################





# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase train --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./test_RF_aleatoric.py --num_workers 8 --experiment_name "RF_T1T2_aleatoric" --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/RF_T1T2_aleatoric/diffusion-ep-100.pth"

# python3 ./test_RF.py --num_workers 8 --experiment_name "RF_MRtoCT" --epoch "50" --task "MRtoCT"
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_ddpm_aleatoric_two_forward.py  --spatial_enc_channels 1 --num_workers 8 --experiment_name "two_forward_variance_normalized_T1T2"  --diff_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/diffusion-ep-300.pth" --context_ckpt "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/generative_uncertainty/checkpoints/two_forward_variance_normalized_T1T2/spatial_encoder-ep-300.pth"

# python3 ./test_LDM.py  --num_workers 8 --experiment_name "LDM_CTPET" --task "CTPET" --epoch "300" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset
# python3 ./test_LDM_aleatoric.py  --num_workers 8 --experiment_name "LDM_aleatoric_T1T2" --task "T1T2" --epoch "300" --in_ch 6 --out_ch 3
# python3 ./test_LDM_aleatoric_two_forward.py  --num_workers 8 --experiment_name "LDM_aleatoric_two_forward_CTPET" --task "CTPET" --epoch "50" --spatial_enc_channels 3 --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_LDM.py  --num_workers 8 --experiment_name "LDM_T1T2" --task "T1T2" --epoch "50" --in_ch 6 --out_ch 3
# python3 ./test_LDM_aleatoric.py  --num_workers 8 --experiment_name "LDM_aleatoric_T1T2" --task "T1T2" --epoch "300" --in_ch 6 --out_ch 3
# python3 ./test_LFM_two_forward.py  --num_workers 8 --experiment_name "LFM_aleatoric_two_forward_T1T2" --task "T1T2" --epoch "50" --in_ch 6 --out_ch 3 --spatial_enc_channels 3
# --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./test_LFM.py  --num_workers 8 --experiment_name "LFM_CTPET" --task "CTPET" --epoch "300" --in_ch 6 --out_ch 3 --dataroot "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/FDG-PEt_CT-lesion/lung_slices" --phase test --slice_range 0 10000 --mri_modalities CT PET --under_sample_dataset

# python3 ./utils/prepare_dataset.py