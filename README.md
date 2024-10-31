# EnvSplat
Code for 3D Gaussian Splatting. The Gaussian features are predicted using an encoder-decoder network.

Running code for all different files.

# EnvSplat

This repository contains code for 3D Gaussian Splatting, where Gaussian features are predicted using an encoder-decoder network.

## Table of Contents
- [Setup and Prerequisites](#setup-and-prerequisites)
- [Training and Evaluation](#training-and-evaluation)
  - [Training with SH Decoder](#training-with-sh-decoder)
  - [Training with UV Latent Priors](#training-with-uv-latent-priors)
  - [Training with UV Relit](#training-with-uv-relit)
  - [Training with Simple UV Prior](#training-with-simple-uv-prior)
- [Rendering](#rendering)

## Setup and Prerequisites

Ensure you have the required packages installed and set up paths accordingly.

### Example: Running the Code
Run any of the scripts below with the appropriate arguments:
```bash
python <script_name.py> --source_path <source_path> --model_path <model_path> --eval --iteration <iterations> --sh_degree <degree> --additional_parameters





```
python train_exr_sh.py --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/full_light/ --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_sh_decoder_v3 --eval --iteration 50000 --decoder_lr 1e-3 --sh_degree 3

python train_exr_uv_prior_latent_all.py \
        --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_32_unet_uv \
        --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_32_latent_percep \
        --eval \
        --iteration 100000 \
        --sh_degree 3 \
        --encoder_lr 1e-4 \
        --load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
        --relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv_unet/envmaps_final/latent-all-no_act-32-full/checkpoint/model_599.pth

python train_exr_uv_prior_latent.py \    
--source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_192_unet_uv \
--model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_192_latent \
--eval \
--iteration 50000 \
--sh_degree 3 \
--encoder_lr 1e-4 \
--load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
--relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv_unet/envmaps/latent-color-rgb-192/checkpoint/model_499.pth

python train_exr_uv_prior.py \           
        --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_32_unet_uv \
        --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_32_uvrelit_percep \
        --eval \
        --iteration 100000 \
        --sh_degree 3 \
        --encoder_lr 1e-4 \
        --load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
        --relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv_unet/envmaps_final/uvrelit-all-no_act-32-full/checkpoint/model_599.pth

python train_exr_uv_prior_relit.py \     
--source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_192_unet_uv \
--model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_192_uvrelit_col_test \
--eval \
--iteration 50000 \
--sh_degree 3 \
--encoder_lr 1e-4 \
--load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
--relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv_unet/envmaps/uvrelit-color-rgb-192/checkpoint/model_499.pth

python train_exr_uv_prior_simple.py \
        --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_32_unet_uv \
        --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_32_unet_simple_vgg1_col1 \          
        --eval \
        --iteration 100000 \
        --sh_degree 3 \
        --encoder_lr 1e-4 \
        --load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
        --relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv_unet/envmaps_final/unet_simple-all-no_act-704/checkpoint/model_599.pth \
--vgg 1 \
--color 1

python train_exr_uv.py \             
--source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_10_nvs \
--model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_10_single_unet_load_enc \
--eval \
--iteration 50000 \
--sh_degree 3 \
--encoder_lr 5e-4 \
--load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
--relit_path /scratch/inf0/user/pgera/FlashingLights/3dgs_uv/envmaps/uvrelit-all-rgb-300-max-coeff/checkpoint/model_399.pth

python train_exr_uv_relit_view.py \     
--source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_1 \
--model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_1_single_unet_load_enc1 \
--eval \
--iteration 50000 \
--sh_degree 3 \
--decoder_lr 1e-3 \
--encoder_lr 1e-3 \
--load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_sh_single_unet_59ch_l1_percep_1e4_mask

python render_encoder_sh_uv.py \
--source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/full_light_mask \
--model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_sh_single_unet_59ch_l1_percep_1e4_allscale_l2_1e3_light_append \
--eval \
--sh_degree 3

python render_exr_uv_prior_simple.py \
        --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/$light \
        --model_path /scratch/inf0/user/pgera/FlashingLights/3dgs/envmap_mask_max/relight/sunrise_pullover/pose_01/unet_simple/$light \ 
        --eval \
        --iteration 100000 \
        --sh_degree 3 \
        --load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
        --relit_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_352_unet_simple_vgg1e1/point_cloud/iteration_100000/relit.pth  \
--skip_train \
--gt_geometry /scratch/inf0/user/pgera/FlashingLights/3dgs/envmap_mask_max/output/sunrise_pullover/pose_01/$light

python render_exr_uv_prior_latent_all.py \
        --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/envmap_1 \
        --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_1_uv \
        --eval \
        --iteration 100000 \
        --sh_degree 3 \
        --load_geometry /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_3dgs_mask_test_max/ \
        --relit_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/envmap_352_latent_vgg1_col1/point_cloud/iteration_100000/relit.pth  \
--skip_train
```