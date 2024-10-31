# Lightstage Capture 

## Storage
+ LS_BRM01, LS_BRM02 have 10TB data each
+ `/CT/LS_BRM01/static00/FlashingLightsDataset` has raw videos
+ We are extracting data in `/scratch/inf0/user/pgera/FlashingLights` to preprocess. 
+ After that we extract in `/CT/LS_BRM02/static00/FlashingLights`

## Extract frames:
`cd /CT/prithvi/work/studio-tools/LightStage/post-processing`
Add the sequence to be extracted in list_extract.txt. eg: C072 364 2
`sbatch slurm_red.sh`

## Calibration
`cd /CT/prithvi/work/studio-tools/LightStage/calibration`
Update get_calibration_light_stage.py. Follow instructions in README.md
It will save camera.calib camera.xml model.obj

## Undistort
`cd /CT/prithvi/work/studio-tools/LightStage/segmentation`
Follow instructions to setup environment in README.md
`python undistort_exr.py --frame $1 --path /scratch/inf0/user/pgera/FlashingLights/sandhoodie/pose_01/ --calib /CT/prithvi2/nobackup/sandhoodie/pose_01/OLATS/full_light_10`

## Masks
`cd /CT/prithvi/work/studio-tools/LightStage/segmentation`
Follow instructions to setup environment in README.md
Update run_background_matting.py and then `python run_background_matting.py`

## Optical flow align
Compute flows from tracking frames. Assume the middle tracking frame as canonical one
`python flow_align.py --input /scratch/inf0/user/pgera/FlashingLights/sandhoodie/pose_01/ --output /scratch/inf0/user/pgera/FlashingLights/sandhoodie/align/pose_01/`
Apply interpolated flows from tracking frames to remaining images. Only works on 810x1440 res images
`python flow_apply.py --input /scratch/inf0/user/pgera/FlashingLights/sandhoodie/pose_01/ --output /scratch/inf0/user/pgera/FlashingLights/sandhoodie/align/pose_01 --flow /scratch/inf0/user/pgera/FlashingLights/sandhoodie/align/pose_01/flow --start 15 --end 35`
There are some scripts in `/CT/prithvi2/work/scripts/3dgs` to run this on server

```for i in {204..364}; do sbatch gs_light_stage.sh $i; done```

## Create data folder
Save cameras in transforms_train.json along with images. Currently doing everything at 810x1440 resolution.
```
python calib_camera.py \
--calib /CT/prithvi2/nobackup/sandhoodie/pose_01/OLATS/full_light_10/cameras.calib \
--points_in /CT/prithvi2/nobackup/sandhoodie/pose_01/OLATS/full_light_10/points_3d.ply \
--mask_path /CT/prithvi2/nobackup/sandhoodie/pose_01/OLATS/segmentation/sam \
--img_path /scratch/inf0/user/pgera/FlashingLights/sandhoodie/pose_01/L_203/ \     
--ext exr \
--output /CT/LS_BRM02/static00/FlashingLights/sandhoodie/pose_01/L_203 \
--scale 4 \
--obj /CT/prithvi2/nobackup/sandhoodie/pose_01/OLATS/full_light_10/model.obj \
--create_alpha
```
Can add further flags to customize

To run on server
```for i in {204..364}; do
sbatch create_data.sh $i; done```




