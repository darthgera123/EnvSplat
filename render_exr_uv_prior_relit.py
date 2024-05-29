#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, Scene_exr,Scene_uv_prior_relit
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_uv_prior_relit,render_uv_all_sh
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel,GaussianModel_uv_prior_relit
from imageio.v2 import imread,imwrite
import cv2
import numpy as np
import time
def torch2numpy(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np.power(np_img,2.2)
    return np_img

def exr2png(img):
    np_img = np.clip(np.power(np.clip(img,0,None),0.45),0,1)*255
    return np_img.astype('uint8')

def load_image(path):
    image = np.power(np.clip(imread(path),0,None),0.45)
    if image.dtype == 'uint8':
        resized_image = torch.from_numpy(np.array(image)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(image)) 
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1).cuda()
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1).cuda()
def read_uvmap(path):
    
    img = imread(str(path))
    return img

def norm_img(image):
    min_value = 0.0
    max_value = 1.0

    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

def resize_max_pool(image, pool_size=(16, 16)):
    # Assuming image shape is (height, width, channels)
    img_height, img_width, channels = image.shape
    pool_height, pool_width = pool_size
    
    # Define output dimensions
    out_height = img_height // pool_height
    out_width = img_width // pool_width
    
    image_reshaped = image.reshape(out_height, pool_height, out_width, pool_width, channels)
    
    pooled_image = np.max(image_reshaped, axis=(1, 3))
    return pooled_image

def env(path):
    image = imread(path)
    # image = cv2.resize(image, (32,16), interpolation=cv2.INTER_AREA)
    image = resize_max_pool(image,pool_size=(16,16))
    image = norm_img(image)
    if image.dtype == 'uint8':
        resized_image = torch.from_numpy(np.array(image)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(image)) 
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
def l1_loss(image1, image2):
    return np.mean(np.abs(image1 - image2))    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    times = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")): 
        start_time = time.time()
        envmap = env(view.envmap).cuda()
        
        rendering = render_uv_prior_relit(view, gaussians, pipeline, background,envmap=envmap)
        end_time = time.time()
        execution_time = end_time - start_time
        times += execution_time
        
        # gt = view.original_image[0:3, :, :]
        gt = load_image(view.original_image)
        
        np_render = torch2numpy(rendering['render'])
        np_gt = torch2numpy(gt)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"),np_render)
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".exr"),np_gt)
        
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_render))
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_gt))

        diffuse = rendering['diffuse'].detach().cpu().numpy()
        diffuse = np.clip(diffuse,0,1).reshape(256,256,3)
        diffuse = (diffuse*255).astype('uint8')
        relit_diffuse = rendering['relit_diffuse'].detach().cpu().numpy()
        # scale_img = rendering['scales'].detach().cpu().numpy()
        # df_scale_img = rendering['df_scale'].detach().cpu().numpy()
        relit_diffuse = np.clip(relit_diffuse,0,1)
        relit_diffuse = (relit_diffuse*255).astype('uint8')
        # scale_img = (scale_img-scale_img.min())/(scale_img.max()-scale_img.min())
        # scale_img = (scale_img*255).astype('uint8')
        # df_scale_img = (df_scale_img-df_scale_img.min())/(df_scale_img.max()-df_scale_img.min())
        # df_scale_img = (df_scale_img*255).astype('uint8')
        # uvmap = read_uvmap(view.uvmap)
        # diff_col = np.abs(relit_diffuse/255.0-uvmap/255.0)*255
        # print("Col_diff",l1_loss(uvmap/255.0,relit_diffuse/255.0))
        # np_img = np.hstack([diffuse,relit_diffuse,uvmap,diff_col.astype('uint8')])
        # imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_uv.png"),np_img)
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_relit_uv.png"),relit_diffuse)
        # imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_gt_uv.png"),uvmap)

        del envmap,gt,np_gt
    print(f"The function took {times/len(views)} seconds to execute.")
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel_uv_prior_relit(dataset.sh_degree)
        # scene = Scene_uv_prior_relit(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene = Scene_uv_prior_relit(dataset, gaussians, shuffle=False,load_iteration=iteration)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)