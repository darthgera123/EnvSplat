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
from scene import Scene, Scene_exr,Scene_exr_decoder_sh
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_decoder_sh
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel,GaussianModel_exr_decoder_sh
from imageio.v2 import imread,imwrite
import numpy as np
import time
from utils.sh_utils import SH2RGB
import cv2
def torch2numpy(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np.power(np_img,2.2)
    return np_img

def exr2png(img):
    np_img = np.clip(np.power(np.clip(img,0,None),0.45),0,1)*255
    return np_img.astype('uint8')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    cyan = torch.tensor([[0,1,1]]).cuda().reshape((3,1,1))
    yellow = torch.tensor([[1,1,0]]).cuda().reshape((3,1,1))
    times = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")): 
        start_time = time.time()
        camcenter = view.camera_center.reshape((1,1,3))
        camcenter = camcenter/(camcenter.norm(dim=2,keepdim=True))
        
        full_render = render_decoder_sh(view, gaussians, pipeline, background)
        rendering = full_render['render']
        end_time = time.time()
        execution_time = end_time - start_time
        times += execution_time
        
        gt = view.original_image[0:3, :, :]
        np_render = torch2numpy(rendering)
        np_gt = torch2numpy(gt)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"),np_render)
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".exr"),np_gt)
        
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_render))
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_gt))
    diffuse = full_render['diffuse'].detach().cpu().numpy()
    diffuse = np.clip(SH2RGB(diffuse),0,1).reshape(256,256,3)
    diffuse = (diffuse*255).astype('uint8')
    relit = full_render['relit'].detach().cpu().numpy()
    relit = np.clip(SH2RGB(relit),0,1).reshape(256,256,3)
    relit = (relit*255).astype('uint8')
    opacity = full_render['opacity'].detach().cpu().numpy()
    pos = full_render['pos'].detach().cpu().numpy()
    pos = (pos - pos.min()) / (pos.max() - pos.min())
    np_img = np.hstack([relit,diffuse])
    imwrite(os.path.join(render_path, "relit.png"),np_img)
    cv2.imwrite(os.path.join(render_path, "opacity.png"),(opacity*255).astype('uint8'))
    imwrite(os.path.join(render_path, "pos.png"),(pos*255).astype('uint8'))
    print(f"The function took {times/len(views)} seconds to execute.")
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel_exr_decoder_sh(dataset.sh_degree)
        scene = Scene_exr_decoder_sh(dataset, gaussians, load_iteration=iteration, shuffle=False)

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