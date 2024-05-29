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

import os
import torch
from imageio.v2 import imread,imwrite
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_exp,PerceptualLoss,IDMRFLoss
from gaussian_renderer import render_uv_all_sh
import sys
from scene import Scene_uv_sh,GaussianModel_uv_sh
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import cv2
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import SH2RGB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def tonemap(img):
    return torch.pow(img+1e-5,0.45)

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
def norm_img(image):
    min_value = 0.0
    max_value = 1.0

    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

def env(path):
    image = imread(path)
    image = cv2.resize(image, (32,16), interpolation=cv2.INTER_AREA)
    image = norm_img(image)
    if image.dtype == 'uint8':
        resized_image = torch.from_numpy(np.array(image)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(image)) 
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel_uv_sh(dataset.sh_degree)
    
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    
    
    scene = Scene_uv_sh(dataset, gaussians)
        
        
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    percep = PerceptualLoss()
    # idmrf = IDMRFLoss()
    
    
    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 5000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        envmap = env(viewpoint_cam.envmap).cuda()
        
        render_pkg = render_uv_all_sh(viewpoint_cam, gaussians, pipe, background,\
                                    scaling_modifier=1.0,envmap=envmap)
        image = render_pkg["render"]
        
        del_scale = render_pkg["scaly"].reshape(-1,3)
        del_pos = render_pkg["pos"].reshape(-1,3)
        # scales_norm = torch.norm(del_scale, dim=1, keepdim=True).mean()   
        # l2_reg_output=scales_norm.max()
        l2_reg_output = torch.norm(del_scale, dim=1, keepdim=True).mean()   
        
        l1_pos = torch.norm(del_pos,p=1,dim=1).mean() 
        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        gt_image = load_image(viewpoint_cam.original_image)
        
        
        # Ll1 = l1_loss(tonemap(image), tonemap(gt_image))
        image = torch.clamp(image,1e-6,1)
        Ll1 = l1_loss(image, gt_image)
        # print("NaN check", torch.isnan(image).any())
        Ll2 = percep(tonemap(image),tonemap(gt_image))
        
        loss = loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
        # if iteration > 10000:
        #     Ll2 = idmrf(image,gt_image)
            
        loss += 0.01*Ll2 
        loss += 0.001*l2_reg_output
        loss += 0.01*l1_pos
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_uv_all_sh, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        del gt_image,envmap

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def torch2numpy(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np.power(np_img,2.2)
    return np_img
def exr2png(img):
    np_img = np.clip(np.power(np.clip(img,0,None),0.45),0,1)*255
    return np_img.astype('uint8')


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene , renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        outputs = os.path.join(args.model_path,'renders/')
        os.makedirs(outputs,exist_ok=True)    
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    view = viewpoint.camera_center.reshape((1,1,3))
                    view=view/(view.norm(dim=2, keepdim=True))
        
                    
                    envmap = env(viewpoint.envmap).to("cuda")
                    rendering = renderFunc(viewpoint, scene.gaussians,\
                                                    *renderArgs,envmap=envmap)
                    image = torch.clamp(rendering['render'],0,1)
                    
                    
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image = torch.clamp(load_image(viewpoint.original_image).to("cuda"), 0.0, 1.0)
                    
                    if idx == 2 and config['name'] == 'test':
                        np_img = exr2png(torch2numpy(image))
                        np_gt = exr2png(torch2numpy(gt_image))
                        np_result = np.hstack((np_img,np_gt))
                        diffuse = rendering['diffuse'].detach().cpu().numpy()
                        diffuse = np.clip(SH2RGB(diffuse),0,1).reshape(256,256,3)
                        diffuse = (diffuse*255).astype('uint8')
                        relit_diffuse = rendering['relit_diffuse'].detach().cpu().numpy()
                        scala = rendering['scala'].detach().cpu().numpy()
                        scaly = rendering['scaly'].detach().cpu().numpy()
                        pos = rendering['pos'].detach().cpu().numpy()
                        print("scala",scala.min(),scala.max())
                        print("del Scale",scaly.min(),scaly.max())
                        print("Pos",pos.min(),pos.max())
                        if np.isnan(relit_diffuse).any():
                            roty = rendering['roty'].detach().cpu().numpy()
                            roty = (roty - roty.min()) / (roty.max() - roty.min())
                            scaly = rendering['scaly'].detach().cpu().numpy()
                            scaly = (scaly - scaly.min()) / (scaly.max() - scaly.min())
                            pos = rendering['pos'].detach().cpu().numpy()
                            pos = (pos - pos.min()) / (pos.max() - pos.min())
                            print("Nan check color",np.isnan(relit_diffuse).any())
                            print("Nan check pos",np.isnan(pos).any())
                            print("Nan check scaly",np.isnan(scaly).any())
                            exit()
                        
                        relit_diffuse = np.clip(SH2RGB(relit_diffuse),0,1)
                        relit_diffuse = (relit_diffuse*255).astype('uint8')
                        masky = rendering['masky'].detach().cpu().numpy()
                        masky = (masky*255).astype('uint8')
                        pos = rendering['pos'].detach().cpu().numpy()
                        pos = (pos - pos.min()) / (pos.max() - pos.min())*255
                        np_img = np.hstack([pos,relit_diffuse,diffuse])
                        imwrite(f'{outputs}/output_{iteration}.png',np_result.astype('uint8'))
                        imwrite(f'{outputs}/uv_{iteration}.png',np_img.astype('uint8'))
                        
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    del gt_image,envmap
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(100,500_000,5000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(100,500_000,50_000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--load_geometry", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
