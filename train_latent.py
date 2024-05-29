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
"""
ONLY RGB DECODER
"""
import os
import torch
from imageio.v2 import imread,imwrite
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_exp,PerceptualLoss, IDMRFLoss
from gaussian_renderer import render_latent
import sys
from scene import Scene_latent,GaussianModel_latent
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import lpips
try:

    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def tonemap(img):
    return torch.pow(img+1e-5,0.45)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel_latent(dataset.sh_degree)
    scene = Scene_latent(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # percep = PerceptualLoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    
    
    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        
        
        render_pkg = render_latent(viewpoint_cam, gaussians, pipe, background,\
                                    scaling_modifier=1.0)
        image = render_pkg["render"]
        del_pos = render_pkg["position"]
        scale = render_pkg["scales"]
        rot = render_pkg["rotation"]
        # l2_reg_output = torch.norm(scale.reshape(-1,3), dim=1, keepdim=True).max()
        l2_reg_output = torch.norm(scale.reshape(-1,3), dim=1, keepdim=True).max()
        l1_pos = torch.norm(del_pos.reshape(-1,3),p=1,dim=1).mean()
        l1_rot = torch.norm(rot.reshape(-1,4), dim=1, keepdim=True).mean()
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        image_percep = 2*image-1
        gt_percep = 2*gt_image - 1
        
        Ll1 = l1_loss(image, gt_image)
        Ll2 = loss_fn_vgg.forward(image_percep,gt_percep)
        percep_loss = Ll2.mean().squeeze()
        # Ll3 = idmrf(tonemap(image),tonemap(gt_image))
        # Ll1 = l1_loss_exp(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(tonemap(image),tonemap(gt_image))) 
        loss += opt.vgg*percep_loss
        # loss += 0.01*Ll3
        # loss += 1e-4 * l2_reg_output
        # loss += 1e-2*l1_pos
        # loss += 1e-2*l1_rot
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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_latent, (pipe, background))
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
            del gt_image

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
        
                    # color = scene.gaussians._decoder_rgb(scene.gaussians.get_latent(),view)
                    rendering = renderFunc(viewpoint, scene.gaussians,\
                                                    *renderArgs)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians,\
                    #                                 *renderArgs)["render"], 0.0, 1.0)
                    image = torch.clamp(rendering['render'],0,1)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if idx == 2 and config['name'] == 'test':
                        np_img = exr2png(torch2numpy(image))
                        np_gt = exr2png(torch2numpy(gt_image))
                        
                        relit_diffuse = rendering['relit_diffuse'].detach().cpu().numpy()
                        relit_diffuse = np.clip(relit_diffuse,0,1)
                        relit_diffuse = (relit_diffuse*255).astype('uint8')
                        position = rendering['position'].detach().cpu().numpy().reshape(256,256,3)
                        print("Pos",position.min(),position.max())
                        position = (position-position.min())/(position.max()-position.min())
                        position = (position*255).astype('uint8')
                        scale_img = rendering['scales'].detach().cpu().numpy().reshape(256,256,3)
                        print("Scales",scale_img.min(),scale_img.max())
                        scale_img = (scale_img-scale_img.min())/(scale_img.max()-scale_img.min())
                        scale_img = (scale_img*255).astype('uint8')
                        # opacity_img = rendering['opacity'].detach().cpu().numpy().reshape(256,256,1)
                        # opacity_img = (opacity_img-opacity_img.min())/(opacity_img.max()-opacity_img.min())
                        # opacity_img = (opacity_img*255).astype('uint8')
                        
                        np_result = np.hstack((np_img,np_gt))
                        imwrite(f'{outputs}/output_{iteration}.png',np_result.astype('uint8'))
                        np_img = np.hstack([relit_diffuse,position,scale_img])
                        imwrite(f'{outputs}/uv_{iteration}.png',np_img.astype('uint8'))
                        
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    del gt_image,image
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - decoder_lr', psnr_test, iteration)
                
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(100,100_000,500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100,7_000, 30_000,50_000,100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
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
