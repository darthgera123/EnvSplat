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
import numpy as np
from utils.general_utils import inverse_sigmoid, \
    get_expon_lr_func, build_rotation, get_steplr_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.network import Decoder_RGB,Decoder_shape,\
    EnvMapEncoder, Decoder_RGB_SH, Decoder_Unet, Encoder_Unet, ModifiedEncoderUnet
from utils.model import UVRelit, LatentUnet
from utils.unet_simple import UNet
from utils.sipr import SIPRUnet
from imageio.v2 import imread,imwrite

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

class GaussianModel_exr:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
        


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    def freeze_positions(self):
        self._xyz = self._xyz.detach()
        self._opacity = self._opacity.detach()
        self._scaling = self._scaling.detach()
        self._rotation = self._rotation.detach()
        self._features_dc = self._features_dc.detach()
        # self._features_dc = torch.zeros(self._features_dc.shape).cuda().requires_grad_(True)
        # self._features_dc = self._features_dc.detach()
        # self._features_dc = torch.rand(self._features_dc.shape).cuda().requires_grad_(True)
        # self._features_rest = (torch.rand(self._features_rest.shape)/20.0).cuda().requires_grad_(True)
    
    def freeze_pos(self):
        self._xyz = self._xyz.detach()
        # self._opacity = self._opacity.detach()
        # self._scaling = self._scaling.detach()
        # self._rotation = self._rotation.detach()
        # self._features_dc = self._features_dc.detach()
        # self._features_dc = torch.zeros(self._features_dc.shape).cuda().requires_grad_(True)
        # self._features_dc = self._features_dc.detach()
        # self._features_dc = torch.rand(self._features_dc.shape).cuda().requires_grad_(True)
        # self._features_rest = (torch.rand(self._features_rest.shape)/20.0).cuda().requires_grad_(True)

    def detach_prop(self):
        self._xyz = self._xyz.detach()
        self._opacity = self._opacity.detach()
        self._scaling = self._scaling.detach()
        self._rotation = self._rotation.detach()
        self._features_dc = self._features_dc.detach()
        self._features_rest = self._features_rest.detach()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

class GaussianModel_exr_encoder_sh:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._encoder = Encoder_Unet(in_ch=6+1536)
        self._decoder_rgb = Decoder_Unet(out_ch=48)
        self._decoder_shape = Decoder_Unet(out_ch=48)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    # @property
    # def get_scaling(self):
    #     return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
        
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    # def get_features(self,encoding):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     features = torch.cat((features_dc, features_rest), dim=1)
    #     features = features + self._decoder_rgb(encoding)
    #     return features

    def get_latent(self):
        return self._latent_code
    
    def get_encoder(self,gauss_features):
        
        return self._encoder(gauss_features)

    def get_shape(self,encod_feat,latent_code):
        
        return self._decoder_shape(encod_feat,latent_code)
    
    def get_rgb_sh(self,encod_feat,latent_code):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features = torch.cat((features_dc, features_rest), dim=1)
        lrelu = torch.nn.LeakyReLU(0.2)
        return features + lrelu(self._decoder_rgb(encod_feat,latent_code)).reshape(-1,16,3)

    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)

    @property
    def get_opacity(self):
        return self._opacity
        # return self._opacity
    
    @property
    def get_color(self):
        return self._features_dc
    
    @property
    def get_mask(self):
        return self._mask
    
    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # scales = torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * 1e-3
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self._decoder_shape = self._decoder_shape.to("cuda")
        self._encoder = self._encoder.to("cuda")
        # self._decoder_pos = self._decoder_pos.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': self._encoder.parameters(), 'lr':training_args.encoder_lr, "name": "encoder"},
            # {'params': self._decoder_rgb.parameters(), 'lr':training_args.decoder_rgb_lr, "name": "decoder_rgb"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': self._decoder_shape.parameters(), 'lr':training_args.decoder_lr, "name": "decoder_shape"},
            # {'params': self._decoder_pos.parameters(), 'lr':training_args.position_lr_init * self.spatial_lr_scale, "name": "decoder_pos"}
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=10_000)

        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "encoder":
            #     lr = self.encoder_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            # if param_group["name"] == "decoder_rgb":
            #     lr = self.decoder_rgb_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            # if param_group["name"] == "decoder":
            #     lr = self.decoder_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "xyz" \
                or param_group["name"] == "decoder_rgb" \
                    or param_group["name"] == "encoder" or param_group["name"] == "decoder_shape":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr
            # if 'decoder' in param_group["name"]:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def add_lights(self,diffuse,env_light):
        c,w,h = diffuse.shape
        env_light_flat = env_light.flatten(start_dim=0)
        env_light_rep = env_light_flat.repeat(w*h).view(-1,w,h)
        
        final = torch.cat((diffuse,env_light_rep),dim=0)
        return final.unsqueeze(0)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        # print(shs_view.shape)
        # Ng = shs_view.shape[0]
        # shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (self.max_sh_degree+1)**2)
        # f_dc = shs_view[:,:,:1].transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = shs_view[:,:,1:].transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        xyz = self._xyz.detach().cpu().numpy()
        # xyz = self._xyz + del_pos.reshape(-1,3)
        # xyz = xyz.detach().cpu().numpy()

        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        # opacities = self._opacity + del_opacity.reshape(-1,1)
        # opacities = opacities.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # scale =  self._scaling + del_scale.reshape(-1,3)
        # scale = scale.detach().cpu().numpy()

        # rotation = self._rotation + del_rot.reshape(-1,4)
        # rotation = rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        shs_view = self.get_features
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        name = path.split('.')[0]
        final_path = name+'_init.ply'
        PlyData([el]).write(final_path)


        pos = self._xyz
        colors = self.get_features[:,0,:].reshape(-1,3)
        
        gauss_features = torch.cat([pos,colors],dim=1).reshape(6,256,256)
        envmap = torch.ones((16,32,3)).permute(2,0,1).cuda()
        relit_diffuse = self.add_lights(gauss_features,envmap)
        # encod_feat, latent_code = self.get_encoder(gauss_features.unsqueeze(0))
        encod_feat, latent_code = self.get_encoder(relit_diffuse)
        offset = self.get_shape(encod_feat,latent_code).squeeze(0)
        
        # lrelu = torch.nn.LeakyReLU(0.2)
        # mask = self.get_mask
        # tanh = torch.nn.Tanh()
        # del_pos = lrelu(offset[0:3])
        # del_scale = lrelu(offset[3:6])
        # del_rot = tanh(offset[6:10])
        # del_opa = offset[10:11]
        # del_sh = lrelu(offset[11:])
        
        # xyz = pos + del_pos.reshape(-1,3)
        # xyz = xyz.detach().cpu().numpy()
        # scale = self._scaling + del_scale.reshape(-1,3)
        # scale = scale.detach().cpu().numpy()
        # rotation = self._rotation + del_rot.reshape(-1,4)
        # rotation = rotation.detach().cpu().numpy()
        # opacities = self.get_opacity + del_opa.reshape(-1,1)
        # opacities = opacities.detach().cpu().numpy()
        # shs_view = self.get_features + del_sh.reshape(-1,16,3)

        # f_dc = shs_view[:,:,:1].transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = shs_view[:,:,1:].transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        # attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)
    
    
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._encoder.state_dict(),os.path.join(path, "encoder.pth"))
        

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_encoder(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        self._encoder.load_state_dict(weight_dict)
        self._encoder = self._encoder.to("cuda")

    def init_encoder(self,path):
        
        self._encoder = ModifiedEncoderUnet()
    
    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        rgb_png = SH2RGB(f_dc)
        rgb_exr = np.power(rgb_png,2.2)
        features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

class GaussianModel_unet:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._relit = UNet(in_channels=6+1536,out_channels=58)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    @property
    def get_mask(self):
        return self._mask
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
        
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
class GaussianModel_latent:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self._decoder_rgb = Decoder_RGB_SH(gaussians=256,latent=256,sh_degree=3)
        self._decoder_shape = Decoder_shape(gaussians=256,latent=256,shape_len=11)
        self._latent_code = torch.empty(0)

        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scale(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    @property
    def get_rot(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_latent(self):
        return self._latent_code

    @property
    def get_mask(self):
        return self._mask
    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    def get_decoder_color(self,latent_code):
        return self._decoder_rgb(latent_code)
    def get_decoder_shape(self,latent_code):
        return self._decoder_shape(latent_code)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        latent_code = torch.randn(1,1,256).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._latent_code = nn.Parameter(latent_code.requires_grad_(True))
        self._decoder_rgb = self._decoder_rgb.cuda()
        self._decoder_shape = self._decoder_shape.cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._latent_code], 'lr':0.001,"name":"latent"},
            {'params': self._decoder_rgb.parameters(), 'lr': 1e-4, "name":"decoder_rgb"},
            {'params': self._decoder_shape.parameters(), 'lr': 1e-4, "name":"decoder_shape"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        name = path.split('.')[0]
        final_path = name+'_init.ply'
        PlyData([el]).write(final_path)

        pos = self._xyz
        latent_code = self._latent_code
        del_pos,del_scale,del_rot,del_opacity = self._decoder_shape(latent_code)
        mask = self._mask.reshape(-1,1)
        xyz = pos + del_pos.reshape(-1,3)*mask
        xyz = xyz.detach().cpu().numpy()
        scale = torch.log(torch.exp(self.get_scale) + del_scale.reshape(-1,3)*mask+1e-12)
        scale = scale.detach().cpu().numpy()
        rotation = self.get_rot + del_rot.reshape(-1,4)*mask
        rotation = rotation.detach().cpu().numpy()
        opacities = self.get_opacity+del_opacity.reshape(-1,1)
        opacities = opacities.detach().cpu().numpy()
        shs = self._decoder_rgb(latent_code)
        f_dc = shs[:,:,:1].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = shs[:,:,1:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        
        


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._decoder_rgb.state_dict(),os.path.join(path, "color.pth"))
        torch.save(self._latent_code,os.path.join(path, "latent.pth"))

    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_decoder_rgb(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_rgb.load_state_dict(weight_dict)
        self._decoder_rgb = self._decoder_rgb.to("cuda")
    
    def load_latent(self,path):
        
        self._latent_code = torch.load(path, map_location="cuda")
    
    
class GaussianModel_latent_lights:
    """
    Fit 3dgs on encoder
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self._decoder_rgb = Decoder_RGB_SH(gaussians=256,latent=256,sh_degree=3)
        self._decoder_shape = Decoder_shape(gaussians=256,latent=256,shape_len=11)
        self._encoder = EnvMapEncoder()
        self._latent_code = torch.empty(0)

        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scale(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    @property
    def get_rot(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_latent(self):
        return self._latent_code

    @property
    def get_mask(self):
        return self._mask
    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    def get_decoder_color(self,latent_code):
        return self._decoder_rgb(latent_code)
    def get_decoder_shape(self,latent_code):
        return self._decoder_shape(latent_code)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        latent_code = torch.randn(1,1,256).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._latent_code = nn.Parameter(latent_code.requires_grad_(True))
        self._decoder_rgb = self._decoder_rgb.cuda()
        self._decoder_shape = self._decoder_shape.cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._latent_code], 'lr':0.001,"name":"latent"},
            {'params': self._decoder_rgb.parameters(), 'lr': 1e-4, "name":"decoder_rgb"},
            {'params': self._decoder_shape.parameters(), 'lr': 1e-4, "name":"decoder_shape"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        name = path.split('.')[0]
        final_path = name+'_init.ply'
        PlyData([el]).write(final_path)

        pos = self._xyz
        latent_code = self._latent_code
        del_pos,del_scale,del_rot,del_opacity = self._decoder_shape(latent_code)
        mask = self._mask.reshape(-1,1)
        xyz = pos + del_pos.reshape(-1,3)*mask
        xyz = xyz.detach().cpu().numpy()
        # scale = torch.log(torch.exp(self.get_scale) + del_scale.reshape(-1,3)*mask+1e-12)
        scale = torch.log(self.get_scale + del_scale.reshape(-1,3)*mask+1e-12)
        scale = scale.detach().cpu().numpy()
        rotation = self.get_rot + del_rot.reshape(-1,4)*mask
        rotation = rotation.detach().cpu().numpy()
        opacities = self.get_opacity+del_opacity.reshape(-1,1)
        opacities = opacities.detach().cpu().numpy()
        shs = self._decoder_rgb(latent_code)
        f_dc = shs[:,:,:1].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = shs[:,:,1:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        
        


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._decoder_rgb.state_dict(),os.path.join(path, "color.pth"))
        torch.save(self._latent_code,os.path.join(path, "latent.pth"))

    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_decoder_rgb(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_rgb.load_state_dict(weight_dict)
        self._decoder_rgb = self._decoder_rgb.to("cuda")
    
    def load_latent(self,path):
        
        self._latent_code = torch.load(path, map_location="cuda")










class GaussianModel_uv_sh:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._encoder = Encoder_Unet(in_ch=6+1536)
        self._decoder_shape = Decoder_Unet(out_ch=59)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_color(self):
        return self._features_dc
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_uv(self):
        return self._opacity
    
    def get_encoder(self,gauss_features):
        
        return self._encoder(gauss_features)

    def get_shape(self,encod_feat,latent_code):
        
        return self._decoder_shape(encod_feat,latent_code)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self._decoder_shape = self._decoder_shape.to("cuda")
        self._encoder = self._encoder.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._encoder.parameters(), 'lr':training_args.encoder_lr, "name": "encoder"},
            {'params': self._decoder_shape.parameters(), 'lr':training_args.decoder_lr, "name": "decoder_shape"}
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "xyz" \
                or param_group["name"] == "decoder_rgb" \
                    or param_group["name"] == "encoder" or param_group["name"] == "decoder_shape":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._encoder.state_dict(),os.path.join(path, "encoder.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_encoder(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        self._encoder.load_state_dict(weight_dict)
        self._encoder = self._encoder.to("cuda")

    def load_encoder_light(self,path,in_ch):
        
        self._encoder = ModifiedEncoderUnet(path,in_ch)
        self._encoder = self._encoder.to("cuda")
    
    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

class GaussianModel_uv_sh_relit:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1542,out_ch=48):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._encoder = Encoder_Unet(in_ch=in_ch)
        self._decoder_shape = Decoder_Unet(out_ch=out_ch)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_color(self):
        return self._features_dc
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    
    
    def get_encoder(self,gauss_features):
        
        return self._encoder(gauss_features)

    def get_shape(self,encod_feat,latent_code):
        
        return self._decoder_shape(encod_feat,latent_code)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self._decoder_shape = self._decoder_shape.to("cuda")
        self._encoder = self._encoder.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # f_dc = np.zeros((xyz.shape[0], 3, 1))
        # f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        # features_dc = f_dc

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._encoder.parameters(), 'lr':training_args.encoder_lr, "name": "encoder"},
            {'params': self._decoder_shape.parameters(), 'lr':training_args.decoder_lr, "name": "decoder_shape"}
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "xyz" \
                or param_group["name"] == "decoder_rgb" \
                    or param_group["name"] == "encoder" or param_group["name"] == "decoder_shape":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._encoder.state_dict(),os.path.join(path, "encoder.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_encoder(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        self._encoder.load_state_dict(weight_dict)
        self._encoder = self._encoder.to("cuda")

    def load_encoder_light(self,path,in_ch):
        
        self._encoder = ModifiedEncoderUnet(path,in_ch)
        self._encoder = self._encoder.to("cuda")
    
    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

class GaussianModel_uv_relit:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._encoder = Encoder_Unet(in_ch=in_ch)
        self._decoder_shape = Decoder_Unet(out_ch=out_ch)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_color(self):
        return self._features_dc
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    
    def get_encoder(self,gauss_features):
        
        return self._encoder(gauss_features)

    def get_shape(self,encod_feat,latent_code):
        
        return self._decoder_shape(encod_feat,latent_code)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self._decoder_shape = self._decoder_shape.to("cuda")
        self._encoder = self._encoder.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # f_dc = np.zeros((xyz.shape[0], 3, 1))
        # f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        # features_dc = f_dc

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._encoder.parameters(), 'lr':training_args.encoder_lr, "name": "encoder"},
            {'params': self._decoder_shape.parameters(), 'lr':training_args.decoder_lr, "name": "decoder_shape"}
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "xyz" \
                or param_group["name"] == "decoder_rgb" \
                    or param_group["name"] == "encoder" or param_group["name"] == "decoder_shape":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        
        torch.save(self._decoder_shape.state_dict(),os.path.join(path, "shape.pth"))
        torch.save(self._encoder.state_dict(),os.path.join(path, "encoder.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_encoder(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        self._encoder.load_state_dict(weight_dict)
        self._encoder = self._encoder.to("cuda")

    def load_encoder_light(self,path,in_ch):
        
        self._encoder = ModifiedEncoderUnet(path,in_ch)
        self._encoder = self._encoder.to("cuda")
    
    def load_decoder_shape(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_shape.load_state_dict(weight_dict)
        self._decoder_shape = self._decoder_shape.to("cuda")
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

class GaussianModel_uv_prior:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._gt_features_dc = torch.empty(0)
        self._gt_features_rest = torch.empty(0)
        self._gt_scaling = torch.empty(0)
        self._gt_rotation = torch.empty(0)
        self._init_scaling = torch.empty(0)
        self._gt_opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = UVRelit(in_ch=6,out_ch=59)
        self.gt_scale = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_init_scaling(self):
        return self.scaling_activation(self._init_scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def load_gt_scale(self):
        path = '/scratch/inf0/user/pgera/FlashingLights/3dgs/envmap_mask_max/output/sunrise_pullover/pose_01/9C4A0010-48093a025c_8/point_cloud/iteration_20000/'
        scale_path = os.path.join(path,'scale_map.npy')
        scale = torch.tensor(np.load(scale_path))
        scale_exp = torch.exp(scale)
        self.gt_scale = scale_exp.reshape(-1,3).cuda()


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._init_scaling = scales
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    def load_geometry_gt(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._gt_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    

class GaussianModel_uv_prior_scale:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._gt_xyz = torch.empty(0)
        self._gt_features_dc = torch.empty(0)
        self._gt_features_rest = torch.empty(0)
        self._gt_scaling = torch.empty(0)
        self._gt_rotation = torch.empty(0)
        self._gt_opacity = torch.empty(0)
        self._init_scaling = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = UNet(in_channels=6+1536,out_channels=48)
        self.gt_scale = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_init_scaling(self):
        return self.scaling_activation(self._init_scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._gt_features_dc
        features_rest = self._gt_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def load_gt_scale(self):
        path = '/scratch/inf0/user/pgera/FlashingLights/3dgs/envmap_mask_max/output/sunrise_pullover/pose_01/9C4A0010-48093a025c_8/point_cloud/iteration_20000/'
        scale_path = os.path.join(path,'scale_map.npy')
        scale = torch.tensor(np.load(scale_path))
        scale_exp = torch.exp(scale)
        self.gt_scale = scale_exp.reshape(-1,3).cuda()


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._init_scaling = scales
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    def load_geometry_gt(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._gt_xyz =torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._gt_features_dc =torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._gt_features_rest =torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._gt_opacity =torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._gt_scaling =torch.tensor(scales, dtype=torch.float, device="cuda")
        self._gt_rotation =torch.tensor(rots, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    

class GaussianModel_uv_prior_simple:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._gt_features_dc = torch.empty(0)
        self._gt_features_rest = torch.empty(0)
        self._gt_scaling = torch.empty(0)
        self._init_scaling = torch.empty(0)
        self._gt_rotation = torch.empty(0)
        self._gt_opacity = torch.empty(0)
        self._gt_xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._init_scaling = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = UNet(in_channels=6+1536,out_channels=59)
        self.gt_scale = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_init_scaling(self):
        return self.scaling_activation(self._init_scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def load_gt_scale(self):
        path = '/scratch/inf0/user/pgera/FlashingLights/3dgs/envmap_mask_max/output/sunrise_pullover/pose_01/9C4A0010-48093a025c_8/point_cloud/iteration_20000/'
        scale_path = os.path.join(path,'scale_map.npy')
        scale = torch.tensor(np.load(scale_path))
        scale_exp = torch.exp(scale)
        self.gt_scale = scale_exp.reshape(-1,3).cuda()


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        init_scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._init_scaling = scales
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    def load_geometry_gt(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._gt_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    



class GaussianModel_uv_prior_posenc:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = UVRelit(in_ch=66,out_ch=59)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
 


class GaussianModel_uv_prior_relit:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = UVRelit(in_ch=6,out_ch=48)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features):
        return self._relit(gauss_features)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree


class GaussianModel_uv_prior_latent:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = LatentUnet(in_ch=6,out_ch=48)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features,light):
        return self._relit(gauss_features,light)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=5_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
     
class GaussianModel_uv_prior_latent_all:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,in_ch=1545,out_ch=3):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._gt_features_dc = torch.empty(0)
        self._gt_features_rest = torch.empty(0)
        self._gt_scaling = torch.empty(0)
        self._init_scaling = torch.empty(0)
        self._gt_rotation = torch.empty(0)
        self._gt_opacity = torch.empty(0)
        self._gt_xyz = torch.empty(0)
        self._latent_code = torch.empty(0)
        self.gaussians_num = 256
        self.latent_size = 256
        self.colors = 0
        self._relit = LatentUnet(in_ch=6,out_ch=59)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    @property
    def get_init_scaling(self):
        return self.scaling_activation(self._init_scaling)
    @property
    def get_scaling(self):
        return self._scaling
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    
    # @property
    # def get_rotation(self):
    #     return self._rotation
    
    # @property
    # def get_xyz(self):
    #     return self._xyz

    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_color(self):
        return SH2RGB(self._features_dc).reshape(256,256,3).permute(2,0,1)
    
    @property
    def get_mask(self):
        return self._mask

    def set_mask(self,mask):
        self._mask = torch.tensor(mask.reshape(-1,1)).float().to('cuda')
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_relit(self,gauss_features,light):
        return self._relit(gauss_features,light)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False)) # diffuse color
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False)) # residual
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._init_scaling = scales
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        
        self._relit = self._relit.to("cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def load_geometry(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    def load_geometry_gt(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._gt_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._gt_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._gt_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
    
    def training_setup(self, training_args):
        
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            
            {'params': self._relit.parameters(), 'lr':training_args.encoder_lr, "name": "relit"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.decoder_lr,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.unet_scheduler = get_steplr_lr_func(lr_init=training_args.encoder_lr,
                                                    lr_decay_factor=0.33,
                                                    step_size=20_000)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "xyz":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr
            if param_group["name"] == "relit":
                lr = self.unet_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        folder =os.path.dirname(path) 
        mkdir_p(folder)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        f_dc_png = np.clip(SH2RGB(f_dc),0,1).reshape(256,256,3)
        imwrite(os.path.join(folder,'diffuse.png'),(f_dc_png*255).astype('uint8'))
        # rgb_exr = SH2RGB(f_dc)
        
        # rgb_png = np.clip(np.power(np.clip(rgb_exr,1e-5,None),0.45),0,1)
        
        
        # f_dc_png = RGB2SH(rgb_png)
        attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals,opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        
        
        
        # rgb = (SH2RGB(f_dc)*255.0).astype('uint8')
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals,f_dc_png, f_rest, opacities, scale, rotation,(rgb*255.0).astype('uint8')), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # vertex_element = PlyElement.describe(elements, 'vertex')
        # ply_data = PlyData([vertex_element])
        # name = path.split('.')[0]
        # final_path = name+'_init.ply'
        # ply_data.write(final_path)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._relit.state_dict(),os.path.join(path, "relit.pth"))


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_relit(self,path):
        
        weight_dict = torch.load(path, map_location="cuda")
        if 'relit.pth' in path:
            
            self._relit.load_state_dict(weight_dict)
        else:
            
            self._relit.load_state_dict(weight_dict['model_state_dict'])
        
        self._relit = self._relit.to("cuda")

    
    
    def load_ply(self, path):
        plydata = PlyData.read(path) 
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_dc = np.zeros((xyz.shape[0], 3, 1))
        f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # rgb_png = SH2RGB(f_dc)
        # rgb_exr = np.power(rgb_png,2.2)
        # features_dc = RGB2SH(rgb_exr)

        features_dc = f_dc

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree
