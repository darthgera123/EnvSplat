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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel,\
            GaussianModel_exr,\
            GaussianModel_exr_encoder_sh,GaussianModel_uv_sh, GaussianModel_uv_sh_relit,\
            GaussianModel_uv_relit, GaussianModel_uv_prior, GaussianModel_uv_prior_relit,\
            GaussianModel_uv_prior_latent,GaussianModel_uv_prior_posenc, \
                GaussianModel_uv_prior_scale, GaussianModel_uv_prior_simple, GaussianModel_uv_prior_latent_all
from utils.sh_utils import eval_sh,SH2RGB,RGB2SH
from utils.unet import PosEnc
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scales":scales}


def render_decoder(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    latent_code = pc.get_latent()
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    del_pos,del_scale,del_rot = pc._decoder_shape(latent_code)
    
    means3D = means3D + del_pos.reshape(-1,3)
    scales = torch.exp(scales + del_scale.reshape(-1,3))
    rotations = torch.nn.functional.normalize(rotations + del_rot.reshape(-1,4))
    # opacity = torch.sigmoid(pc._decoder_opacity(latent_code)).reshape(-1,1)

    view = viewpoint_camera.camera_center.reshape((1,1,3))
    view=view/(view.norm(dim=2, keepdim=True))
    shs = None
    colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))

    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_encoder(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    
    latent_code = pc._encoder(envmap)
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    del_pos,del_scale,del_rot = pc._decoder_shape(latent_code)
    
    means3D = means3D + del_pos.reshape(-1,3)
    scales = torch.exp(scales + del_scale.reshape(-1,3))
    rotations = torch.nn.functional.normalize(rotations + del_rot.reshape(-1,4))
    # opacity = torch.sigmoid(pc._decoder_opacity(latent_code)).reshape(-1,1)

    view = viewpoint_camera.camera_center.reshape((1,1,3))
    view=view/(view.norm(dim=2, keepdim=True))
    shs = None
    colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))

    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_decoder_sh(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    latent_code = pc.get_latent()
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    del_pos,del_scale,del_rot = pc._decoder_shape(latent_code)
    
    means3D = means3D + del_pos.reshape(-1,3)
    scales = torch.exp(scales + del_scale.reshape(-1,3))
    # scales = scales + del_scale.reshape(-1,3)
    rotations = torch.nn.functional.normalize(rotations + del_rot.reshape(-1,4))
    # opacity = torch.sigmoid(opacity+del_opacity.reshape(-1,1))
    diff = pc.get_color
    diff = diff.reshape(256,256,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    diff = diff.permute(2,0,1)
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    shs = pc.get_features(latent_code)
    # colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))
    
    relit_diffuse = shs[:,0].reshape(256,256,3)
    op_map = opacity.reshape(256,256,1)
    pos = means3D.reshape(256,256,3)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "diffuse": df,
            "relit": relit_diffuse,
            "opacity": op_map,
            "pos": pos}


def render_latent(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    latent_code = pc.get_latent
    
    del_pos,del_scale,del_rot,del_opacity = pc.get_decoder_shape(latent_code)

    mask = pc.get_mask.reshape(-1,1)
    means3D = pc.get_xyz + del_pos.reshape(-1,3)
    means3D = means3D*mask
    # means3D = pc.get_xyz*mask
    means2D = screenspace_points
    # opacity = torch.sigmoid(pc.get_opacity)*mask
    opacity = torch.sigmoid(pc.get_opacity+del_opacity.reshape(-1,1))*mask
    # opacity = torch.sigmoid(del_opacity.reshape(-1,1))*mask

    scales = torch.exp(pc.get_scale+ del_scale.reshape(-1,3)*mask)
    # scales = torch.exp(pc.get_scale)+ del_scale.reshape(-1,3)*mask
    rotations = torch.nn.functional.normalize(pc.get_rot + del_rot.reshape(-1,4)*mask)
    # scales = torch.exp(pc.get_scale)
    # rotations = torch.nn.functional.normalize(pc.get_rot)
    # print(means3D.shape,scales.shape,rotations.shape,opacity.shape)
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # shs = pc.get_features
    shs = pc.get_decoder_color(latent_code)

    relit_diffuse = SH2RGB(shs[:,0]*mask).reshape(256,256,3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scales":del_scale.reshape(-1,3)*mask,
            "rotation":del_rot.reshape(-1,4)*mask,
            "position":del_pos.reshape(-1,3)*mask,
            "opacity": del_opacity.reshape(-1,1)*mask,
            "relit_diffuse":relit_diffuse}


def add_lights(diffuse,env_light):
    c,w,h = diffuse.shape
    env_light_flat = env_light.flatten(start_dim=0)
    env_light_rep = env_light_flat.repeat(w*h).view(-1,w,h)
    
    final = torch.cat((diffuse,env_light_rep),dim=0)
    return final.unsqueeze(0)


def render_encoder_sh(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Pretrained the network on fully lit geometry with this
    Single network n
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    pos = pc.get_xyz
    colors = pc.get_features[:,0,:].reshape(-1,3)
    
    gauss_features = torch.cat([pos,colors],dim=1).reshape(6,256,256)
    envmap = torch.ones((16,32,3)).permute(2,0,1).cuda()
    relit_diffuse = add_lights(gauss_features,envmap)
    # encod_feat, latent_code = pc.get_encoder(gauss_features.unsqueeze(0))
    encod_feat, latent_code = pc.get_encoder(relit_diffuse)
    offset = pc.get_shape(encod_feat,latent_code).squeeze(0)
    # rgb_sh =  pc.get_rgb_sh(encod_feat,latent_code).squeeze(0)
    diff = pc.get_color
    diff = diff.reshape(256,256,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    lrelu = torch.nn.LeakyReLU(0.2)
    means2D = screenspace_points
    opacity = pc.get_opacity
    mask = pc.get_mask
    tanh = torch.nn.Tanh()
    # del_pos = offset[0:3].permute(1,2,0)
    # del_scale = lrelu(offset[3:6]).permute(1,2,0)
    # del_scale = offset[3:6]
    # del_rot = offset[6:10]
    # del_rot = tanh(del_rot).permute(1,2,0)
    # del_posy = lrelu(del_pos.reshape(-1,3))*mask
    # del_posy = del_pos.reshape(-1,3)*mask
    
    # del_opa = offset[10:11].permute(1,2,0)
    # del_sh = offset[11:].permute(1,2,0)
    del_sh = offset[:].permute(1,2,0)
    
    # del_scale = offset[4:7]
    # means3D = pc.get_xyz + lrelu(del_pos.reshape(-1,3))
    # means3D = means3D*mask
    means3D = pc.get_xyz*mask
    # opacity = torch.sigmoid(pc.get_opacity + del_opa.reshape(-1,1))*mask
    opacity = torch.sigmoid(pc.get_opacity)*mask
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # scales = torch.exp(pc.get_scaling+del_scale.reshape(-1,3))
    # scales = torch.exp(pc.get_scaling)
    
    # scales = torch.exp(scales+del_scale.reshape(-1,3))
    # elu = torch.nn.ELU()
    # scales = torch.exp(pc.get_scaling + del_scale.reshape(-1,3)*mask).clip(1e-6)
    # scales = torch.exp(pc.get_scaling) +del_scale.reshape(-1,3)*mask.clip(1e-6)
    # scales = torch.sigmoid(torch.exp( del_scale.reshape(-1,3)).clip(1e-6))*0.1
    # scales = torch.exp(pc.get_scaling)
    # print(scales.max(),scales.min())
    # rotations = pc.get_rotation
    # rotations = torch.nn.functional.normalize(rotations+ del_rot.reshape(-1,4)*mask)
    # rotations = torch.nn.functional.normalize(rotations)
    # opacity = torch.sigmoid(pc._decoder_opacity(latent_code)).reshape(-1,1)

    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if True:
    #         # shs_view = pc.get_features(latent_code)
    #         # # print(shs_view.shape)
    #         # Ng = shs_view.shape[0]
    #         # shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         # dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    # colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))
    relu = torch.nn.ReLU()
    # shs = lrelu(del_sh.reshape(-1,1,3))
    # shs = pc.get_features + lrelu(del_sh.reshape(-1,16,3))
    shs = pc.get_features
    relit_diffuse = shs[:,0].reshape(256,256,3)
    # posy = del_pos.reshape(256,256,3)
    posy = means3D.reshape(256,256,3)
    # roty = del_rot.reshape(256,256,4)
    rota = rotations.reshape(256,256,4)
    opa = opacity.reshape(256,256,1)
    scaly = scales.reshape(256,256,3)
    scala = scales.reshape(256,256,3)
    masky = mask.reshape(256,256,1)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "diffuse": df,
            "relit_diffuse": relit_diffuse,
            "pos": posy,
            "opa": opa,
            "scales":scales,
            "scaly": scaly,
            "masky":masky,
            "scala": scala,
            "rota": rota}

def render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = pc.get_xyz
    colors = pc.get_features[:,0,:].reshape(-1,3)
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()

    gauss_features = torch.cat([pos,colors],dim=1).reshape(6,256,256)
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = pc.get_mask
    # print(envmap.min(),envmap.max())
    encod_feat, latent_code = pc.get_encoder(relit_diffuse)
    # encod_feat, latent_code = pc.get_encoder(gauss_features.unsqueeze(0))
    offset = pc.get_shape(encod_feat,latent_code).squeeze(0)
    
    del_pos = lrelu(offset[0:3])
    del_scale = lrelu(offset[3:6])
    del_rot = tanh(offset[6:10]) 
    del_opa = offset[10:11]
    del_sh = offset[11:]
    
    diff = pc.get_color
    diff = diff.reshape(256,256,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    
    means3D = pc.get_xyz + del_pos.reshape(-1,3)
    means3D = means3D*mask
    scales = torch.exp((pc.get_scaling + del_scale.reshape(-1,3)*mask)).clip(1e-6)
    rotations = torch.nn.functional.normalize((pc.get_rotation+del_rot.reshape(-1,4)*mask))
    opacity = torch.sigmoid(pc.get_opacity + del_opa.reshape(-1,1))*mask
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    shs = (pc.get_features + lrelu(del_sh.reshape(-1,16,3)))
    # shs = shs*mask.reshape(-1,1)
    # shs = lrelu(del_sh.reshape(-1,16,3))
    # colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))
    
    # colors_precomp = torch.relu(features.reshape(-1,3))
    relit_diffuse = shs[:,0].reshape(256,256,3)
    posy = del_pos.reshape(256,256,3)
    roty = del_rot.reshape(256,256,4)
    rota = rotations.reshape(256,256,4)
    opa = opacity.reshape(256,256,1)
    scaly = del_scale.reshape(256,256,3)
    scala = scales.reshape(256,256,3)
    masky = mask.reshape(256,256,1)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "diffuse": df,
            "relit_diffuse": relit_diffuse,
            "pos": posy,
            "opa": opa,
            "roty": roty,
            "scaly": scaly,
            "masky":masky,
            "scala": scala,
            "rota": rota}

def render_uv_sh_relit(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    
    mask = pc.get_mask

    pos = pc.get_xyz
    colors = pc.get_features[:,0,:].reshape(-1,3)
    gauss_features = torch.cat([pos,colors],dim=1).reshape(6,256,256)
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    encod_feat, latent_code = pc.get_encoder(relit_diffuse)
    offset = pc.get_shape(encod_feat,latent_code).squeeze(0)
    del_sh = offset[0:]


    means3D = pc.get_xyz * mask
    scales = torch.exp(pc.get_scaling)
    rotations = torch.nn.functional.normalize(pc.get_rotation)
    opacity = torch.sigmoid(pc.get_opacity)*mask

    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))
    lrelu = torch.nn.LeakyReLU(0.2)
    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    # shs = features.reshape(-1,1,3)
    # colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))
    
    shs = (pc.get_features + del_sh.reshape(-1,16,3))
    relit_diffuse = shs[:,0].reshape(256,256,3)
    diff = pc.get_color
    diff = diff.reshape(256,256,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "diffuse": df,
            "relit_diffuse": relit_diffuse}


def render_uv_relit(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    view = viewpoint_camera.camera_center.reshape((1,1,3))
    view=view/(view.norm(dim=2, keepdim=True))
    
    mask = pc.get_mask

    pos = pc.get_xyz
    colors = pc.get_features[:,0,:].reshape(-1,3)
    
    dir_pp = (pos - viewpoint_camera.camera_center.repeat(256*256, 1))
    view_vector = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    relu = torch.nn.ReLU()
    softplus = torch.nn.Softplus()

    gauss_features = torch.cat([pos,colors,view_vector],dim=1).reshape(9,256,256)
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,9+1536,256,256])
    encod_feat, latent_code = pc.get_encoder(relit_diffuse)
    offset = pc.get_shape(encod_feat,latent_code).squeeze(0)
    del_colors = torch.sigmoid(offset[0:])
    del_colors = relu(offset[0:]).clamp(0,1)


    means3D = pc.get_xyz * mask
    scales = torch.exp(pc.get_scaling)
    rotations = torch.nn.functional.normalize(pc.get_rotation)
    opacity = torch.sigmoid(pc.get_opacity)*mask

    

    

    lrelu = torch.nn.LeakyReLU(0.2)
    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = del_colors.reshape(-1,3)*mask
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    # shs = features.reshape(-1,1,3)
    # colors_precomp = torch.relu(pc._decoder_rgb(latent_code,view).reshape(-1,3))
    
    # shs = (pc.get_features + del_sh.reshape(-1,16,3))
    relit_diffuse = colors_precomp.reshape(256,256,3)
    diff = pc.get_color
    diff = diff.reshape(256,256,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "diffuse": df,
            "relit_diffuse": relit_diffuse}

def render_uv_prior_scale(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    softplus = torch.nn.Softplus()

    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0)
    
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    
    

    scaleparam=1
    # del_pos = lrelu(offset[:3])*mask
    # del_scale = sigmoid(offset[0:3])*mask
    # del_scale = offset[0:3]
    # del_scale = sigmoid(offset[0:3])*mask
    

    # del_scale = offset[0:3]*mask
    # del_scale = del_scale*scaleparam
    # del_col = softplus(offset[0:3])*mask
    del_col = offset[0:3]*mask
    del_sh = offset[3:]*mask
    
    # del_rot = tanh(offset[6:10])
    # del_rot = 2*sigmoid(offset[6:10])-1
    # del_rot = del_rot*mask
    # del_rot = offset[6:10]
    # del_opacity = sigmoid(offset[10:11])*mask
    # del_col = relu(offset[11:14])*mask
    # del_sh = lrelu(offset[14:])
    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    # rot_init = torch.zeros((pc.get_xyz.shape[0],4),device='cuda')
    # rot_init[:,0] = 1
    new_pos = pc._gt_xyz
    # new_pos = pos
    # means3D = new_pos.permute(1,2,0).reshape(-1,3)
    means3D = new_pos
    # scales = pc.get_init_scaling + del_scale.permute(1,2,0).reshape(-1,3)
    scales = torch.exp(pc._gt_scaling)
    # scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    # new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    # rotations = torch.nn.functional.normalize(rot_init + new_rotation)
    # rotations = new_rotation
    
    rotations = torch.nn.functional.normalize(pc._gt_rotation)
    # opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    opacity = sigmoid(pc._gt_opacity)*mask.reshape(-1,1)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    # sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    # sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    # sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)


    # shs = sh_col
    # shs = pc.get_features
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)
    shs = sh_col
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    # pos_img = del_pos.permute(1,2,0)
    # rot_img = del_rot.permute(1,2,0)
    
    # opa_img = del_opacity.permute(1,2,0)
    # scale_img = torch.abs(del_scale).permute(1,2,0)
    scale_img = scales.reshape(256,256,3)
    
    # relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scales":scale_img,
            "diffuse":diff,
            "mask":masky
            }

def render_uv_prior(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0)
    
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    
    

    scaleparam=1
    del_pos = offset[:3]*mask
    del_scale = offset[3:6]*mask
    del_scale = del_scale*scaleparam
    del_rot = offset[6:10]*mask
    del_opacity = sigmoid(offset[10:11])*mask
    # del_col = relu(offset[11:14])*mask
    del_col = offset[11:14]*mask
    del_sh = offset[14:]*mask
    df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    rot_init = torch.zeros((pc.get_xyz.shape[0],4),device='cuda')
    rot_init[:,0] = 1
    new_pos = pos+del_pos
    # new_pos = pos
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    # means3D = pc._xyz   
    # scales = del_scale.permute(1,2,0).reshape(-1,3)
    scales = pc.get_init_scaling + del_scale.permute(1,2,0).reshape(-1,3)
    # scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    rotations = torch.nn.functional.normalize(rot_init + new_rotation)
    
    # rotations = pc._gt_rotation
    opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    # opacity = sigmoid(pc._gt_opacity)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    pos_img = del_pos.permute(1,2,0)
    rot_img = del_rot.permute(1,2,0)
    
    opa_img = del_opacity.permute(1,2,0)
    scale_img = del_scale.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rotation":rot_img,
            "pos":pos_img,
            "opacity":opa_img,
            "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }

def render_uv_prior_simple(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0)
    
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    
    

    scaleparam=1
    # del_pos = lrelu(offset[:3])*mask
    # del_scale = sigmoid(offset[3:6])*mask
    del_pos = offset[:3]*mask
    del_scale = offset[3:6]*mask
    del_scale = del_scale*scaleparam
    del_rot = offset[6:10]*mask
    del_opacity = sigmoid(offset[10:11])*mask
    # del_col = relu(offset[11:14])*mask
    del_col = offset[11:14]*mask
    del_sh = offset[14:]*mask
    # del_rot = tanh(offset[6:10])
    # del_rot = 2*sigmoid(offset[6:10])-1
    # del_rot = del_rot*mask
    # del_rot = offset[6:10]
    # del_opacity = sigmoid(offset[10:11])*mask
    # del_col = relu(offset[11:14])*mask
    # del_sh = lrelu(offset[14:])
    df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    rot_init = torch.zeros((pc.get_xyz.shape[0],4),device='cuda')
    rot_init[:,0] = 1
    new_pos = pos+del_pos
    # new_pos = pos
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    # means3D = pc._xyz   
    scales = pc.get_init_scaling + del_scale.permute(1,2,0).reshape(-1,3)
    # scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    rotations = torch.nn.functional.normalize(rot_init + new_rotation)
    # rotations = rot_init + new_rotation
    # rotations = new_rotation
    # rotations = torch.nn.functional.normalize(pc._rotation)
    # rotations = pc._gt_rotation
    opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    # opacity = sigmoid(pc._opacity)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # means3D = pc._gt_xyz
    # scales = torch.exp(pc._gt_scaling)
    # rotations = torch.nn.functional.normalize(pc._gt_rotation)
    # opacity = torch.sigmoid(pc._gt_opacity)


    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation

    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    pos_img = del_pos.permute(1,2,0)
    rot_img = del_rot.permute(1,2,0)
    
    opa_img = del_opacity.permute(1,2,0)
    scale_img = del_scale.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rotation":rot_img,
            "pos":pos_img,
            "opacity":opa_img,
            "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }

def render_unet(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    scales = None
    rotations = None
    cov3D_precomp = None
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    envmap = torch.ones((16,32,3)).permute(2,0,1).cuda()
    gauss_features = torch.cat([pos,colors],dim=0)
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    del_pos = offset[0:3].permute(1,2,0)
    del_scale = lrelu(offset[3:6]).permute(1,2,0)
    del_rot = offset[6:10]
    del_rot = tanh(del_rot).permute(1,2,0)
    # del_opa = torch.sigmoid(offset[10:11]).permute(1,2,0)
    del_sh = offset[10:].permute(1,2,0)
    mask = pc.get_mask
    # opacity = torch.sigmoid(pc._opacity + del_opa.reshape(-1,1))*mask
    opacity = torch.sigmoid(pc._opacity)*mask
    means3D = pc.get_xyz + lrelu(del_pos.reshape(-1,3))
    means3D = means3D * mask
    means2D = screenspace_points
    # opacity = pc.get_opacity * mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
        
    #     rotations = pc.get_rotation
    # scales = pc.get_scaling
    scales = torch.exp(pc._scaling + del_scale.reshape(-1,3)*mask).clip(1e-6)
    # rotations = pc.get_rotation
    rotations = pc._rotation
    rotations = torch.nn.functional.normalize(rotations+ del_rot.reshape(-1,4)*mask)
    
    shs = None
    colors_precomp = None
    # shs = pc.get_features + lrelu(del_sh.reshape(-1,16,3))
    
    shs = lrelu(del_sh.reshape(-1,16,3))
    # shs = SH2RGB(shs[:,0].reshape(256,256,3))        
    relit_diffuse = SH2RGB(shs[:,0].reshape(256,256,3))*mask.reshape(256,256,1)
    
    shs[:,0] = RGB2SH(relit_diffuse.reshape(-1,3))
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rotation":rotations,
            "pos":means3D,
            "opacity":opacity,
            "scales":scales,
            "relit_diffuse":relit_diffuse,
            "mask":mask.reshape(256,256,1)
            }

def render_uv_prior_posenc(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    pos_enc = PosEnc(10)
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    enc_pos = pos_enc(pos)
    # gauss_features = torch.cat([pos,colors],dim=0)
    gauss_features = torch.cat([enc_pos,colors],dim=0)
    
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    
    

    scaleparam=1
    del_pos = lrelu(offset[:3])*mask
    del_scale = sigmoid(offset[3:6])*mask
    del_scale = del_scale*scaleparam
    
    del_rot = tanh(offset[6:10])
    # del_rot = 2*sigmoid(offset[6:10])-1
    # del_rot = offset[6:10]
    del_opacity = sigmoid(offset[10:11])*mask
    del_col = relu(offset[11:14])*mask
    del_sh = lrelu(offset[14:])
    df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    # new_pos = pos+del_pos
    new_pos = pos
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    # scales = del_scale.permute(1,2,0).reshape(-1,3)
    scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    # new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    # rotations = torch.nn.functional.normalize(new_rotation)
    rotations = pc._rotation
    # opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    # opacity = sigmoid(pc._opacity)*mask.reshape(-1,1)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    pos_img = del_pos.permute(1,2,0)
    rot_img = del_rot.permute(1,2,0)
    
    opa_img = del_opacity.permute(1,2,0)
    scale_img = del_scale.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "pos":pos_img,
            "opa_img":opa_img,
            "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }

def render_uv_prior_relit(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0)
    
    relit_diffuse = add_lights(gauss_features,envmap) # ([1,6+1536,256,256])
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(relit_diffuse).squeeze(0)
    
    df_scales = torch.exp(pc._scaling)

    scaleparam=0.02
    # del_pos = lrelu(offset[:3])*mask
    # del_scale = sigmoid(offset[3:6])*mask
    # del_scale = del_scale*scaleparam
    
    # del_rot = tanh(offset[6:10])*mask
    # del_rot = offset[6:10]
    # del_opacity = sigmoid(offset[10:11])*mask
    del_col = relu(offset[0:3])*mask
    del_sh = lrelu(offset[3:])
    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    # new_pos = pos+del_pos
    new_pos = pos * mask
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    # scales = del_scale.permute(1,2,0).reshape(-1,3)
    scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    # new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    # rotations = torch.nn.functional.normalize(new_rotation)
    rotations = pc._rotation
    # opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    opacity = sigmoid(pc._opacity)*mask.reshape(-1,1)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)
    # sh_col = sh_deg_0


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    # pos_img = pos.permute(1,2,0)
    # rot_img = del_rot.permute(1,2,0)
    
    # opa_img = opacity.permute(1,2,0)
    # scale_img = scales.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # "pos":pos_img,
            
            # "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }

def render_uv_prior_latent(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0).unsqueeze(0)
    envmap = envmap.unsqueeze(0)
    
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(gauss_features,envmap).squeeze(0)
    
    df_scales = torch.exp(pc._scaling)

    scaleparam=0.02
    # del_pos = lrelu(offset[:3])*mask
    # del_scale = sigmoid(offset[3:6])*mask
    # del_scale = del_scale*scaleparam
    
    # del_rot = tanh(offset[6:10])*mask
    # del_rot = offset[6:10]
    # del_opacity = sigmoid(offset[10:11])*mask
    del_col = relu(offset[0:3])*mask
    del_sh = lrelu(offset[3:])
    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    # new_pos = pos+del_pos
    new_pos = pos * mask
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    # scales = del_scale.permute(1,2,0).reshape(-1,3)
    scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    # new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    # rotations = torch.nn.functional.normalize(new_rotation)
    rotations = pc._rotation
    # opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    opacity = sigmoid(pc._opacity)*mask.reshape(-1,1)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)
    # sh_col = sh_deg_0


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    # pos_img = pos.permute(1,2,0)
    # rot_img = del_rot.permute(1,2,0)
    
    # opa_img = opacity.permute(1,2,0)
    # scale_img = scales.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # "pos":pos_img,
            
            # "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }

def render_uv_prior_latent_all(viewpoint_camera, pc, pipe, bg_color,
           scaling_modifier=1.0,envmap=None, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # means = pc.get_xyz.reshape(256,256,3).permute(2,0,1)
    # scales = pc.get_scaling.reshape(256,256,3).permute(2,0,1)
    # rotations = pc.get_rotation.reshape(256,256,4).permute(2,0,1)
    # opacity = pc.get_opacity.reshape(256,256,1).permute(2,0,1)
    # diff = pc.get_color.reshape(256,256,3)
    
    pos = (pc.get_xyz).reshape(256,256,3).permute(2,0,1)
    colors = pc.get_color
    # activations
    lrelu = torch.nn.LeakyReLU(0.2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()
    sigmoid = torch.sigmoid
    gauss_features = torch.cat([pos,colors],dim=0).unsqueeze(0)
    envmap = envmap.unsqueeze(0)
    
    mask = (pc.get_mask).reshape(256,256,1).permute(2,0,1)
    # print(envmap.min(),envmap.max())
    offset = pc.get_relit(gauss_features,envmap).squeeze(0)
    
    df_scales = torch.exp(pc._scaling)

    scaleparam=1
    # del_pos = lrelu(offset[:3])*mask
    # del_scale = sigmoid(offset[3:6])*mask
    del_pos = offset[:3]*mask
    del_scale = offset[3:6]*mask
    del_scale = del_scale*scaleparam
    del_rot = offset[6:10]*mask
    del_opacity = sigmoid(offset[10:11])*mask
    # del_col = relu(offset[11:14])*mask
    del_col = offset[11:14]*mask
    del_sh = offset[14:]*mask
    df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # diff = pc.get_color
    # diff = diff.reshape(256,256,3)
    # dt = diff[:,:,:3].reshape(-1,1,3)
    
    # df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    
    new_pos = pos+del_pos
    # new_pos = pos * mask
    means3D = new_pos.permute(1,2,0).reshape(-1,3)
    scales = pc.get_init_scaling + del_scale.permute(1,2,0).reshape(-1,3)
    # scales = df_scales.reshape(-1,3)
    # print(scales.min(),scales.max())
    # new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    # rotations = torch.nn.functional.normalize(new_rotation)
    rot_init = torch.zeros((pc.get_xyz.shape[0],4),device='cuda')
    rot_init[:,0] = 1
    new_rotation = del_rot.permute(1,2,0).reshape(-1,4)
    rotations = torch.nn.functional.normalize(rot_init + new_rotation)
    opacity = del_opacity.permute(1,2,0).reshape(-1,1)
    # opacity = sigmoid(pc._opacity)*mask.reshape(-1,1)
    
    # view = viewpoint_camera.camera_center.reshape((1,1,3))
    # view=view/(view.norm(dim=2, keepdim=True))

    # shs = pc.get_features(latent_code)
    shs = None
    colors_precomp = None

    # df_scales = torch.exp(pc._scaling).reshape(256,256,3)
    # df_rotation = pc._rotation
    # if override_color is None:
    #     if True:
    #         shs_view = pc.get_features(latent_code)
    #         # print(shs_view.shape)
    #         Ng = shs_view.shape[0]
    #         shs_view = shs_view.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            
    #         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(Ng, 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
    #     else:
    #         shs = pc.get_features(latent_code)
    # else:
    #     colors_precomp = override_color
    sh_deg_0 = RGB2SH(del_col).permute(1,2,0).reshape(-1,1,3)
    sh_deg_rest = del_sh.permute(1,2,0).reshape(-1,15,3)
    sh_col = torch.cat((sh_deg_0,sh_deg_rest),dim=1)


    shs = sh_col
    # shs = pc.get_features
    # shs = shs*mask.rdef render_uv_all_sh(viewpoint_camera, pc, pipe, bg_color,s[:,0].reshape(256,256,3)
    pos_img = del_pos.permute(1,2,0)
    rot_img = del_rot.permute(1,2,0)
    
    opa_img = del_opacity.permute(1,2,0)
    scale_img = del_scale.permute(1,2,0)
    
    relit_diff = del_col.permute(1,2,0)
    diff = colors.permute(1,2,0)
    masky = mask.permute(1,2,0)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rotation":rot_img,
            "pos":pos_img,
            "opacity":opa_img,
            "scales":scale_img,
            "relit_diffuse":relit_diff,
            "diffuse":diff,
            "mask":masky,
            "df_scale": df_scales
            }