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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
class MeshEncoder(nn.Module):
    def __init__(self):
        super(MeshEncoder, self).__init__()
        # Assuming the input is flattened (5023*3)
        self.fc1 = nn.Linear(5023*3, 2048)  # First layer
        self.fc2 = nn.Linear(2048, 1024)    # Second layer
        self.fc3 = nn.Linear(1024, 512)     # Third layer
        self.fc4 = nn.Linear(512, 256)      # Fourth layer (output)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 5023*3)

        # Passing through the MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation after the last layer
        return x
class CNNEncoder_1(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # Output: (64, 256, 256)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: (128, 128, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Output: (256, 64, 64)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (256, 32, 32)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 256 * 32 * 32)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Adjusting the first convolutional layer to have 9 input channels
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1)    # Output: (64, 128, 128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # Output: (128, 64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: (256, 32, 32)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 16, 16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 8, 8)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 4, 4)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 2, 2)

        # Adjusting the size of the first linear layer
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Flatten and pass through the linear layers
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNNEncoder512(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Define seven convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)    # Output: (64, 256, 256)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # Output: (128, 128, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: (256, 64, 64)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 32, 32)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 16, 16)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 8, 8)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Output: (512, 4, 4)

        # Define two linear layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Flatten and pass through the linear layers
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNNEncoderold(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.encoder_net(x)
        return x
class MLPDecoder(nn.Module):
    def __init__(self):
        super(MLPDecoder, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.fc4 = nn.Linear(4096, 15069)  # Change the output dimension

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation in the last layer
        x = x.view(-1, 5023, 3)  # Reshape the output
        return x
class RefinementNetwork(nn.Module):
    def __init__(self):
        super(RefinementNetwork, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Add as many layers as necessary for your refinement
        
        # Final convolutional layer to get back to 3 channels
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # Pass input through the convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Final convolution without a ReLU, because we might need the output to be able to represent
        # a full range of color intensity (0-255 for each channel)
        x = self.final_conv(x)
        
        return x


# Decoder using Transposed Convolutions

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 128 * 32 * 32)
        
        # Adding more layers and adjusting parameters to achieve the desired output shape
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 10, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.fc(x))
        x = x.view(x.size(0), 128, 32, 32)
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = nn.ReLU()(self.deconv3(x))
        x = nn.ReLU()(self.deconv4(x))
        x = self.deconv5(x)
        x=x.reshape(10,512*512)
        trans_delta=x[0:3,:]
        scales_delta=x[3:6,:]
        rotations_delta=x[6:10,:]

        return trans_delta,scales_delta,rotations_delta
class Decoder_alpha(nn.Module):
    def __init__(self):
        super(Decoder_alpha, self).__init__()
        self.fc = nn.Linear(256, 128 * 32 * 32)
        
        # Adding more layers and adjusting parameters to achieve the desired output shape
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.fc(x))
        x = x.view(x.size(0), 128, 32, 32)
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = nn.ReLU()(self.deconv3(x))
        x = nn.ReLU()(self.deconv4(x))
        x = self.deconv5(x)
        x=x.reshape(1,512*512)
        opacity=x[0:3,:]
        #scales_delta=x[3:6,:]
        #rotations_delta=x[6:10,:]

        return opacity
class Decoder_RGB(nn.Module):
    def __init__(self):
        super(Decoder_RGB, self).__init__()
        # Define the initial fully connected layer
        self.fc = nn.Linear(259, 256 * 32 * 32)  # Increase from 128 to 256

        # Define the transposed convolutional layers
        # The sequence is designed to double the spatial dimensions step by step
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 64x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)   # Output: 128x128
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)    # Output: 256x256
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)    # Output: 512x512
        self.deconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)                      # Output: 512x512


    def forward(self, x):
        x = nn.ReLU()(self.fc(x))
        x = x.view(x.size(0), 256, 32, 32)
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = nn.ReLU()(self.deconv3(x))
        x = nn.ReLU()(self.deconv4(x))
        x = self.deconv5(x)
        x=x.reshape(3,512*512)
        x=x.T
        #x=x.reshape(512*512,3)
        #scales_delta=x[3:6,:]
        #rotations_delta=x[6:10,:]
        return x
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
        self.color_in= torch.empty(0)
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
        latent_initial = torch.randn(210,1, 256) 
        latent_code =torch.sigmoid(latent_initial)
        #latent_code.requires_grad = True
        self._latent_code=latent_code
        # Model, Loss, Optimizer
        self._decoder = Decoder()
        self._mlp_decoder = MLPDecoder()
        #self._cnn_encoder = CNNEncoder()
        self._cnn_encoder = CNNEncoder()
        self._decoder_opacity = Decoder_alpha()
        self._decoder_rgb = Decoder_RGB()
        self._refinement = RefinementNetwork()
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
    
    #@property
    def get_rotation(self,time):
        return self.rotation_activation(self._rotation[time])
    
    #@property
    def get_xyz(self,time):
        #print(self._xyz.shape)
        return self._xyz[time]    
    #@property
    def get_latent(self,time):
        #print(time)
        return self._latent_code[time]
    
    #@property
    #def get_features(self):
    #    features_dc = self._features_dc
    #    features_rest = self._features_rest
    #    #print(features_dc.shape)
    #    return features_dc.reshape(512*512,3)
    def get_features(self,campos,time,encoding):
        #features_dc = self._features_dc
        #features_rest = self._features_rest
       
        campos=campos.reshape(1,3)
        campos=campos/(campos.norm(dim=1, keepdim=True))
        #print(campos.shape)
        #print(self._latent_code.shape)
        input = torch.cat((encoding, campos), dim=1)

        features_dc= self._decoder_rgb(input)
        #print(features_dc.shape,print(features_rest.shape))
        #print(features_rest.shape)
        #print(features_dc.shape)

        return features_dc
    
    #@property
    def get_opacity(self,time,encoding):
        #print(self._latent_code[time].shape)
        return self.opacity_activation(self._decoder_opacity(encoding).T)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
       # pcd = np.array(pcd)
       # print(pcd)
        #pcd_colors = np.array([pcdx.colors for pcdx in pcd])
        pcd_points = pcd#np.array([pcdx.points for pcdx in pcd])
       # print(self.colors.shape)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd_points)).float().cuda()
        #fused_color = RGB2SH(torch.tensor(np.asarray(pcd_colors)).float().cuda())
        #features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #features[:, :3, 0 ] = fused_color
        #features[:, 3:, 1:] = 0.0
        #self.color_in = torch.tensor(np.asarray(pcd_colors)).float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dists=[]
        #for num in range (pcd_points.shape[0]):
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd_points[0])).float().cuda()), 0.0000001)

        #print(dist2.shape)
        print(torch.log(torch.sqrt(dist2))[...,None].shape)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0],fused_point_cloud.shape[1], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        
        #self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        #self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        print(self._xyz.shape)
        print(self._scaling.shape)
        print(self._rotation.shape)
        #self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #latent=inverse_sigmoid(0.1 * torch.ones((1,256), dtype=torch.float, device="cuda"))
        #self._latent_code=nn.Parameter(latent.requires_grad_(True))
        
        self._latent_code=self._latent_code.to("cuda")
        self._latent_code.requires_grad = True
        #self._latent_code.requires_grad = True
        self._decoder=self._decoder.to("cuda")
        self._mlp_decoder=self._mlp_decoder.to("cuda")
        self._cnn_encoder=self._cnn_encoder.to("cuda")
        self._decoder_opacity=self._decoder_opacity.to("cuda")
        self._decoder_rgb = self._decoder_rgb.to("cuda")
        #self._refinement = self._refinement.to("cuda")
    def create_from_points(self,pcl):
       # pcd = np.array(pcd)
       # print(pcd)
        #pcd_colors = np.array([pcdx.colors for pcdx in pcd])
        pcd_points = pcl#np.array([pcdx.points for pcdx in pcd])
       # print(self.colors.shape)
        #self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = pcl

        xyz = fused_point_cloud
        
        #self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #latent=inverse_sigmoid(0.1 * torch.ones((1,256), dtype=torch.float, device="cuda"))
        #self._latent_code=nn.Parameter(latent.requires_grad_(True))
        
        return xyz[0]
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        #self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        #self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            
           # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            #{'params': [self._features_rest], 'lr': training_args.feature_lr, "name": "f_rest"},
          #  {'params': self._refinement.parameters(), 'lr': 0.0001, "name": "opacity"},
            {'params': self._decoder_opacity.parameters(), 'lr': 0.0006, "name": "opacity"},
            {'params': self._decoder_rgb.parameters(), 'lr':0.0006, "name": "f_dc"},
            {'params': self._decoder.parameters(), 'lr': training_args.scaling_lr/10, "name": "decoder"},
           # {'params': [self._latent_code], 'lr': 0.001, "name": "latent_code"},
            {'params':  self._mlp_decoder.parameters(), 'lr': 0.0001, "name": "vertex decoder"},
            {'params':  self._cnn_encoder.parameters(), 'lr': 0.00005, "name": "CNN encoder"}
            
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    def save_checkpoint(self, path):
      #  mkdir_p(os.path.dirname(path))
        torch.save(self._decoder.state_dict(),os.path.join(path, "deltas.pth"))
        torch.save(self._decoder_opacity.state_dict(),os.path.join(path, "opacity.pth"))
        #torch.save(self._features_dc,os.path.join(path, "colors.pth"))
        torch.save(self._latent_code,os.path.join(path, "latent.pth"))
        torch.save(self._decoder_rgb.state_dict(),os.path.join(path, "colors.pth"))
        torch.save(self._mlp_decoder.state_dict(),os.path.join(path, "mlp.pth"))
        torch.save(self._cnn_encoder.state_dict(),os.path.join(path, "encoder.pth"))
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "decoder":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    def load_mlp_decoder(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._mlp_decoder.load_state_dict(weight_dict)
        self._mlp_decoder = self._mlp_decoder.to("cuda")
    def load_encoder(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._cnn_encoder.load_state_dict(weight_dict)
        self._cnn_encoder = self._cnn_encoder.to("cuda")
        #print(self._features_dc)
    def load_colors(self,path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_rgb.load_state_dict(weight_dict)
        self._decoder_rgb = self._decoder_rgb.to("cuda")
        #print(self._features_dc)
    def load_latent(self,path):
        #weight_dict = torch.load(path, map_location="cuda")
        self._latent_code= torch.load(path, map_location="cuda")
       # print(self._features_dc)
    def load_decoder(self, path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder.load_state_dict(weight_dict)
        self._decoder = self._decoder.to("cuda")
    
    def load_decoder_opacity(self, path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_opacity.load_state_dict(weight_dict)
        self._decoder_opacity = self._decoder_opacity.to("cuda")
        
    
    def load_decoder_rgb(self, path):
        weight_dict = torch.load(path, map_location="cuda")
        self._decoder_rgb.load_state_dict(weight_dict)
        self._decoder_rgb = self._decoder_rgb.to("cuda")
        
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

        #translates_delta,rotations_delta=self._decoder(self.get_latent,time)
        #print(translates_delta.T)
        #scales=scales+scales+0.01*scales_delta.T
        #means3D=means3D+0.1*translates_delta.T     
        
        #rotations=rotations+0.01*rotations_delta.T
        
        #rotations=rotations+0.08*rotations_delta.T
      #  xyz = self._xyz.detach().cpu().numpy()
      #  normals = np.zeros_like(xyz)
        #f_dc=f_rest=self._decoder_rgb(self)
     #   f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
     #   f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self.get_opacity.detach().cpu().numpy()
     #   scale = self._scaling.detach().cpu().numpy()
     #   rotation = self._rotation.detach().cpu().numpy()

     #   dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

     #   elements = np.empty(xyz.shape[0], dtype=dtype_full)
     #   attributes = np.concatenate((xyz, normals, f_dc, f_rest, scale, rotation), axis=1)
     #   elements[:] = list(map(tuple, attributes))
     #   el = PlyElement.describe(elements, 'vertex')
     #   PlyData([el]).write(path)

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
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
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