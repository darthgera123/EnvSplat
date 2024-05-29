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
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision
import torch.nn as nn
from math import exp
from functools import reduce


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l1_loss_exp(network_output, gt):
    pred_log = torch.log(1+network_output)
    gt_log = torch.log(1+gt)
    return torch.abs((pred_log-gt_log)).mean()




# class PerceptualLoss(nn.Module):
#     def __init__(self, device='cuda'):
#         super(PerceptualLoss, self).__init__()
#         vgg = models.vgg19(pretrained=True).features
#         vgg = vgg.to(device).eval()
#         self.features = nn.Sequential()
#         self.device = device

#         # Define the layers to use for feature extraction
#         layer_names = ['0', '5', '10', '19', '28']
#         for name, layer in vgg._modules.items():
#             self.features.add_module(name, layer)
#             if name in layer_names:
#                 break

#     def forward(self, network_output, gt):
#         # Normalize the input images
#         # x = (x + 1.0) / 2.0
#         # y = (y + 1.0) / 2.0
#         pred_log = torch.log(1+network_output)
#         gt_log = torch.log(1+gt)
#         # Extract features from the intermediate layers
#         features_x = self.features(pred_log)
#         features_y = self.features(gt_log)

#         # Compute the perceptual loss as the L1 distance between features
#         # loss = F.l1_loss(features_x, features_y)
#         loss = F.l1_loss(features_x, features_y)

#         return loss

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[4, 9, 16, 23], requires_grad=False):
        super(PerceptualLoss, self).__init__()
        # Load the VGG16 model pretrained on ImageNet
        vgg = models.vgg19(pretrained=True).features
        vgg = vgg.cuda()
        # vgg = vgg16(pretrained=True).features
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad = False

        # Define slices of VGG to use for perceptual loss
        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg[i] for i in range(layers[j])])
            for j in range(len(layers))
        ])

        # Normalization mean and std for ImageNet
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def forward(self, x, y):
        # Normalize inputs
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # Compute loss as sum of L1 loss across each slice
        loss = 0
        for slice_module in self.slices:
            x_slice = slice_module(x)
            y_slice = slice_module(y)
            loss += nn.functional.l1_loss(x_slice, y_slice)
        return loss

# class VGGLoss(nn.Module):
#     """Computes the VGG perceptual loss between two batches of images.
#     The input and target must be 4D tensors with three channels
#     ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
#     normalized to the range [0,1].
#     The VGG perceptual loss is the mean squared difference between the features
#     computed for the input and target at layer :attr:`layer` (default 8, or
#     ``relu2_2``) of the pretrained model specified by :attr:`model` (either
#     ``'vgg16'`` (default) or ``'vgg19'``).
#     If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
#     pixels in both height and width will be applied to all images in the input
#     and target. The shift will only be applied when the loss function is in
#     training mode, and will not be applied if a precomputed feature map is
#     supplied as the target.
#     :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
#     similarly to the loss functions in :mod:`torch.nn`. The default is
#     ``'mean'``.
#     :meth:`get_features()` may be used to precompute the features for the
#     target, to speed up the case where inputs are compared against the same
#     target over and over. To use the precomputed features, pass them in as
#     :attr:`target` and set :attr:`target_is_features` to :code:`True`.
#     Instances of :class:`VGGLoss` must be manually converted to the same
#     device and dtype as their inputs.

#     From: https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py
#     """

#     models = {
#         'vgg16': models.vgg16,
#         'vgg19': models.vgg19
#         }

#     def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
#         super().__init__()
#         self.shift = shift
#         self.reduction = reduction
#         self.normalize = torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         self.model = self.models[model](pretrained=True).features[:layer+1]
#         self.model.eval()
#         self.model.requires_grad_(False)
#         self.model = self.model.cuda()

#     def get_features(self, input):
#         return self.model(self.normalize(input))

#     def train(self, mode=True):
#         self.training = mode

#     def forward(self, input, target, target_is_features=False):
#         if target_is_features:
#             input_feats = self.get_features(input)
#             target_feats = target
#         else:
#             sep = input.shape[0]
#             batch = torch.cat([input, target])
#             if self.shift and self.training:
#                 padded = F.pad(batch, [self.shift] * 4, mode='replicate')
#                 batch = torchvision.transforms.RandomCrop(batch.shape[2:])(padded)

#             feats = self.get_features(batch)
#             input_feats, target_feats = feats[:sep], feats[sep:]

#         return F.mse_loss(input_feats, target_feats, reduction=self.reduction)
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5, wt_mode=0):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        if wt_mode == 0:
            self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        elif wt_mode == 1:
            self.weights = (1.0, 0.1, 0.0001, 0.0, 0.0)
        elif wt_mode == 2:
            self.weights = (1.0, 1.0, 0.1, 0.10, 0.0001)
        else:
            raise NotImplementedError

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        # source, target = (source + 1) / 2, (target + 1) / 2
        source = (source-self.mean) / self.std
        target = (target-self.mean) / self.std
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)

        return loss 




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x/self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out


def downsample_image(image_tensor):
    # Downsample the image by a factor of 2 using bilinear interpolation
    
    output_tensor = F.interpolate(image_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
    return output_tensor

    
class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        cosine_dist_zero_2_one = cosine_dist_zero_2_one.clamp(0,1)
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen = downsample_image(gen.unsqueeze(0))    
        tar = downsample_image(tar.unsqueeze(0))
        
        
        
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content
        
        return self.style_loss + self.content_loss

        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss

