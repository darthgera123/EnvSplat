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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.dataset_readers_exr import sceneLoadTypeCallbacks_exr
from scene.dataset_readers_exr_light import sceneLoadTypeCallbacks_exr_light
from scene.gaussian_model import GaussianModel, GaussianModel_exr, \
     GaussianModel_exr_encoder_sh, GaussianModel_uv_sh, GaussianModel_uv_sh_relit,\
     GaussianModel_uv_relit, GaussianModel_uv_prior,GaussianModel_uv_prior_relit, \
     GaussianModel_uv_prior_latent,GaussianModel_uv_prior_latent_all, \
    GaussianModel_uv_prior_posenc, GaussianModel_uv_prior_scale, GaussianModel_uv_prior_simple, \
    GaussianModel_unet, GaussianModel_latent
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    

class Scene_exr:

    gaussians : GaussianModel_exr

    def __init__(self, args : ModelParams, gaussians : GaussianModel_exr, load_iteration=None, shuffle=True, resolution_scales=[1.0],load_init=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.load_init:
            self.gaussians.load_ply(self.load_init)
            self.gaussians.freeze_positions()
            print("Load Initialized point cloud")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

   
class Scene_exr_encoder_sh:

    gaussians : GaussianModel_exr_encoder_sh

    def __init__(self, args : ModelParams, gaussians : GaussianModel_exr_encoder_sh, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud_init.ply"))
           self.gaussians.load_decoder_shape(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "shape.pth"))
            
           self.gaussians.load_encoder(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "encoder.pth"))
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
                                                           
        # elif self.load_init:
        #     self.gaussians.load_ply(self.load_init)
        #     self.gaussians.freeze_positions()
        #     print("Load Initialized point cloud")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # self.gaussians.load_ply(os.path.join(self.load_geometry_path,
            #                                                "point_cloud",
            #                                                "iteration_" + str(self.load_init),
            #                                                "point_cloud.ply"))
            # self.gaussians._decoder_rgb = self.
            
            # self.gaussians.load_decoder_rgb(os.path.join(self.load_geometry_path,
            #                                                "point_cloud",
            #                                                "iteration_" + str(self.load_init),
            #                                                "colors.pth"))
            # self.gaussians._encoder = self.gaussians._encoder.to('cuda')

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class Scene_latent:

    gaussians : GaussianModel_latent

    def __init__(self, args : ModelParams, gaussians : GaussianModel_latent, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # if args.load_geometry:
        #     if self.load_init == -1:
        #         self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
        #     else:
        #         self.load_init = load_init
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud_init.ply"))
           self.gaussians.load_decoder_shape(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "shape.pth"))
           self.gaussians.load_decoder_rgb(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "color.pth"))
           self.gaussians.load_latent(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "latent.pth"))
            
           
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
                                                           
        elif self.load_geometry_path != "":
            """
            Need to update this as well
            """
            self.gaussians.load_ply(os.path.join(self.load_geometry_path,
                                                           "point_cloud_init.ply"))
            self.gaussians.load_decoder_shape(os.path.join(self.load_geometry_path,
                                                           "shape.pth"))
            self.gaussians.load_decoder_rgb(os.path.join(self.load_geometry_path,
                                                           "color.pth"))
            print("Load Initialized point cloud")
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # self.gaussians.load_ply(os.path.join(self.load_geometry_path,
            #                                                "point_cloud",
            #                                                "iteration_" + str(self.load_init),
            #                                                "point_cloud.ply"))
            
            
            # self.gaussians.load_decoder_rgb(os.path.join(self.load_geometry_path,
            #                                                "point_cloud",
            #                                                "iteration_" + str(self.load_init),
            #                                                "colors.pth"))
            # self.gaussians._encoder = self.gaussians._encoder.to('cuda')

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    


class Scene_uv_sh:

    gaussians : GaussianModel_uv_sh

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_sh, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
           self.gaussians.load_decoder_shape(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "shape.pth"))
            
           self.gaussians.load_encoder(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "encoder.pth"))
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
           
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            
            if self.load_geometry_path:
                self.gaussians.load_decoder_shape(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "shape.pth"))
                # from utils.network import Encoder_Unet
                # self.gaussians._encoder = Encoder_Unet(in_ch=1536+6).cuda()
                
                self.gaussians.load_encoder(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "encoder.pth"))
                # self.gaussians.load_encoder_light(os.path.join(self.load_geometry_path,
                #                                            "point_cloud",
                #                                            "iteration_" + str(self.load_init),
                #                                            "encoder.pth"),in_ch=6+1536)
                
            else:
                from utils.network import Encoder_Unet
                self.gaussians._encoder = Encoder_Unet(in_ch=1536+6).cuda()


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class Scene_uv_relit:

    gaussians : GaussianModel_uv_relit

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_relit, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
           self.gaussians.load_decoder_shape(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "shape.pth"))
            
           self.gaussians.load_encoder(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "encoder.pth"))
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            
            if self.load_geometry_path:
                # self.gaussians.load_decoder_shape(os.path.join(self.load_geometry_path,
                #                                            "point_cloud",
                #                                            "iteration_" + str(self.load_init),
                #                                            "shape.pth"))
                # from utils.network import Encoder_Unet
                # self.gaussians._encoder = Encoder_Unet(in_ch=1536+6).cuda()
                
                # self.gaussians.load_encoder(os.path.join(self.load_geometry_path,
                #                                            "point_cloud",
                #                                            "iteration_" + str(self.load_init),
                #                                            "encoder.pth"))
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_sh_relit:

    gaussians : GaussianModel_uv_sh_relit

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_sh_relit, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
           self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           self.gaussians.load_decoder_shape(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "shape.pth"))
            
           self.gaussians.load_encoder(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "encoder.pth"))
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            
            if self.load_geometry_path:
                # self.gaussians.load_decoder_shape(os.path.join(self.load_geometry_path,
                #                                            "point_cloud",
                #                                            "iteration_" + str(self.load_init),
                #                                            "shape.pth"))
                # from utils.network import Encoder_Unet
                # self.gaussians._encoder = Encoder_Unet(in_ch=1536+6).cuda()
                
                # self.gaussians.load_encoder(os.path.join(self.load_geometry_path,
                #                                            "point_cloud",
                #                                            "iteration_" + str(self.load_init),
                #                                            "encoder.pth"))
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior:

    gaussians : GaussianModel_uv_prior

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        self.gt_geometry = args.gt_geometry

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)        
           if self.gt_geometry:
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply")) 
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
            if self.gt_geometry:
                
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_simple:

    gaussians : GaussianModel_uv_prior_simple

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_simple, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        self.gt_geometry = args.gt_geometry

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)        
           if self.gt_geometry:
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply")) 
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
            if self.gt_geometry:
                
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_unet:

    gaussians : GaussianModel_unet

    def __init__(self, args : ModelParams, gaussians : GaussianModel_unet, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        self.gt_geometry = args.gt_geometry

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
        #    self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.load_init),
        #                                                    "point_cloud.ply"))
           
            
        #    self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)        
           if self.gt_geometry:
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply")) 
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            # self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
            if self.gt_geometry:
                
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_scale:

    gaussians : GaussianModel_uv_prior_scale

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_scale, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        self.gt_geometry = args.gt_geometry
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)        
           if self.gt_geometry:
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
            if self.gt_geometry:
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_posenc:

    gaussians : GaussianModel_uv_prior_posenc

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_posenc, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)        
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_relit:

    gaussians : GaussianModel_uv_prior_relit

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_relit, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = 0    
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
        
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_latent:

    gaussians : GaussianModel_uv_prior_latent

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_latent, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
        
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene_uv_prior_latent_all:

    gaussians : GaussianModel_uv_prior_latent_all

    def __init__(self, args : ModelParams, gaussians : GaussianModel_uv_prior_latent_all, load_iteration=None, shuffle=True,resolution_scales=[1.0],load_init=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_init = load_init
        self.gaussians = gaussians
        self.load_geometry_path = args.load_geometry
        self.relit_path = args.relit_path
        self.gt_geometry = args.gt_geometry

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if args.load_geometry:
            if self.load_init == -1:
                self.load_init = searchForMaxIteration(os.path.join(self.load_geometry_path, "point_cloud"))
            else:
                self.load_init = load_init
            print("initializing encoder, decoder")
        
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks_exr_light["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks_exr_light["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.spatial_lr_scale = self.cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
           
        #    self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
           self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 
           self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
           
            
           self.gaussians.load_relit(os.path.join(self.relit_path))
           if 'relit.pth' in self.relit_path:
               print("Hello")
               self.loaded_iter = load_iteration

           else:
               self.loaded_iter = None
        
           if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
        
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            print(os.path.join(args.source_path, "mask.png"))
            if os.path.exists(os.path.join(args.source_path, "mask.png")):
                from imageio.v2 import imread
                mask = imread(os.path.join(args.source_path, "mask.png"))/255.0
                self.gaussians.set_mask(mask)
            # Relit model path
            self.gaussians.load_relit(os.path.join(self.relit_path))
            if 'relit.pth' in self.relit_path:
               self.loaded_iter = None
            else:
               self.loaded_iter = load_iteration
            # Relit diffuse geometry
            
            if self.load_geometry_path:
                self.gaussians.load_geometry(os.path.join(self.load_geometry_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.load_init),
                                                           "point_cloud.ply"))
            # GT geometry effects
            if self.gt_geometry:
                
                self.gaussians.load_geometry_gt(os.path.join(self.gt_geometry,
                                                           "point_cloud",
                                                           "iteration_20000",
                                                           "point_cloud.ply"))
                
            


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_checkpoint(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]