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
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.flexible_deform_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset

class Scene:

    gaussians : GaussianModel
    
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None):
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
        
        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](args.source_path)
            print("Found poses_bounds.py and extra marks with EndoNeRF")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, args.white_background, args.eval)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert False, "Could not recognize scene type!"
                
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # self.cameras_extent = args.camera_extent
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras 
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras 
        print("Loading Video Cameras")
        self.video_camera =  scene_info.video_cameras 
        
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        # self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
            self.gaussians.load_color_mlp(os.path.join(self.model_path, "color_mlp.pth"))
            self.gaussians.load_tone_mapper(os.path.join(self.model_path, "tone_mapper.pth"))
            self.load_exposure_time(os.path.join(self.model_path, "exposure_time.json"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, args.camera_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # self.gaussians.save_deformation(point_cloud_path)
        self.gaussians.save_color_mlp(os.path.join(self.model_path, "color_mlp.pth"))
        self.gaussians.save_tone_mapper(os.path.join(self.model_path, "tone_mapper.pth"))
        self.save_exposure_time(os.path.join(self.model_path, "exposure_time.json"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
    def save_exposure_time(self, path):
        rows = []
        for camera in self.train_camera:
            rows.append({
                "image": camera.image_name,
                "exposure_time": float(camera.exposure_time.detach().cpu().item())
            })
        for camera in self.test_camera:
            rows.append({
                "image": camera.image_name,
                "exposure_time": float(camera.exposure_time.detach().cpu().item())
            })

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    def load_exposure_time(self, path):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        exp_map = {r["image"]: float(r["exposure_time"]) for r in rows}

        def apply(cams):
            for cam in cams:
                v = exp_map.get(cam.image_name, None)
                with torch.no_grad():
                    cam.exposure_time.data.fill_(v)

        apply(self.train_camera)
        apply(self.test_camera)
        apply(self.video_camera)
