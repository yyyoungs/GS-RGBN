import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from kiui.op import safe_normalize
import rembg
from core.graphics_utils import *
import random
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



class ObjaverseDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.all_items = []
        self.input_path = self.opt.train_data
        self.input_view = self.opt.view
        load_path = os.path.join(self.input_path, "training.json")
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                self.all_items = json.load(f)
            print(f"Load json file at {load_path}")
        else:
            sub_idx = []
            for idx in os.listdir(self.input_path):
                idx_path = os.path.join(self.input_path, idx)
                for name in os.listdir(idx_path):
                    obj_path = os.path.join(idx_path, name)
                    for k_id in os.listdir(obj_path):
                        if os.path.isdir(os.path.join(obj_path, k_id)):
                            sub_idx.append(os.path.join(obj_path, k_id))      
            json_file_path = os.path.join(self.input_path, "training.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(sub_idx, json_file)
            print(f"Generated json file at {json_file_path}")
            with open(load_path, 'r') as f:
                self.all_items = json.load(f)

        self.items = self.all_items[:]
        print(f"Total items: {len(self.items)}")
        


        
    def img_read(self,image_path,is_known):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.0
        gt_mask = img[..., 3:]
        gt_img = img[..., :3] * gt_mask + (1 - gt_mask)
        # bgr to rgb
        gt_img = gt_img[..., ::-1].copy()
        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)
        gt_mask = torch.from_numpy(gt_mask).permute(2, 0, 1)
        if is_known:
            input_img = gt_img
        else:
            image_path_g = image_path.replace("gt.png","mv_rgb.png")
            img = cv2.imread(image_path_g, cv2.IMREAD_UNCHANGED)
            img = img.astype(np.float32) / 255.0
            input_img = img[..., ::-1].copy()
            input_img = torch.from_numpy(input_img).permute(2, 0, 1)
        ### input normal
        normal_path = image_path.replace("gt.png","mv_normal.png")
        # normal = cv2.imread(normal_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normal = normal.astype(np.float32) / 255.0
        normal = normal[..., ::-1].copy()
        input_normal = torch.from_numpy(normal).permute(2, 0, 1)
        ### input depth
        depth_path = image_path.replace("gt.png","gt_depth0001.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth[depth == 65504] = 0
        depth = np.expand_dims(depth[:,:,0],axis=-1)
        gt_depth = torch.from_numpy(depth).permute(2, 0, 1)
        return gt_img,gt_mask,gt_depth,input_img,input_normal
    
    def img_read_gt_view(self,image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.0
        gt_mask = img[..., 3:]
        gt_img = img[..., :3] * gt_mask + (1 - gt_mask)
        # bgr to rgb
        gt_img = gt_img[..., ::-1].copy()
        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)
        gt_mask = torch.from_numpy(gt_mask).permute(2, 0, 1)
        depth_path = image_path.replace("gt.png","gt_depth0001.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth[depth == 65504] = 0
        depth = np.expand_dims(depth[:,:,0],axis=-1)
        gt_depth = torch.from_numpy(depth).permute(2, 0, 1)
        input_img = None
        input_normal = None
        return gt_img,gt_mask,gt_depth,input_img,input_normal
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        # print(uid)
        results = {}
        input_images = []
        input_normals = []
        gt_images = []
        gt_depths = []
        masks = []
        world_view_transforms = []
        full_proj_transforms = []
        camera_centers = []
        rays_ori = []
        rays_down = []
        c2ws = []
        w2cs = []
        transformsfile = os.path.join(uid, "transforms_train.json").replace("\\","/")
        name = transformsfile.split("/")[-3]
        with open(transformsfile) as json_file:
            ###
            contents = json.load(json_file)
            ###
            fovx = contents["camera_angle_x"]
            fx = contents["fl_x"]
            fy = contents["fl_y"]
            cx = contents["cx"]
            cy = contents["cy"]
            frames = contents["frames"]
            zfar = 100.0
            znear = 0.01
            for idx, frame in enumerate(frames):
                image_path = frame["file_path"].replace("\\","/").split("/")[2]+"/gt.png"
                image_path = os.path.join(uid,image_path)
                if idx == 0:
                    gt_img,gt_mask,gt_depth,input_img,input_normal = self.img_read(image_path,is_known=True)
                elif idx<self.input_view:
                    gt_img,gt_mask,gt_depth,input_img,input_normal = self.img_read(image_path,is_known=False)
                else:
                    gt_img,gt_mask,gt_depth,input_img,input_normal = self.img_read_gt_view(image_path)
                image_height = gt_img.shape[1]
                image_width = gt_img.shape[2]
                if input_img is not None:
                    input_images.append(input_img)
                    input_normals.append(input_normal)
                #####
                gt_images.append(gt_img)
                gt_depths.append(gt_depth)
                masks.append(gt_mask)
                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"],dtype='float32')
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                # c2w[[1, 2]] = c2w[[2, 1]]
                c2w[:3, 1:3] *= -1
                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
                fovy = focal2fov(fov2focal(fovx, image_width), image_height)
                FovY = fovy 
                FovX = fovx
                results["FovY"] = FovY
                results["FovX"] = FovX
                K = torch.zeros(2,3)
                K[0,0] = fx
                K[1,1] = fy
                K[0,2] = cx
                K[1,2] = cy
                world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
                projection_matrix = getProjectionMatrixCorrect(znear=znear, zfar=zfar, H=image_height, W=image_width, K=K).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]  
                ray_down = build_rays(c2ws=c2w,ixts=K,H=image_height,W=image_width,scale=1.0/16)
                ray_ori = build_rays(c2ws=c2w,ixts=K,H=image_height,W=image_width,scale=1.0)
                world_view_transforms.append(world_view_transform)
                full_proj_transforms.append(full_proj_transform)
                camera_centers.append(camera_center)
                rays_down.append(ray_down)
                rays_ori.append(ray_ori)
                results["ixt"] = K
                c2ws.append(torch.tensor(c2w))
                w2cs.append(torch.tensor(w2c))
                ##
        
        results["input_images"] = torch.stack(input_images, dim=0)
        results["input_normals"] = torch.stack(input_normals, dim=0)
        results["gt_images"] = torch.stack(gt_images, dim=0)
        results["gt_masks"] = torch.stack(masks, dim=0)
        results["world_view_transforms"] = torch.stack(world_view_transforms, dim=0)
        results["full_proj_transforms"] = torch.stack(full_proj_transforms, dim=0)
        results["camera_centers"] = torch.stack(camera_centers, dim=0)
        results["rays_down"] = torch.stack(rays_down, dim=0)
        results["rays_ori"] = torch.stack(rays_ori, dim=0)
        results["c2ws"] = torch.stack(c2ws, dim=0)
        results["w2c"] = torch.stack(w2cs, dim=0)
        results["gt_depths"] = torch.stack(gt_depths, dim=0)
        results["name"] = name
        
        return results
        
                
