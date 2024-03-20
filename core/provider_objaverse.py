import os, json
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
from PIL import Image

def get_intr():
    h, w = 32, 32
    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    return K

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    cond_camera_matrix = np.eye(4)
    cond_camera_matrix[:3, 0] = np.array(json_content['x'])
    cond_camera_matrix[:3, 1] = -np.array(json_content['y'])
    cond_camera_matrix[:3, 2] = -np.array(json_content['z'])
    cond_camera_matrix[:3, 3] = np.array(json_content['origin'])


    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])

    return camera_matrix, cond_camera_matrix

class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        self.items = []
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())

        # @NOTE instead we read json
        items = []
        data_dir = "/mnt/objaverse/dataset/raw/0"
        item_path = "/mnt/objaverse/dataset/raw/0/lvis_thres_28.json"

        lines = json.load(open(item_path))
        lines = [os.path.join(data_dir, item) for item in lines]
        items.extend(lines)

        # self.items = items[:10]*10000
        self.items = items

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        print(self.proj_matrix)



    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            # vids = np.random.permutation(np.arange(0, 24))[:self.opt.num_input_views].tolist() + np.random.permutation(24).tolist()
            vids = np.arange(0, self.opt.num_input_views).tolist()
        else:
            # fixed views
            # vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
            vids = np.arange(0, self.opt.num_input_views).tolist()
        
        for vid in vids:
            image_path = os.path.join(uid, f'campos_512_v4/{vid:05d}/{vid:05d}.png')
            camera_path = os.path.join(uid, f'campos_512_v4/{vid:05d}/{vid:05d}.json')
            # print(image_path, camera_path)
            try:
                # TODO: load data (modify self.client here)
                # image = np.frombuffer(self.client.get(image_path), np.uint8)
                image = torch.from_numpy(np.array(Image.open(image_path))/255)
                _, c2w = read_camera_matrix_single(camera_path)
                c2w = torch.from_numpy(c2w).float()
                # mask = torch.from_numpy(np.array(img.resize((self.resolution[1], self.resolution[0])))[:,:,-1]).unsqueeze(-1)
                # mask_list.append(mask)
                # width = img.width
                # height = img.height
                # # grey_scale = random.randint(128, 130) # random gray color
                # grey_scale = 255
                # image = Image.new('RGB', size=(width, height), color=(grey_scale,grey_scale,grey_scale))
                # image.paste(img,(0,0),mask=img)
                
                # image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                # c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                # c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue

        # frame_list = []
        # camera_list = []
        # mask_list = []
        # for frame_idx, frame_path in enumerate(frames_path):
        #     img = Image.open(frame_path).convert('RGBA')
        #     mask = torch.from_numpy(np.array(img.resize((self.resolution[1], self.resolution[0])))[:,:,-1]).unsqueeze(-1)
        #     mask_list.append(mask)
        #     width = img.width
        #     height = img.height
        #     grey_scale = random.randint(128, 130) # random gray color
        #     image = Image.new('RGB', size=(width, height), color=(grey_scale,grey_scale,grey_scale))
        #     image.paste(img,(0,0),mask=img)
        #     frame_list.append(image.resize((self.resolution[1], self.resolution[0])))

        #     _, camera_embedding = read_camera_matrix_single(camera_path[frame_idx])
        #     camera_list.append(torch.from_numpy(camera_embedding.flatten().astype(np.float32)))

        # camera_data = torch.stack(camera_list, dim=0) # [24,16]
        # mask_data = torch.stack(mask_list, dim=0) 
        # video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])

            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            # c2w[1] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale # 不需要scale, camera_distance会变
            c2w[:3,3] /= 0.45
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            # image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5/0.45], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()
        # print(cam_poses_input.shape)
        # data augmentation
        # if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            # if random.random() < self.opt.prob_grid_distortion:
            #     images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            # if random.random() < self.opt.prob_cam_jitter:
            #     cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input.half()

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results