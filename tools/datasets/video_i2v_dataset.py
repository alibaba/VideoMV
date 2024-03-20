import os
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.registry_class import DATASETS
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

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

@DATASETS.register_class()
class Video_I2V_Dataset(Dataset):
    def __init__(self, 
            data_list,
            data_dir_list,
            caption_dir,
            max_words=1000,
            resolution=(384, 256),
            vit_resolution=(224, 224),
            max_frames=16,
            sample_fps=8,
            transforms=None,
            vit_transforms=None,
            get_first_frame=True, 
            prepare_lgm=False,
            **kwargs):

        self.prepare_lgm = prepare_lgm
        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = resolution
        self.vit_resolution = vit_resolution
        self.sample_fps = sample_fps
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.get_first_frame = get_first_frame

        # @NOTE instead we read json
        image_list = []
        # self.captions = json.load(open(caption_dir))
        self.captions = None
        for item_path, data_dir in zip(data_list, data_dir_list):
            lines = json.load(open(item_path))
            lines = [[data_dir, item] for item in lines]
            image_list.extend(lines)
        self.image_list = image_list
        self.replica = 1000

        if self.prepare_lgm:
            from core.options import config_defaults
            self.opt = config_defaults['big']
            # default camera intrinsics
            self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
            self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
            self.proj_matrix[0, 0] = 1 / self.tan_half_fov
            self.proj_matrix[1, 1] = 1 / self.tan_half_fov
            self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
            self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
            self.proj_matrix[2, 3] = 1

    def __getitem__(self, index):
        index = index % len(self.image_list)
        data_dir, file_path = self.image_list[index]
        video_key = file_path
        caption = ""

        try:
            ref_frame, vit_frame, video_data, fullreso_video_data, camera_data, mask_data, fullreso_mask_data = self._get_video_data(data_dir, file_path)
            if self.prepare_lgm:
                results = self.prepare_gs(camera_data.clone(), fullreso_mask_data.clone(), fullreso_video_data.clone())
                results['images_output'] = fullreso_video_data # GT renderings of [512, 512] resolution in the range [0,1]
        except Exception as e:
            print(e)
            return self.__getitem__((index+1)%len(self)) # next available data

        if self.prepare_lgm:
            return results, ref_frame, vit_frame, video_data, camera_data, mask_data, caption, video_key
        else:
            return ref_frame, vit_frame, video_data, camera_data, mask_data, caption, video_key

    def prepare_gs(self, camera_data, mask_data, video_data): # mask_data [24,512,512,1]

        results = {}
        
        mask_data = mask_data.permute(0,3,1,2) 
        results['masks_output'] = mask_data/255.0 # TODO normalize to [0, 1]

        T = camera_data.shape[0]
        camera_data = camera_data.view(T,4,4).contiguous()
        
        camera_data[:,1] *= -1
        camera_data[:,[1, 2]] = camera_data[:,[2, 1]]
        cam_dis = np.sqrt(camera_data[0,0,3]**2 + camera_data[0,1,3]**2 + camera_data[0,2,3]**2)

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cam_dis], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(camera_data[0])
        cam_poses = transform.unsqueeze(0) @ camera_data  # [V, 4, 4]

        cam_poses_input = cam_poses.clone()

        rays_embeddings = []
        for i in range(T):
            rays_o, rays_d = get_rays(cam_poses_input[i], 256, 256, self.opt.fovy) # [h, w, 3] 
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=24, 6, h, w]
        results['input'] = rays_embeddings

        # opengl to colmap camera for gs renderer
        cam_poses_input[:,:3,1:3] *= -1

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses_input).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses_input[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        return results

    def _get_video_data(self, data_dir, file_path):
        prefix = os.path.join(data_dir, file_path, 'campos_512_v4')

        frames_path = [os.path.join(prefix, "{:05d}/{:05d}.png".format(frame_idx, frame_idx)) for frame_idx in range(24)]
        camera_path = [os.path.join(prefix, "{:05d}/{:05d}.json".format(frame_idx, frame_idx)) for frame_idx in range(24)]

        frame_list = []
        fullreso_frame_list = []
        camera_list = []
        mask_list = []
        fullreso_mask_list = []
        for frame_idx, frame_path in enumerate(frames_path):
            img = Image.open(frame_path).convert('RGBA')
            mask = torch.from_numpy(np.array(img.resize((self.resolution[1], self.resolution[0])))[:,:,-1]).unsqueeze(-1)
            mask_list.append(mask)
            fullreso_mask = torch.from_numpy(np.array(img)[:,:,-1]).unsqueeze(-1)
            fullreso_mask_list.append(fullreso_mask)

            width = img.width
            height = img.height
            grey_scale = 255 
            image = Image.new('RGB', size=(width, height), color=(grey_scale,grey_scale,grey_scale))
            image.paste(img,(0,0),mask=img)

            fullreso_frame_list.append(torch.from_numpy(np.array(image)/255.0).float()) # for LGM rendering NOTE notice the data range [0,1]
            frame_list.append(image.resize((self.resolution[1], self.resolution[0])))

            _, camera_embedding = read_camera_matrix_single(camera_path[frame_idx])
            camera_list.append(torch.from_numpy(camera_embedding.flatten().astype(np.float32)))

        camera_data = torch.stack(camera_list, dim=0) # [24,16]
        mask_data = torch.stack(mask_list, dim=0) 
        fullreso_mask_data = torch.stack(fullreso_mask_list, dim=0) 

        video_data = torch.zeros(self.max_frames, 3,  self.resolution[1], self.resolution[0])

        fullreso_video_data = torch.zeros(self.max_frames, 3,  512, 512)

        if self.get_first_frame:
            ref_idx = 0
        else:
            ref_idx = int(len(frame_list)/2)

        mid_frame = copy(frame_list[ref_idx])
        vit_frame = self.vit_transforms(mid_frame)
        frames = self.transforms(frame_list)
        video_data[:len(frame_list), ...] = frames

        if True: # random augmentation
            split_idx = np.random.randint(0, len(frame_list))
            video_data = torch.cat([video_data[split_idx:], video_data[:split_idx]], dim=0)

        fullreso_video_data[:len(fullreso_frame_list), ...] = torch.stack(fullreso_frame_list, dim=0).permute(0,3,1,2)

        ref_frame = copy(frames[ref_idx])
        
        return ref_frame, vit_frame, video_data, fullreso_video_data, camera_data, mask_data, fullreso_mask_data

    def __len__(self):
        return len(self.image_list)*self.replica

