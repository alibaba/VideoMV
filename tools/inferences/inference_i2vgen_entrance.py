import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
import random
import pynvml
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp

from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel

import utils.transforms as data
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION

from utils.camera_utils import get_camera
from core.utils import get_rays

@INFER_ENGINE.register_function()
def inference_i2vgen_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg, cfg_update)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg


def worker(gpu, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    cfg = assign_signle_cfg(cfg, cfg_update, 'vldm_cfg')
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.test_list_path).split('.')[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    test_model = osp.basename(cfg.test_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (exp_name))
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    cfg.log_file = log_file
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    logging.info(f"Going into it2v_fullid_img_text inference on {gpu} gpu")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text="")
    print("Negative:", cfg.negative_prompt)
    _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()
    black_image_feature = torch.zeros([1, 1, cfg.UNet.y_dim]).cuda()

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    # [Model] UNet 
    model = MODEL.build(cfg.UNet)
    checkpoint_dict = torch.load(cfg.test_model, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']
    resume_step = checkpoint_dict['step']
    status = model.load_state_dict(state_dict, strict=False) # for lpips
    logging.info('Load model from {} with status {}'.format(cfg.test_model, status))
    model = model.to(gpu)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()

    # intrinsics
    from core.options import config_defaults
    opt = config_defaults['big']
    # default camera intrinsics
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    elevation = 5
    camera_dist = 1.7
    camera_data = get_camera(cfg.max_frames, elevation=elevation, azimuth_start=0, azimuth_span=360, camera_distance=camera_dist).unsqueeze(0)
    
    camera_data = camera_data.reshape(1,24,4,4)
    camera_data[:,:,1,:] *= -1
    # camera_data[:,:,:3,1:3] *= -1
    camera_data[:,:,[0,1],:] = camera_data[:,:,[1,0],:]
    
    camera_data = camera_data.reshape(1,24,16)

    # prepare gs data
    gs_camera = camera_data.clone().squeeze(0)
    results = {}
    T = gs_camera.shape[0]
    gs_camera = gs_camera.view(T,4,4).contiguous()

    gs_camera[:,1] *= -1
    gs_camera[:,[1, 2]] = gs_camera[:,[2, 1]]
    gs_camera[:,:3,1:3] *= -1

    # c2w[:3, 1:3] *= -1 # invert up and forward direction 
    # camera_data[:,:3,3] /= 0.45 
    cam_dis = np.sqrt(gs_camera[0,0,3]**2 + gs_camera[0,1,3]**2 + gs_camera[0,2,3]**2)

    # normalized camera feats as in paper (transform the first pose to a fixed position)
    transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cam_dis], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(gs_camera[0])
    cam_poses = transform.unsqueeze(0) @ gs_camera  # [V, 4, 4]

    cam_poses_input = cam_poses.clone()

    rays_embeddings = []
    for i in range(T):
        rays_o, rays_d = get_rays(cam_poses_input[i], 256, 256, opt.fovy) # [h, w, 3] 
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        rays_embeddings.append(rays_plucker)

    rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=24, 6, h, w]
    results['input'] = rays_embeddings.unsqueeze(0)

    # opengl to colmap camera for gs renderer
    cam_poses_input[:,:3,1:3] *= -1

    # cameras needed by gaussian rasterizer
    cam_view = torch.inverse(cam_poses_input).transpose(1, 2) # [V, 4, 4]
    cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
    cam_pos = - cam_poses_input[:, :3, 3] # [V, 3]
    
    results['cam_view'] = cam_view.unsqueeze(0)
    results['cam_view_proj'] = cam_view_proj.unsqueeze(0)
    results['cam_pos'] = cam_pos.unsqueeze(0)
    gs_data = results

    # [Test List]
    test_list = open(cfg.test_list_path).readlines()
    test_list = [item.strip() for item in test_list]
    # test_list = ['stucking_cups']
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for item in test_list for _ in range(cfg.round)]

    for idx, line in enumerate(test_list):
        if line.startswith('#'):
            logging.info(f'Skip {line}')
            continue
        logging.info(f"[{idx}]/[{num_videos}] Begin to sample {line} ...")
        img_key = line
        img_name = line
        caption = ""
        captions = [caption]
        try:
            img = Image.open(img_name).convert('RGBA') # 使用4通道Image，背景纯白色
        except:
            continue
        mask = torch.from_numpy(np.array(img.resize((256,256)))[:,:,-1]).unsqueeze(-1)

        width = img.width
        height = img.height
        grey_scale = 255 # fixed the background
        image = Image.new('RGB', size=(width, height), color=(grey_scale,grey_scale,grey_scale))
        image.paste(img,(0,0),mask=img)

        with torch.no_grad():
            image_tensor = vit_trans(image)
            image_tensor = image_tensor.unsqueeze(0)
            y_visual, y_text, y_words = clip_encoder(image=image_tensor, text=captions)
            y_visual = y_visual.unsqueeze(1)

        fps_tensor =  torch.tensor([cfg.target_fps], dtype=torch.long, device=gpu)
        image_id_tensor = train_trans([image]).to(gpu)
        local_image = autoencoder.encode_firsr_stage(image_id_tensor, cfg.scale_factor).detach()
        local_image = local_image.unsqueeze(2).repeat_interleave(repeats=cfg.max_frames, dim=2)

        with torch.no_grad():
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
            # Sample images
            with amp.autocast(enabled=cfg.use_fp16):
                # NOTE: For reproducibility, we have alread recorde the seed ``cur_seed''
                # torch.manual_seed(cur_seed) 
                # cur_seed = torch.get_rng_state()[0]
                # logging.info(f"Current seed {cur_seed}...")
                noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
                noise = noise.to(gpu)
                
                infer_img = black_image_feature if cfg.use_zero_infer else None
                model_kwargs=[
                    {'y': y_words, 'image':y_visual, 'local_image':local_image, 'fps': fps_tensor, 'camera_data': camera_data}, 
                    {'y': zero_y_negative, 'image':infer_img, 'local_image':local_image, 'fps': fps_tensor, 'camera_data': camera_data}]
                
                video_data = diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)

                model_kwargs=[
                    {'y': y_words, 'image':y_visual, 'local_image':local_image, 'fps': fps_tensor, 'camera_data': camera_data, 'gs_data': gs_data}, 
                    {'y': zero_y_negative, 'image':infer_img, 'local_image':local_image, 'fps': fps_tensor, 'camera_data': camera_data, 'gs_data': gs_data}]
                
                video_data_gs = diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model.eval(),
                    autoencoder=autoencoder,
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)

      

        video_data = 1. / cfg.scale_factor * video_data # [1, 4, 32, 46]
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size)
        
        text_size = cfg.resolution[-1]
        cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
        file_name = f'{img_name}_{cfg.world_size:02d}_{cfg.rank:02d}_{cap_name}_{idx:02d}.mp4'
        local_path = os.path.join(cfg.log_dir, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            save_i2vgen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            # NOTE: If you want to visualize the comparison between input and output, you can use the following function.
            # save_i2vgen_video(local_path, image_id_tensor.cpu(), video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            logging.info('Save video to dir %s:' % (local_path))
        except Exception as e:
            logging.info(f'Step: save text or video error with {e}')

        video_data = 1. / cfg.scale_factor * video_data_gs # [1, 4, 32, 46]
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size)
        
        text_size = cfg.resolution[-1]
        cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
        file_name = f'{img_name}_{cfg.world_size:02d}_{cfg.rank:02d}_{cap_name}_{idx:02d}_gs.mp4'
        local_path = os.path.join(cfg.log_dir, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            save_i2vgen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            # NOTE: If you want to visualize the comparison between input and output, you can use the following function.
            # save_i2vgen_video(local_path, image_id_tensor.cpu(), video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            logging.info('Save video to dir %s:' % (local_path))
        except Exception as e:
            logging.info(f'Step: save text or video error with {e}')
    
    logging.info('Congratulations! The inference is completed!')
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

