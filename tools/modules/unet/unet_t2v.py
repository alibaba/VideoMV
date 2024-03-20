import math
import random
import torch
import xformers
import xformers.ops
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper

from .util import *
# from .mha_flash import FlashAttentionBlock
from utils.registry_class import MODEL

# LGM utils
import tyro
import time
import random
import kiui
import torch
from safetensors.torch import load_file

import torch.cuda.amp as amp
import sys
USE_TEMPORAL_TRANSFORMER = True

from PIL import Image
import numpy as np

def get_intr():
    h, w = 256, 256
    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    return K

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)

def q_sample(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, x0, t, noise):
    r"""Sample from q(x_t | x_0).
    """
    return _i(sqrt_alphas_cumprod, t, x0) * x0 + \
            _i(sqrt_one_minus_alphas_cumprod, t, x0) * noise


@MODEL.register_class()
class UNetSD_T2VBase(nn.Module):
    def __init__(self,
            config=None,
            in_dim=4,
            dim=512,
            y_dim=512,
            context_dim=512,
            hist_dim = 156,
            dim_condition=4,
            out_dim=6,
            num_tokens=4,
            dim_mult=[1, 2, 3, 4],
            num_heads=None,
            head_dim=64,
            camera_dim=16,
            num_res_blocks=3,
            attn_scales=[1 / 2, 1 / 4, 1 / 8],
            use_scale_shift_norm=True,
            dropout=0.1,
            temporal_attn_times=1,
            temporal_attention = True,
            use_checkpoint=False,
            use_image_dataset=False,
            use_sim_mask = False,
            training=True,
            inpainting=True,
            use_fps_condition=False,
            use_camera_condition=False,
            use_lgm_refine=False,
            p_all_zero=0.1,
            p_all_keep=0.1,
            zero_y = None,
            adapter_transformer_layers = 1,
            **kwargs):
        super(UNetSD_T2VBase, self).__init__()
        
        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        self.zero_y = zero_y
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        ### for temporal attention
        self.num_heads = num_heads
        ### for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask
        self.training=training
        self.inpainting = inpainting
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep
        self.use_fps_condition = use_fps_condition
        self.use_camera_condition = use_camera_condition 
        self.camera_dim = camera_dim
        self.use_lgm_refine = use_lgm_refine
        
        if self.use_lgm_refine:
            from core.options import config_defaults
            from core.models import LGM
            lgm_opt = config_defaults['big'] # 利用pretrain
            self.lgm_big = LGM(lgm_opt)

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), # [320,1280]
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        if self.use_camera_condition: # add camera if you want to do 3D Generation rather than a 
            self.camera_embedding = nn.Sequential(
                nn.Linear(self.camera_dim, embed_dim), 
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            nn.init.zeros_(self.camera_embedding[-1].weight)
            nn.init.zeros_(self.camera_embedding[-1].bias)
        
        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(heads = num_heads, max_distance = 32)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformer(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset)])
                if scale in attn_scales:
                    block.append(
                            SpatialTransformer(
                                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                                disable_self_attn=False, use_linear=True
                            )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(TemporalTransformer(out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
        
        self.middle_block = nn.ModuleList([
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,),
            SpatialTransformer(
                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                disable_self_attn=False, use_linear=True
            )])        
        
        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                 TemporalTransformer(
                            out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal,
                            multiply_zero=use_image_dataset,
                        )
                )
            else:
                self.middle_block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))        

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(in_dim + shortcut_dims.pop(), embed_dim, dropout, out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset, )])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=1024,
                            disable_self_attn=False, use_linear=True
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset
                                    )
                            )
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)
        # print(len(self.middle_block), len(self.output_blocks)) # 4 12
        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)

    def resume_lgm(self, path):
        # pass
        ckpt = load_file(path, device='cpu')
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = self.lgm_big.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')


    def forward(self, 
        x,
        t,
        x0=None,
        gs_data = None,
        sqrt_alphas_cumprod=None,
        sqrt_one_minus_alphas_cumprod=None, # x0, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod for x_{t-1}
        sqrt_recip_alphas_cumprod=None,
        sqrt_recipm1_alphas_cumprod=None, # use to sample fake_x0
        autoencoder=None,
        y = None,
        fps = None,
        masked = None,
        camera_data = None,
        video_mask = None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0,  # mask last frame num
        **kwargs):
        
        xt = x.clone().detach()
        assert self.inpainting or masked is None, 'inpainting is not supported'

        batch, c, f, h, w= x.shape
        device = x.device
        self.batch = batch

        #### image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        else:
            time_rel_pos_bias = None
        
        # [Embeddings]
        if self.use_fps_condition and fps is not None:
            embeddings = self.time_embed(sinusoidal_embedding(t, self.dim)) + self.fps_embedding(sinusoidal_embedding(fps, self.dim))
        else:
            embeddings = self.time_embed(sinusoidal_embedding(t, self.dim))
        embeddings = embeddings.repeat_interleave(repeats=f, dim=0)

        # [Camera Embeddings]
        if self.use_camera_condition and camera_data is not None:
            # print("add camera conditions.")
            camera_emb = rearrange(camera_data, 'b f c -> (b f) c')
            camera_emb = self.camera_embedding(camera_emb)
            # camera_data = rearrange(camera_data, '(b f) c -> b f c', b=batch)  # no need to turn it back.
            embeddings = embeddings + camera_emb


        # [Context] text prompt feature
        context = x.new_zeros(batch, 0, self.context_dim)
        if y is not None:
            y_context = y
            context = torch.cat([context, y_context], dim=1)
        else:
            y_context = self.zero_y.repeat(batch, 1, 1)[:, :1, :]
            context = torch.cat([context, y_context], dim=1)
        context = context.repeat_interleave(repeats=f, dim=0)

        x = rearrange(x, 'b c f h w -> (b f) c h w')
        xs = []
        for block in self.input_blocks:
            x, name = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)
        
        # middle
        for block in self.middle_block:
            x, name = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)


        # decoder
        for index, block in enumerate(self.output_blocks):
            x = torch.cat([x, xs.pop()], dim=1)
            x, name = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x) # [32, 4, 32, 32]

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)

        if self.use_lgm_refine and x0 is not None: 
            fake_x0 = _i(sqrt_recip_alphas_cumprod, t, xt) * xt - _i(sqrt_recipm1_alphas_cumprod, t, xt) * x
            total_frames = fake_x0.shape[1]
            idxs = np.random.permutation(np.arange(0, 24))[:4].tolist() 
            decode_fake_x0 = fake_x0[:,:,idxs] # B c 4 H W
            decode_fake_x0 = rearrange(decode_fake_x0, 'b c f h w -> (b f) c h w')

            decode_fake_x0 = 1. / 0.18215 * decode_fake_x0 
            decode_fake_mv = autoencoder.decode(decode_fake_x0) # B*4 3 256 256
            decode_fake_mv = rearrange(decode_fake_mv, '(b f) c h w -> b f c h w', b=batch)
            
            decode_fake_mv = decode_fake_mv.mul_(0.5).add_(0.5)
            decode_fake_mv.clamp_(0, 1) 
            vid_mean = torch.tensor([0.485, 0.456, 0.406], device=decode_fake_mv.device).view(1, 1, -1, 1, 1) #ncfhw
            vid_std = torch.tensor([0.229, 0.224, 0.225], device=decode_fake_mv.device).view(1, 1, -1, 1, 1) #ncfhw
            decode_fake_mv = decode_fake_mv.sub_(vid_mean).div_(vid_std) 

            gs_data['input'] = torch.cat([decode_fake_mv, gs_data['input'][:,idxs]], dim=2) # B 4 9 H W

            extra_idxs = np.random.permutation(np.arange(0, 24))[:4].tolist() 
            extra_idxs.extend(idxs)

            gs_data['masks_output'] = gs_data['masks_output'][:,extra_idxs]
            gs_data['images_output'] = gs_data['images_output'][:,extra_idxs]
            gs_data['cam_view'] = gs_data['cam_view'][:,extra_idxs]
            gs_data['cam_view_proj'] = gs_data['cam_view_proj'][:,extra_idxs]
            gs_data['cam_pos'] = gs_data['cam_pos'][:,extra_idxs]

            gs_out_data = self.lgm_big(gs_data)

            return gs_out_data
        else:
            if autoencoder is None:
                return x
            else:
                fake_x0 = _i(sqrt_recip_alphas_cumprod, t, xt) * xt - _i(sqrt_recipm1_alphas_cumprod, t, xt) * x
               
                idxs = [0, 6, 12, 18] 
                decode_fake_x0 = fake_x0[:,:,idxs] # B c 4 H W
                decode_fake_x0 = rearrange(decode_fake_x0, 'b c f h w -> (b f) c h w')
                
                decode_fake_x0 = 1. / 0.18215 * decode_fake_x0 
                decode_fake_mv = autoencoder.decode(decode_fake_x0) # B*4 3 256 256
                decode_fake_mv = rearrange(decode_fake_mv, '(b f) c h w -> b f c h w', b=batch)
             
                decode_fake_mv = decode_fake_mv.mul_(0.5).add_(0.5)
                decode_fake_mv.clamp_(0, 1) 
                vid_mean = torch.tensor([0.485, 0.456, 0.406], device=decode_fake_mv.device).view(1, 1, -1, 1, 1) #ncfhw
                vid_std = torch.tensor([0.229, 0.224, 0.225], device=decode_fake_mv.device).view(1, 1, -1, 1, 1) #ncfhw
                decode_fake_mv = decode_fake_mv.sub_(vid_mean).div_(vid_std) 

                gs_data['input'] = torch.cat([decode_fake_mv, gs_data['input'][:,idxs]], dim=2) # B 4 9 H W

                gs_out_data = self.lgm_big.infer(gs_data)

                infer_images = gs_out_data['images_pred']
                infer_images = rearrange(infer_images, 'b f c h w -> (b f) c h w')
                infer_images = F.interpolate(infer_images, (256, 256), mode='nearest')
                infer_images = infer_images.sub_(0.5).div_(0.5)

                latent_z = autoencoder.encode_firsr_stage(infer_images, 0.18215)
                latent_z = rearrange(latent_z, '(b f) c h w -> b c f h w', b=batch)

                return latent_z

    
    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
            name = "ResidualBlock"
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
            name = "ResBlock"
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
            name = "SpatialTransformer"
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "TemporalTransformer"
        elif isinstance(module, TemporalTransformer_attemask):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "TemporalTransformer_attemask"
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
            name = "CrossAttention"
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
            name = "MemoryEfficientCrossAttention"
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
            name = "BasicTransformerBlock"
        elif isinstance(module, FeedForward):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
            name = "FeedForward"
        elif isinstance(module, Upsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
            name = "Upsample"
        elif isinstance(module, Downsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
            name = "Downsample"
        elif isinstance(module, Resample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, reference)
            name = "Resample"
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "TemporalAttentionBlock"
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "TemporalAttentionMultiBlock"
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "InitTemporalConvBlock"
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            name = "TemporalConvBlock"
        elif isinstance(module, nn.ModuleList):
            name = []
            for block in module:
                x, name_ = self._forward_single(block,  x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference)
                name.append(name_)
        else:
            x = module(x)
            name = "Unknown"
        return x, name