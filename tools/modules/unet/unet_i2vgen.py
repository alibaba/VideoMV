import math
import torch
import xformers
import xformers.ops
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper
from safetensors.torch import load_file
from .util import *
# from .mha_flash import FlashAttentionBlock
from utils.registry_class import MODEL

import numpy as np
USE_TEMPORAL_TRANSFORMER = True

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


@MODEL.register_class()
class UNetSD_I2VGen(nn.Module):
    def __init__(self,
            config=None,
            in_dim=7,
            dim=512,
            y_dim=512,
            context_dim=512,
            hist_dim = 156,
            concat_dim = 8,
            dim_condition=4,
            out_dim=6,
            num_tokens=4,
            dim_mult=[1, 2, 3, 4],
            num_heads=None,
            head_dim=64,
            num_res_blocks=3,
            attn_scales=[1 / 2, 1 / 4, 1 / 8],
            use_scale_shift_norm=True,
            dropout=0.1,
            temporal_attn_times=1,
            camera_dim=16,
            temporal_attention = True,
            use_checkpoint=False,
            use_image_dataset=False,
            use_sim_mask = False,
            use_camera_condition=False,
            use_lgm_refine=False,
            training=True,
            inpainting=True,
            p_all_zero=0.1,
            p_all_keep=0.1,
            zero_y = None,
            adapter_transformer_layers = 1,
            **kwargs):
        super(UNetSD_I2VGen, self).__init__()
        
        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        self.zero_y = zero_y
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
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
        concat_dim = self.in_dim

        self.use_camera_condition = use_camera_condition
        self.camera_dim = camera_dim
        self.use_lgm_refine = use_lgm_refine

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        if self.use_lgm_refine:
            from core.options import config_defaults
            from core.models import LGM
            lgm_opt = config_defaults['big'] 
            self.lgm_big = LGM(lgm_opt)

        # Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), # [320,1280]
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        self.context_embedding = nn.Sequential(
            nn.Linear(y_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, context_dim * self.num_tokens))

        if self.use_camera_condition:
            self.camera_embedding = nn.Sequential(
                nn.Linear(self.camera_dim, embed_dim), # [320,1280]
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.camera_embedding[-1].weight)
            nn.init.zeros_(self.camera_embedding[-1].bias)
        
        self.fps_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        nn.init.zeros_(self.fps_embedding[-1].weight)
        nn.init.zeros_(self.fps_embedding[-1].bias)
        
        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(heads = num_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # [Local Image embeding]
        self.local_image_concat = nn.Sequential(
            nn.Conv2d(4, concat_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))

        self.local_temporal_encoder = TransformerV2(
                heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, 
                dropout_atte = 0.05, mlp_dim=concat_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)

        self.local_image_embedding = nn.Sequential(
            nn.Conv2d(4, concat_dim * 8, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(concat_dim * 8, concat_dim * 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 16, 1024, 3, stride=2, padding=1))

        # encoder
        self.input_blocks = nn.ModuleList()
        # init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim + concat_dim, dim, 3, padding=1)])

        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformer(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))
        # elif temporal_conv:
        # init_block.append(InitTemporalConvBlock(dim,dropout=dropout,use_image_dataset=use_image_dataset))
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,)])
                if scale in attn_scales:
                    # block.append(FlashAttentionBlock(out_dim, context_dim, num_heads, head_dim))
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
                    # block = nn.ModuleList([ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 'downsample')])
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    # block.append(TemporalConvBlock(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
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

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        
        # zero out the last layer params
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
        image = None,
        local_image = None,
        camera_data=None,
        masked = None,
        fps = None,
        video_mask = None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0,  # mask last frame num
        **kwargs):
        
        assert self.inpainting or masked is None, 'inpainting is not supported'
        xt = x.clone().detach()
        batch, c, f, h, w= x.shape
        device = x.device
        self.batch = batch
        if local_image.ndim == 5 and local_image.size(2) > 1:
            local_image = local_image[:, :, :1, ...] 
        elif local_image.ndim != 5:
            local_image = local_image.unsqueeze(2)

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

        # [Concat]
        concat = x.new_zeros(batch, self.concat_dim, f, h, w)
        if f > 1:
            mask_pos = torch.cat([(torch.ones(local_image[:,:,:1].size())*( (tpos+1)/(f-1) )).cuda() for tpos in range(f-1)], dim=2)
            _ximg = torch.cat([local_image[:,:,:1], mask_pos], dim=2) 
            _ximg = rearrange(_ximg, 'b c f h w -> (b f) c h w')
        else:
            _ximg = rearrange(local_image, 'b c f h w -> (b f) c h w')

        _ximg = self.local_image_concat(_ximg)
        _h = _ximg.shape[2]
        _ximg = rearrange(_ximg, '(b f) c h w -> (b h w) f c', b = batch)
        _ximg = self.local_temporal_encoder(_ximg)
        _ximg = rearrange(_ximg, '(b h w) f c -> b c f h w', b = batch, h = _h)
        concat += _ximg
        concat += _ximg  # TODO: This is a bug, but it doesn't matter.
        
        # [Embeddings]
        embeddings = self.time_embed(sinusoidal_embedding(t, self.dim)) + self.fps_embedding(sinusoidal_embedding(fps, self.dim))
        embeddings = embeddings.repeat_interleave(repeats=f, dim=0)

        # [Camera Embeddings]
        if self.use_camera_condition and camera_data is not None:
            # print("add camera conditions.")
            camera_emb = rearrange(camera_data, 'b f c -> (b f) c')
            camera_emb = self.camera_embedding(camera_emb)
            # camera_data = rearrange(camera_data, '(b f) c -> b f c', b=batch)  # no need to turn it back.
            embeddings = embeddings + camera_emb

        # [Context]
        # [C] for text input
        context = x.new_zeros(batch, 0, self.context_dim)
        if y is not None:
            y_context = y
            context = torch.cat([context, y_context], dim=1)
        else:
            y_context = self.zero_y.repeat(batch, 1, 1)[:, :1, :]
            context = torch.cat([context, y_context], dim=1)

        # [C] for local input
        local_context = rearrange(local_image, 'b c f h w -> (b f) c h w')
        local_context = self.local_image_embedding(local_context)
        h = local_context.shape[2]
        local_context = rearrange(local_context, 'b c h w -> b (h w) c', b = batch, h = h) # [12, 64, 1024]
        context = torch.cat([context, local_context], dim=1)

        # [C] for global input
        if image is not None:
            image_context = self.context_embedding(image)
            image_context = image_context.view(-1, self.num_tokens, self.context_dim)
            context = torch.cat([context, image_context], dim=1)
        context = context.repeat_interleave(repeats=f, dim=0)

        x = torch.cat([x, concat], dim=1)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)
        
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias,focus_present_mask, video_mask)
        
        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias,focus_present_mask, video_mask, reference=xs[-1] if len(xs) > 0 else None)
        
        # head
        x = self.out(x) # [32, 4, 32, 32]
        
        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)

        if self.use_lgm_refine and x0 is not None:
            fake_x0 = _i(sqrt_alphas_cumprod, t, xt) * xt - \
                _i(sqrt_one_minus_alphas_cumprod, t, xt) * x
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
                fake_x0 = _i(sqrt_alphas_cumprod, t, xt) * xt - \
                    _i(sqrt_one_minus_alphas_cumprod, t, xt) * x
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

                gs_out_data = self.lgm_big.infer(gs_data, bg_color_factor=0.7)

                infer_images = gs_out_data['images_pred']
                infer_images = rearrange(infer_images, 'b f c h w -> (b f) c h w')
                # interpolate back to (256, 256)
                infer_images = F.interpolate(infer_images, (256, 256), mode='nearest')
                infer_images = infer_images.sub_(0.5).div_(0.5)
                # encode the infer_images again, use latent_z to substitute fake_x0
                latent_z = autoencoder.encode_firsr_stage(infer_images, 0.18215)
                latent_z = rearrange(latent_z, '(b f) c h w -> b c f h w', b=batch)
                return latent_z



    
    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalTransformer_attemask):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, Upsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Downsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Resample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block,  x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference)
        else:
            x = module(x)
        return x
