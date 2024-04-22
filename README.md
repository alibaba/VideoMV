## VideoMV: Consistent Multi-View Generation Based on Large Video Generative Model.

[Qi Zuo\*](https://scholar.google.com/citations?view_op=list_works&hl=en&user=UDnHe2IAAAAJ),
[Xiaodong Gu\*](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao),
[Lingteng Qiu](https://lingtengqiu.github.io/),
[Yuan Dong](dy283090@alibaba-inc.com),
[Zhengyi Zhao](bushe.zzy@alibaba-inc.com),
[Weihao Yuan](https://weihao-yuan.com/),
[Rui Peng](https://prstrive.github.io/),
[Siyu Zhu](https://sites.google.com/site/zhusiyucs/home/),
[Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=zh-CN&oi=ao),
[Liefeng Bo](https://research.cs.washington.edu/istc/lfb/),
[Qixing Huang](https://www.cs.utexas.edu/~huangqx/)

https://github.com/alibaba/VideoMV/assets/58206232/3a78e28d-bda4-4d4c-a2ae-994d0320a301

## [Project page](https://aigc3d.github.io/VideoMV) | [Paper](https://arxiv.org/abs/2311.16918) | [YouTube](https://www.youtube.com/watch?v=zxjX5p0p0Ks) | [3D Rendering Dataset](https://aigc3d.github.io/gobjaverse)

## TODO  :triangular_flag_on_post:
- [ ]  Release GS、Neus、NeRF reconstruction code.
- [x]  News: Release text-to-mv (G-Objaverse + Laion) training code and pretrained model(2024.04.22). Check the Inference&&Training Guidelines.
- [x]  Release the training code.
- [x]  Release multi-view inference code and pretrained weight(G-Objaverse).

## Architecture

![architecture](assets/f.png)

## Install

- System requirement: Ubuntu20.04
- Tested GPUs: A100

Install requirements using following scripts.

```bash
git clone https://github.com/alibaba/VideoMV.git
conda create -n VideoMV python=3.8
conda activate VideoMV
cd VideoMV && bash install.sh
```

## Inference

```bash
# Download our pretrained models
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/pretrained_models.zip
unzip pretrained_models.zip
# text-to-mv sampling
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg ./configs/t2v_infer.yaml
# text-to-mv sampling using pretrained model trained on laion+Gobjaverse
wget oss://virutalbuy-public/share/aigc3d/videomv_laion/non_ema_00365000.pth
# modify the [test_model] as the location of [non_ema_00365000.pth]
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg ./configs/t2v_infer.yaml


# image-to-mv sampling
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg ./configs/i2vgen_xl_infer.yaml

# To test raw prompts: type the prompts in ./data/test_prompts.txt

# To test raw images: use Background-Remover(https://www.remove.bg/) to get the foreground of images
# place the images all in /path/to/your_dir
# Then run
python -m utils.recenter_i2v /path/to/your_dir
# The recenter results will be saved in ./data/images
# add test image paths in ./data/test_images.txt
# Then run
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg ./configs/i2vgen_xl_infer.yaml
```

## Training

```bash
# Download our dataset(G-Objaverse) following the instructions at 
# https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse
# Modify the vid_dataset.data_dir_list as your download data_root 
# in ./configs/t2v_train.yaml and ./configs/i2vgen_xl_train.yaml

# Text-to-mv finetuning
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./configs/t2v_train.yaml
# Text-to-mv fintuning using both Laion and Gobjaverse. 
# (Note we use 24 A100 for training both datasets. If your computation resource is not sufficient, do not try it!)
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./configs/t2v_train_laion.yaml

# Text-to-mv Feed-forward reconstruction finetuning.
# Modify the UNet.use_lgm_refine as 'True' in ./configs/t2v_train.yaml. Then
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./configs/t2v_train.yaml


# Image-to-mv finetuning
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./configs/i2vgen_xl_train.yaml
# Image-to-mv Feed-forward reconstruction finetuning.
# Modify the UNet.use_lgm_refine as 'True' in ./configs/i2vgen_xl_train.yaml. Then
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./configs/i2vgen_xl_train.yaml
```

## Tips

- You will observe a sudden convergence in Text-to-MV finetuning(~5min).

- You will not observe a sudden convergence in Image-to-MV finetuning. Usually it takes half a day for a initial convergence.

- Remove the background of test image use [Background-Remover](https://www.remove.bg/) instead of rembg to get a better result. The artifacts of segmentation mask will influence the quality of multi-view generation results.

## Future Works

- Dense View Large Reconstruction Model.

- More general and high-quality Text-to-MV using better Video Diffusion Model(like HiGen) and novel finetuning techniques.

## Acknowledgement

This work is built on many amazing research works and open-source projects:

- [VGen](https://github.com/ali-vilab/VGen)
- [LGM](https://github.com/3DTopia/LGM)
- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [GaussianSplatting](https://github.com/graphdeco-inria/gaussian-splatting)

Thanks for their excellent work and great contribution to 3D generation area.

We would like to express our special gratitude to [Jiaxiang Tang](https://github.com/ashawkey), [Yuan Liu](https://github.com/liuyuan-pal) for the valuable discussion in LGM and SyncDreamer.


## Citation	

```
@misc{zuo2024videomv,
      title={VideoMV: Consistent Multi-View Generation Based on Large Video Generative Model}, 
      author={Qi Zuo and Xiaodong Gu and Lingteng Qiu and Yuan Dong and Zhengyi Zhao and Weihao Yuan and Rui Peng and Siyu Zhu and Zilong Dong and Liefeng Bo and Qixing Huang},
      year={2024},
      eprint={2403.12010},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
