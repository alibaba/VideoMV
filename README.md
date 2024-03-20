## VideoMV: Consistent Multi-View Generation Based on Large Video Generative Model.

[Qi Zuo\*](https://scholar.google.com/citations?view_op=list_works&hl=en&user=UDnHe2IAAAAJ),
[Xiaodong Gu\*](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao),
[Lingteng Qiu](https://lingtengqiu.github.io/),
[Yuan Dong](https://mutianxu.github.io/),
Zhengyi Zhao,
[Weihao Yuan](https://weihao-yuan.com/),
[Rui Peng](https://prstrive.github.io/),
[Siyu Zhu](https://sites.google.com/site/zhusiyucs/home/),
[Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=zh-CN&oi=ao),
[Liefeng Bo](https://research.cs.washington.edu/istc/lfb/),
[Qixing Huang](https://www.cs.utexas.edu/~huangqx/)

## [Project page](https://aigc3d.github.io/VideoMV) | [Paper](https://arxiv.org/abs/2311.16918) | [YouTube](https://www.youtube.com/watch?v=zxjX5p0p0Ks) | [3D Rendering Dataset](https://aigc3d.github.io/gobjaverse)

## TODO  :triangular_flag_on_post:
- [ ]  Release text-to-mv (G-Objaverse + Laion) training code and pretrained model.
- [ ]  Release GS、Neus、NeRF reconstruction code.
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

## Acknowledgement

This work is built on many amazing research works and open-source projects:

- [VGen](https://github.com/ali-vilab/VGen)
- [LGM](https://github.com/3DTopia/LGM)
- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [GaussianSplatting](https://github.com/graphdeco-inria/gaussian-splatting)

Thanks for their excellent work and great contribution to 3D generation area.

We would like to express our special gratitude to [Jiaxiang Tan](https://github.com/ashawkey), [Yuan Liu](https://github.com/liuyuan-pal) for the valuable discussion in LGM and SyncDreamer.