import os
import math
import cv2
import torch
import random
import logging
import tempfile
import numpy as np
from functools import partial
from copy import copy
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import albumentations
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import webdataset as wds

try:
    from utils.registry_class import DATASETS
except Exception as ex:
    print("#" * 20)
    print("import error, try fixed by appending path")
    import sys
    sys.path.append("./")
    from utils.registry_class import DATASETS



def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def my_decoder(key, value):
    # solve the issue: https://github.com/webdataset/webdataset/issues/206

    if key.endswith('.jpg'):
        # return Image.open(BytesIO(value))
        return np.asarray(Image.open(BytesIO(value)).convert('RGB'))

    return None


class filter_fake:

    def __init__(self, punsafety=0.2, aest=4.5):
        self.punsafety = punsafety
        self.aest = aest

    def __call__(self, src):
        for sample in src:
            img, prompt, json = sample
            # watermark filter
            if json['pwatermark'] is not None:
                if json['pwatermark'] > 0.3:
                    continue

            # watermark
            if json['punsafe'] is not None:
                if json['punsafe'] > self.punsafety:
                    continue

            # watermark
            if json['AESTHETIC_SCORE'] is not None:
                if json['AESTHETIC_SCORE'] < self.aest:
                    continue

            # ratio filter
            w, h = json['width'], json['height']
            if max(w / h, h / w) > 3:
                continue

            yield img, prompt, json['AESTHETIC_SCORE'], json['key']


class Laion2b_Process(object):
    
    def __init__(self,
                 size=None,
                 degradation=None,
                 downscale_f=4,
                 min_crop_f=0.8,
                 max_crop_f=1.,
                 random_crop=True,
                 debug: bool = False):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        # downsacle_f = 0

        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA)


    def __call__(self, samples):
        example = {}
        image, caption, aesthetics, key = samples

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(
            self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(
                height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(
                height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)['image']
        image = self.image_rescaler(image=image)['image']

        # -1, 1
        ref_image = (image / 127.5 - 1.0).astype(np.float32)
        ref_image = ref_image.transpose(2, 0, 1)
        vit_image = ref_image
        video_data = ref_image[np.newaxis, :, :, :]
        
        
        # example['image'] = image
        # # depth prior is set to 384
        # example['prior'] = resize_image(HWC3(image), 384)
        # example['caption'] = caption
        # example['aesthetics'] = aesthetics
        # example['key'] = key

        return ref_image, vit_image, video_data, caption, key


@DATASETS.register_class()
class LAIONImageDataset():
    def __init__(self, 
            data_list, 
            data_dir_list,
            max_words=1000,
            vit_resolution=[224, 224],
            resolution=(256, 256),
            max_frames=1,
            transforms=None,
            vit_transforms=None,
            **kwargs):
        
        aest = kwargs.get("aest", 4.0)
        punsafety = kwargs.get("punsafety", 0.2)
        min_crop_f = kwargs.get("min_crop_f", 1.0)
        self.num_samples = kwargs.get("num_samples", 60580*2000)
        
        assert resolution[0] == resolution[1]
        assert len(data_dir_list) == 1
        assert len(data_list) == 1
        
        self.web_dataset = wds.WebDataset(os.path.join(data_dir_list[0], data_list[0]), resampled=True).decode(
                        my_decoder, 'rgb8').shuffle(1000).to_tuple(
                            'jpg', 'txt', 'json').compose(
                                filter_fake(aest=aest, punsafety=punsafety)).map(
                                    Laion2b_Process(
                                        size=resolution[0],
                                        min_crop_f=min_crop_f)
                                )

    def create_dataloader(self, batch_size, world_size, workers):
        num_samples = self.num_samples
        self.dataset = self.web_dataset.batched(batch_size, partial=False)
        round_fn = math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = self.dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

        self.dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
    
        self.dataloader.num_batches = num_batches
        self.dataloader.num_samples = num_samples
        
        print("#"*50)
        print(f"dataloder, num_batches:{num_batches}, num_samples:{num_samples}")
        print("#"*50)
        return self.dataloader
        
        

if __name__ == "__main__":
    dataset = LAIONImageDataset(
            data_list=['{00000..00001}.tar'],
            data_dir_list=['/home/gxd/projects/Normal-Depth-Diffusion-Model/tools/download_dataset/laion-2ben-5_aes/'],
            max_words=1000,
            resolution=(256, 256),
            vit_resolution=(224, 224),
            max_frames=24,
            sample_fps=1,
            transforms=None,
            vit_transforms=None,
            get_first_frame=True,
            num_samples=1000,
            debug=True)
    
    batch_size = 20
    world_size = 1
    workers = 10

    dataloader = dataset.create_dataloader(batch_size, world_size, workers)

    import tqdm
    key_list = []
    for data in tqdm.tqdm(dataloader):
        pass
        print(data[0].shape, data[1].shape, data[2].shape)
        key_list.extend(data[4])
    print(len(key_list), len(set(key_list)))
        

