import importlib
import torchvision
import torch
from torch import optim
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
import cv2
import PIL

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def prepare_inputs(image_path, elevation_input, crop_size=-1, image_size=256):
    image_input = Image.open(image_path)

    if crop_size!=-1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
        image_input = add_margin(ref_img_, size=image_size)
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((image_size, image_size), resample=Image.BICUBIC)

    image_input = np.asarray(image_input)
    image_input = image_input.astype(np.float32) / 255.0
    if image_input.shape[-1]==4:
        ref_mask = image_input[:, :, 3:]
        image_input[:, :, :3] = image_input[:, :, :3] * ref_mask + 1 - ref_mask  # white background
    return image_input

root_dir = sys.argv[1]
items = [os.path.join(root_dir, item) for item in os.listdir(root_dir)]
for idx, item in enumerate(items):
    res = prepare_inputs(item, 15, 200)
    Image.fromarray((res*255.0).astype(np.uint8)).save("./data/images", "{:05d}.png".format(idx))
