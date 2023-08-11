import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
from random import randint, random

from dataclasses import dataclass
import os


@dataclass
class Models:
    G: nn.Module
    B: nn.Module

    def train(self):
        self.G.train()
        self.B.train()

    def eval(self):
        self.G.eval()
        self.B.eval()

    def save(self, path=""):
        torch.save(self.G.state_dict(), os.path.join(path, "G.pt"))
        torch.save(self.B.state_dict(), os.path.join(path, "B.pt"))

    def load(self, path=""):
        self.G.load_state_dict(torch.load(os.path.join(path, "G.pt")))
        self.B.load_state_dict(torch.load(os.path.join(path, "B.pt")))


@dataclass
class Losses:
    G: nn.Module
    B: nn.Module


@dataclass
class Optimizers:
    G: nn.Module
    B: nn.Module

    def zero_grad(self):
        self.G.zero_grad()
        self.B.zero_grad()

    def step(self):
        self.G.step()
        self.B.step()


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :] * 255
    g: torch.Tensor = image[..., 1, :, :] * 255
    b: torch.Tensor = image[..., 2, :, :] * 255

    y = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    y[y > 1] = 1
    y[y < 0] = 0
    cb = (-0.168736 * r - 0.331264 * g + 0.5 * b) / 255
    cb[cb > 1] = 1
    cb[cb < -1] = 0
    cr = (0.5 * r - 0.418688 * g - 0.081312 * b) / 255
    cr[cr > 1] = 1
    cr[cr < -1] = 0
    ycbcr = torch.stack([y, cb, cr], -3)

    return ycbcr


def crop_padding(image, size):
    im_h, im_w = image.shape[-1], image.shape[-2]
    res = torch.zeros_like(image)
    res_o = []
    for i in range(image.shape[0]):
        w, h = size[0][i], size[1][i]
        if w > h and w > im_w:
            h = int(im_w * h / w)
            w = im_w
        elif h > w and h > im_h:
            w = int(im_h * w / h)
            h = im_h
        y = (im_h - int(h)) // 2
        x = (im_w - int(w)) // 2
        im = F.crop(image[i], y, x, h, w)
        res_o.append(im)
        res[i] = T.Resize((im_h, im_w))(im)
    return res_o, res


def mask_image(image, data, min_mask_size=8):
    outer_masked_image = image.clone()
    inner_masked_image = data.clone()
    mask = torch.ones((image.shape[0], image.shape[2], image.shape[3])).to(image.device)
    height, width = mask.shape[1], mask.shape[2]
    for i in range(image.shape[0]):
        w_1, h_1 = randint(0, width - min_mask_size - 1), randint(0, height - min_mask_size - 1)
        w_2, h_2 = randint(w_1, width - min_mask_size - 1) + min_mask_size, randint(h_1, height - min_mask_size - 1) + min_mask_size
        mask[i, h_1:h_2, w_1:w_2] = 0
        blend = (50 + 40 * random()) / 100
        outer_masked_image[i, :, h_1:h_2, w_1:w_2] = blend * outer_masked_image[i, :, h_1:h_2, w_1:w_2] + (1 - blend) * data[i, :, h_1:h_2, w_1:w_2]
        inner_masked_image[i, :, h_1:h_2, w_1:w_2] = blend * inner_masked_image[i, :, h_1:h_2, w_1:w_2] + (1 - blend) * image[i, :, h_1:h_2, w_1:w_2]
    mask = T.Resize((64, 64))(mask)
    return outer_masked_image, mask, inner_masked_image, 1 - mask


def imsave(path, image):
    image = image.squeeze().clone()
    if len(image.shape) == 3:
        image[image < 0] = 0
        image[image > 1] = 1
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        plt.imsave(path, image)
    else:
        image = image.detach().cpu().numpy()
        plt.imsave(path, image, vmin=0, vmax=1, cmap="inferno")
