import argparse

import torch
import torch.nn as nn
from os.path import isfile, join
from os import listdir
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as T


def list_videos(folder):
    return [(f[:-4], join(folder, f)) for f in listdir(folder) if isfile(join(folder, f)) and not f.endswith(".txt")]


def dico_videos(folder):
    return {f[:-4]: join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and not f.endswith(".txt")}


def pair_images(original_folder, watermark_folder):
    watermark_files = dico_videos(watermark_folder)
    original_files = list_videos(original_folder)

    print(len(watermark_files), len(original_files))
    res = []
    for original in original_files:
        if original[0] in watermark_files.keys():
            watermark = watermark_files[original[0]]
            watermark_infos = {
                "id": original[0],
                "path": watermark,
            }
            original_infos = {
                "id": original[0],
                "path": original[1],
            }
            res.append((watermark_infos, original_infos))
    return res


def f_psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def main(args):

    paired_images = pair_images(args.originals, args.watermarked)
    psnr = 0
    N = len(paired_images)

    for (watermarked_infos, original_infos) in tqdm(paired_images):
        ori = original_infos['path']
        wat = watermarked_infos['path']
        img_ori = T.ToTensor()(Image.open(ori).convert("RGB")).to('cuda')
        img_wat = T.ToTensor()(Image.open(wat).convert("RGB")).to('cuda')
        psnr += f_psnr(img_ori, img_wat).item()
    print(psnr / N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--originals",
        default="/data",
        type=str,
        help="Repository containing the original images",
    )
    parser.add_argument(
        "--watermarked",
        default="/data",
        type=str,
        help="Repository containing the watermarked images",
    )
    parser.add_argument(
        "-m",
        "--method",
        default="faceshifter",
        type=str,
        help="deepfake model used",
    )
    args = parser.parse_args()
    main(args)