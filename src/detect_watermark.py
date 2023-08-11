import argparse
import os

import torch
from tqdm import tqdm

from loader import loader_with_resize
from models import Generator, Bob
from utils import Models, imsave, rgb_to_ycbcr
from jpeg import add_jpeg_noise

import torchvision.transforms as T


def detect_watermarks(models, imgs, device, output, imgs_ids, quality):
    imgs = imgs.to(device)

    if quality <= 100:
        imgs = add_jpeg_noise(imgs, device, quality)

    bob_preds = models.B(imgs)
    pred = T.Resize(imgs[0].shape[2:], interpolation=T.InterpolationMode.NEAREST)(bob_preds)
    img_map = rgb_to_ycbcr(imgs[0].clone())
    img_map[1] = pred
    img_map[2] = 1 - pred
    imsave(os.path.join(output, f'{imgs_ids[0]}.png'), img_map)


def detect(models, data_loader, output_folder, device):
    try:
        os.mkdir(output_folder)
    except OSError as error:
        print("Warning", error)

    batch_nb = len(data_loader)
    # Compression above 100 correspond to no compression
    for data in tqdm(data_loader, total=batch_nb):
        imgs_id, imgs, _ = data
        models.eval()
        with torch.no_grad():
            detect_watermarks(models, imgs, device, output_folder, imgs_id, 101)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    generator = Generator(in_channels=3, out_channels=3).to(device)
    bob = Bob(in_channels=3, out_channels=1).to(device)
    models = Models(generator, bob)
    models.load(args.weights)

    # Data loader
    test_loader = loader_with_resize(
        args.dataset,
        512,
        1,
        "",
        drop_last=False,
    )

    detect(models, test_loader, args.output, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="/data",
        type=str,
        help="Repository containing the dataset",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="/weights",
        type=str,
        help="Repository containing the pretrained weights",
    )
    parser.add_argument(
        "-c",
        "--compression",
        default=101,
        type=int,
        help="Compression quality (above 100 is without compression)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/output",
        type=str,
        help="Repository containing the result watermarked images",
    )
    args = parser.parse_args()
    main(args)