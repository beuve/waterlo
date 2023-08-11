import argparse
import os

import torch
from tqdm import tqdm

from loader import loader_with_padding
from models import Generator, Bob
from utils import Models, imsave, crop_padding


def save_watermarks(models, data, device, output, alpha):
    im_nb, imgs, size, annotations = data
    imgs = imgs.to(device)

    batch_size = imgs.shape[0]
    w = models.G(imgs)
    print(alpha)
    watermarks = imgs + alpha * w
    watermarks[watermarks < 0] = 0
    watermarks[watermarks > 1] = 1

    croped_images, _ = crop_padding(imgs, size)
    croped_watermarks, _ = crop_padding(watermarks, size)
    f = open(os.path.join(output, '..', 'originals', "facial_landmarks.txt"), "a")
    torch.cuda.empty_cache()
    for i in range(batch_size):
        w, h = size[0][i], size[1][i]
        w_n, h_n = croped_watermarks[i].shape[-1], croped_watermarks[i].shape[-2]
        x_a, y_a, w_a, h_a = annotations
        if w > h and w > w_n:
            w_a = int((w_n / w) * w_a)
            x_a = int((w_n / w) * x_a)
            h_a = int((w_n / w) * h_a)
            y_a = int((w_n / w) * y_a)
        elif h > w and h > h_n:
            h_a = int((h_n / h) * h_a)
            y_a = int((h_n / h) * y_a)
            w_a = int((h_n / h) * w_a)
            x_a = int((h_n / h) * x_a)
        s = im_nb[i].split('_')
        f.write(f"{s[0]}\t{s[1]}\t{x_a}\t{y_a}\t{w_a}\t{h_a}\n")

        imsave(os.path.join(output, f'{im_nb[i]}.png'), croped_watermarks[i])
        imsave(os.path.join(output, '..', 'originals', f'{im_nb[i]}.png'), croped_images[i])
    f.close()


def apply(models, data_loader, output_folder, device, alpha):
    try:
        os.mkdir(output_folder)
    except OSError as error:
        print("Warning", error)

    batch_nb = len(data_loader)

    for data in tqdm(data_loader, total=batch_nb):

        models.eval()
        with torch.no_grad():
            save_watermarks(models, data, device, output_folder, alpha)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    generator = Generator(in_channels=3, out_channels=3).to(device)
    bob = Bob(in_channels=3, out_channels=1).to(device)
    models = Models(generator, bob)
    models.load(args.weights)

    # Data loader
    test_loader = loader_with_padding(args.dataset, 512, 1, "", annotations=True, drop_last=False)

    # Train
    apply(models, test_loader, args.output, device, args.alpha)


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
        help="Repository containing the trained weights",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        default="0.002",
        type=float,
        help="Value of alpha",
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