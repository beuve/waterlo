import argparse

import torch
from tqdm import tqdm

from loader import loader_with_resize
from models import Generator, Bob
from utils import Models
from jpeg import add_jpeg_noise

import torchvision.transforms as T


def get_df_bin_prediction(prediction, annotations, size, label):
    w_i, h_i = size[0].item(), size[1].item()
    w_n, h_n = prediction.shape[-1], prediction.shape[-2]
    x_a, y_a, w_a, h_a = annotations
    x_a, y_a, w_a, h_a = x_a.item(), y_a.item(), w_a.item(), h_a.item()
    x_a, y_a, w_a, h_a = int(round(x_a + w_a / 2)), int(round(y_a + h_a / 2)), int(round(w_a / 2)), int(round(h_a / 2))
    x_p, y_p, w_p, h_p = int((w_n / w_i) * x_a), int((h_n / h_i) * y_a), int((w_n / w_i) * w_a), int((h_n / h_i) * h_a)

    # outer
    w_o, h_o = int(w_p * 1.3), int(w_p * 1.3)
    w_o = w_o if (x_p - w_o >= 0) else x_p
    w_o = w_o if (x_p + w_o < w_n) else (w_n - w_o)
    h_o = h_o if (y_p - h_o >= 0) else y_p
    h_o = h_o if (y_p - h_o < h_n) else (h_n - h_o)

    if label == "w":
        inner_box = prediction[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o].unsqueeze(0)
        inner = torch.sum(inner_box < 0.8).item() < inner_box.shape[-1] * inner_box.shape[-2] * 0.1

        mask = torch.ones_like(prediction)
        mask[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o] = 0
        outer = (torch.sum(prediction * mask) / torch.sum(mask)).item() > 0.7
    elif label == "d":
        inner_box = prediction[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o].unsqueeze(0)
        inner = torch.sum(inner_box < 0.8).item() > inner_box.shape[-1] * inner_box.shape[-2] * 0.1

        mask = torch.ones_like(prediction)
        mask[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o] = 0
        outer = (torch.sum(prediction * mask) / torch.sum(mask)).item() > 0.7
    elif label == "p":
        inner_box = prediction[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o].unsqueeze(0)
        inner = torch.sum(inner_box < 0.8).item() > inner_box.shape[-1] * inner_box.shape[-2] * 0.1

        mask = torch.ones_like(prediction)
        mask[0, y_p - h_o:y_p + h_o, x_p - w_o:x_p + w_o] = 0
        outer = (torch.sum(prediction * mask) / torch.sum(mask)).item() < 0.7
    else:
        assert (False)

    return inner and outer


def detect_watermarks(models, imgs, annotations, size, device, quality, label):
    imgs = imgs.to(device)

    if quality <= 100:
        imgs = add_jpeg_noise(imgs, device, quality)

    bob_preds = models.B(imgs)

    res = get_df_bin_prediction(bob_preds[0], annotations, [size[0][0], size[1][0]], label)
    return res


def detect(models, data_loader, device, label):

    batch_nb = len(data_loader)
    accuracy = 0
    # Compression above 100 correspond to no compression
    for compression in range(0, 111, 10):
        print('### COMPRESSION :', compression)
        for data in tqdm(data_loader, total=batch_nb):
            imgs_id, imgs, size, annotations = data
            models.eval()
            with torch.no_grad():
                accuracy += detect_watermarks(models, imgs, annotations, size, device, compression, label)

        print(f"ACCURACY: {accuracy / batch_nb}")
        accuracy = 0


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
        annotations=True,
    )

    detect(models, test_loader, device, args.label)


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
        "-l",
        "--label",
        default="w",
        type=str,
        help="Weither the image is a watermark (w), pristine (p), or deepfake (d)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="/weights",
        type=str,
        help="Repository containing the pretrained weights",
    )
    args = parser.parse_args()
    main(args)