from enum import Enum
import math
import argparse

import torch
from tqdm import tqdm
import os

from random import randint

from loader import loader_with_padding
from models import Generator, Bob
from loss import GeneratorLoss, BobLoss
from utils import Models, Losses, Optimizers
from random import seed

from jpeg import add_jpeg_noise
from utils import crop_padding, mask_image, imsave


class LEARNING_STEP(Enum):
    VALIDATION = 1
    TRAINING = 0


def get_predictions(models, data, criterions, device, alpha, save_image=False, output='outputs', compression=False):
    _, imgs, size = data
    imgs = imgs.to(device)

    watermarks_noise = models.G(imgs)
    watermarks = imgs + alpha * watermarks_noise
    watermarks[watermarks < 0] = 0
    watermarks[watermarks > 1] = 1

    croped_watermarks, resized_watermarks = crop_padding(watermarks, size)
    croped_imgs, resized_imgs = crop_padding(imgs, size)
    outer_masked_watermark, outer_masks, inner_masked_watermark, inner_masks = mask_image(resized_watermarks, resized_imgs)

    if (randint(1, 5) != 1 and compression):
        quality = 100 - randint(10, 40)
        compressed_watermarks = add_jpeg_noise(resized_watermarks, device, quality)
        compressed_imgs = add_jpeg_noise(resized_imgs, device, quality)
        compressed_outer = add_jpeg_noise(outer_masked_watermark, device, quality)
        compressed_inner = add_jpeg_noise(inner_masked_watermark, device, quality)
    else:
        compressed_watermarks = watermarks
        compressed_imgs = imgs
        compressed_outer = outer_masked_watermark
        compressed_inner = inner_masked_watermark

    bob_watermark_preds = models.B(compressed_watermarks)
    bob_outer_masked_watermark_preds = models.B(compressed_outer)
    bob_inner_masked_watermark_preds = models.B(compressed_inner)
    bob_original_preds = models.B(compressed_imgs)

    if save_image:
        imsave(os.path.join(output, "original.png"), croped_imgs[0])
        imsave(os.path.join(output, "watermark.png"), croped_watermarks[0])
        imsave(os.path.join(output, "inner_mask_gt.png"), inner_masks[0])
        imsave(os.path.join(output, "outer_mask_gt.png"), outer_masks[0])
        imsave(os.path.join(output, "outer_mask_bob.png"), bob_outer_masked_watermark_preds[0])
        imsave(os.path.join(output, "inner_mask_bob.png"), bob_inner_masked_watermark_preds[0])
        imsave(os.path.join(output, "bob_real.png"), bob_original_preds[0])
        imsave(os.path.join(output, "bob_fake.png"), bob_watermark_preds[0])
        croped_watermark_noise, _ = crop_padding(watermarks_noise, size)
        maxi = torch.max(croped_watermark_noise[0])
        mini = torch.min(croped_watermark_noise[0])
        imsave(os.path.join(output, "watermark_noise.png"), (maxi - croped_watermark_noise[0]) / (maxi - mini))

    generator_loss = criterions.G(watermarks_noise, imgs)
    ones, zeros = torch.ones_like(inner_masks), torch.zeros_like(inner_masks)
    bob_loss = criterions.B(
        torch.cat((bob_watermark_preds, bob_original_preds, bob_inner_masked_watermark_preds, bob_outer_masked_watermark_preds), dim=0).squeeze(),
        torch.cat((ones, zeros, inner_masks, outer_masks), dim=0),
    )
    return generator_loss, bob_loss


def train_one_epoch(models, data, criterions, optimizers, device, alpha, save_image=False, output="outputs", compression=False, lambd=100):

    # Train generator
    optimizers.G.zero_grad()
    optimizers.B.zero_grad()

    models.G.train()
    models.B.train()

    generator_loss, bob_loss = get_predictions(models, data, criterions, device, alpha, save_image=save_image, output=output, compression=compression)
    (bob_loss + lambd * generator_loss).backward()

    optimizers.B.step()
    optimizers.G.step()

    return generator_loss.item(), bob_loss.item()


def valid_one_epoch(models, data, criterions, device, alpha, save_image=False, output="outputs", compression=False):
    models.eval()
    with torch.no_grad():
        generator_loss, bob_loss = get_predictions(models, data, criterions, device, alpha, save_image=save_image, output=output, compression=compression)
        return generator_loss.item(), bob_loss.item()


def one_epoch(models, data_loader, criterions, optimizers, device, step, output, compression, lambd, alpha):

    epoch_generator_loss = 0.0
    epoch_bob_loss = 0.0

    batch_nb = len(data_loader)
    save_image = True
    for i, data in tqdm(enumerate(data_loader), total=batch_nb):
        torch.cuda.empty_cache()
        if i % 100 == 0:
            save_image = True

        if step == LEARNING_STEP.VALIDATION:
            generator_loss, bob_loss = valid_one_epoch(models, data, criterions, device, alpha, save_image, output, compression=compression)

        else:
            generator_loss, bob_loss = train_one_epoch(models, data, criterions, optimizers, device, alpha, save_image=save_image, output=output, compression=compression, lambd=lambd)

        save_image = False

        epoch_generator_loss += generator_loss
        epoch_bob_loss += bob_loss

    epoch_generator_loss /= batch_nb
    epoch_bob_loss /= batch_nb

    return epoch_generator_loss, epoch_bob_loss


def fit(models, epochs, criterions, optimizers, device, train_loader, valid_loader, output, compression, lambd, alpha) -> None:

    best_loss = math.inf

    for epoch in range(1, epochs + 1):

        print("=" * 20)
        print(f"EPOCH {epoch} TRAINING...")

        train_generator_loss, train_bob_loss = one_epoch(models, train_loader, criterions, optimizers, device, LEARNING_STEP.TRAINING, output, compression, lambd, alpha)
        print(f"[TRAIN] EPOCH {epoch} - GENERATOR | LOSS: {train_generator_loss:2.4f} | BOB | LOSS:{train_bob_loss:2.4f} |")

        if valid_loader is not None:
            print("EPOCH " + str(epoch) + " - VALIDATING...")

            valid_generator_loss, valid_bob_loss = one_epoch(models, valid_loader, criterions, optimizers, device, LEARNING_STEP.VALIDATION, output, compression, lambd, alpha)

            if (valid_generator_loss + valid_bob_loss < best_loss):
                print('--- ACC has improved, saving model ---')
                models.save(output)
                best_loss = valid_generator_loss + valid_bob_loss

        print(f"[VALID] - GENERATOR | LOSS: {valid_generator_loss:2.4f} | BOB | LOSS:{valid_bob_loss:2.4f} |")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)

    # Models
    generator = Generator(in_channels=3, out_channels=3).to(device)
    bob = Bob(in_channels=3, out_channels=1).to(device)
    models = Models(generator, bob)
    if (args.weights != "None"):
        print(f"Load weights from {args.weights}")
        models.load(args.weights)

    # Losses
    generator_loss = GeneratorLoss(device, alpha=args.alpha, loss=args.loss)
    bob_loss = BobLoss(device)
    criterions = Losses(generator_loss, bob_loss)

    # Optimizers
    generator_optimizer = torch.optim.AdamW(models.G.parameters(), lr=2e-4)
    bob_optimizer = torch.optim.AdamW(models.B.parameters(), lr=2e-4)
    optimizers = Optimizers(generator_optimizer, bob_optimizer)

    # Data loaders
    train_loader = loader_with_padding(args.dataset, args.size, args.batch, "train")
    valid_loader = loader_with_padding(args.dataset, args.size, args.batch, "valid")

    # Train
    alpha = args.alpha if args.loss != 'jnd' else 1
    fit(models, args.epochs, criterions, optimizers, device, train_loader, valid_loader, args.output, args.compression, args.lambd, alpha)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch",
        default=2,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/data",
        type=str,
        help="Repository containing the dataset",
    )
    parser.add_argument(
        "-s",
        "--size",
        default=512,
        type=int,
        help="Max size of output",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="None",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "--lambd",
        default=100,
        type=float,
    )
    parser.add_argument(
        "-l",
        "--loss",
        default="jnd",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--compression",
        action='store_true',
    )
    args = parser.parse_args()
    main(args)