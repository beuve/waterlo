import argparse
from tqdm import tqdm
import numpy as np

import timm
from timm.data import resolve_data_config
import torch

from loader import loader


def get_output(
    model,
    device,
    loader,
    num_labels=1,
    num=0,
    num_per_labels=None,
):
    outputs = None
    labels = None
    per_labels = [0] * num_labels
    batch_nb = len(loader)
    for i, (data, target) in tqdm(enumerate(loader), total=min(batch_nb, num)):
        data, target = data.to(device), target.to(device)

        if num > 0 and i >= num:
            break

        if (num_per_labels != None and per_labels[target.argmax(dim=1)] >= num_per_labels):
            if np.sum(per_labels) >= num_labels * num_per_labels:
                break
            else:
                continue

        with torch.no_grad():
            output = model.forward(data)
            if outputs == None:
                outputs = output
                labels = target
            else:
                outputs = torch.cat((outputs, output), 0)
                labels = torch.cat((labels, target), 0)

        if num_per_labels != None:
            per_labels[target.argmax(dim=1)] += 1

    return outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()


def get_prediction(model, device, data_folder, input_size):
    test_loader = loader(data_folder, input_size, split='valid')
    test_feats, test_labels = get_output(model, device, test_loader)
    test_preds = test_feats.argmax(axis=1)
    return test_preds, test_labels


def accuracy(preds, labels):
    acc = (preds == labels.argmax(axis=1)).astype(float).mean()

    print(f'ACCURACY ----------- {100* acc:2.4f}')


def balanced_accuracy(outputs, gt, nb_class):
    accs = 0
    for i in range(nb_class):
        acc = ((outputs == i) * (gt.argmax(axis=1) == i)).astype(float).sum()
        nb = (gt.argmax(axis=1) == i).astype(float).sum()
        print(i, acc / nb)
        accs += acc / nb

    print(f'BALANCED ACCURACY -- {100* accs / nb_class:2.4f}')
    return


def main(args):
    model_name = args.model
    weights = args.weights
    data_folder = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model_config = resolve_data_config({}, model=model)
    if args.weights != None:
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    print('=' * 10, 'CONFIG', '=' * 10)
    print('MODEL:              ', model_name)
    print('WEIGHTS:            ', weights)
    print('=' * 28)
    outputs, gt = get_prediction(model, device, data_folder, model_config['input_size'][1])
    print()
    print('=' * 10, 'RESULT', '=' * 10)
    accuracy(outputs, gt)
    balanced_accuracy(outputs, gt, nb_class=2)
    print('=' * 28)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="xception",
        type=str,
        help="TIMM Model",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="path to model weights. Use imagenet from TIMM if unspecified.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/data",
        type=str,
        help="Repository containing the dataset",
    )

    args = parser.parse_args()
    main(args)
