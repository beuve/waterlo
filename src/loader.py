import torch
import torch.nn.functional as F
import torchvision.transforms as T

import os
from PIL import Image
import csv


def list_images(directory):
    paths = []
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            file = os.path.join(root, name)
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPEG"):
                paths.append(file)
    return paths


def doc_images(directory):
    paths = {}
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            file = os.path.join(root, name)
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPEG"):
                img_id = file.split("/")[-1].split(".")[0]
                source_id = f'{img_id[:3]}_{img_id[8:]}'
                paths[source_id] = (img_id, file)
    return paths


def get_annotations(directory):
    annotations = {}
    with open(os.path.join(directory, 'facial_landmarks.txt')) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            annotations[f'{row[0]}'] = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
    return annotations


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, directory, transforms=None):
        self.labels = os.listdir(directory)
        if len(self.labels) == 2:
            self.labels.reverse()
        self.transforms = transforms
        self.paths = list_images(directory)
        print(f"found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_nb = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path).convert("RGB")
        size = img.size
        if self.transforms is not None:
            img = self.transforms(img)
        return img_nb, img, size


class DataLoaderAnnotations(torch.utils.data.Dataset):

    def __init__(self, directory, transforms=None):
        self.labels = os.listdir(directory)
        self.directory = directory
        if len(self.labels) == 2:
            self.labels.reverse()
        self.transforms = transforms
        self.paths = list_images(directory)
        self.annotations = get_annotations(directory)
        print(f"found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_nb = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path).convert("RGB")
        source_nb = '_'.join(img_nb.split('_')[:2])
        annotations = self.annotations[source_nb]
        size = img.size

        if self.transforms is not None:
            img = self.transforms(img)

        return img_nb, img, size, annotations


def padding(x, size):
    h, w = x.shape[1], x.shape[2]
    new_h = size
    new_w = size
    if max(h, w) > size:
        x = T.Resize(size - 1, max_size=size)(x)
        h, w = x.shape[1], x.shape[2]
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    mask = torch.zeros(new_h, new_w)
    mask[padding_top:padding_top + h, padding_left:padding_left + w] = 1
    return F.pad(
        x.unsqueeze(0),
        (padding_left, padding_right, padding_top, padding_bottom),
        mode='constant',
    ).squeeze()


def loader_with_padding(path, img_size: int, batch_size=1, split='test', shuffle=True, annotations=False, drop_last=True):

    transforms_list = T.Compose([T.ToTensor(), T.Lambda(lambda x: padding(x, size=img_size))])

    if annotations:
        dataset = DataLoaderAnnotations(
            path + "/" + split,
            transforms=transforms_list,
        )
    else:
        dataset = DataLoader(
            path + "/" + split,
            transforms=transforms_list,
        )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, num_workers=1, shuffle=shuffle)
    return loader


def loader_with_resize(path, img_size, batch_size=1, split='test', shuffle=True, annotations=False, drop_last=True):

    transforms_list = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    if annotations:
        dataset = DataLoaderAnnotations(
            path + "/" + split,
            transforms=transforms_list,
        )
    else:
        dataset = DataLoader(
            path + "/" + split,
            transforms=transforms_list,
        )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, num_workers=1, shuffle=shuffle)
    return loader