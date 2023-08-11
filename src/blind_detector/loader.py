from pathlib import Path
import torch
import os
import csv
from PIL import Image
import torchvision.transforms as transforms


def find_label(path, labels, root):
    for i, l in enumerate(labels):
        p = Path(path)
        _root = Path(os.path.join(root, l))
        if _root in (p, *p.parents):
            return i


def list_images(directory, labels):
    paths = []
    for root, _, files in os.walk(directory, topdown=False):
        l = find_label(root, labels, directory)
        for name in files:
            file = os.path.join(root, name)
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                paths.append((file, l))
    return paths


def get_annotations(directory):
    annotations = {}
    with open(os.path.join(directory, 'facial_landmarks.txt')) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            annotations[f'{row[0]}_{row[1]}'] = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))
    return annotations


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, directory, transforms=None, shuffle=True):
        self.labels = os.listdir(directory)
        self.labels.remove('facial_landmarks.txt')
        self.transforms = transforms
        print(self.labels)
        self.paths = list_images(directory, self.labels)
        self.annotations = get_annotations(directory)
        print(f"found {len(self.paths)} file from {len(self.labels)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, label = self.paths[index]
        source_label = '_'.join(img_path.split('/')[-1].split('.')[0].split('_')[:2])
        (x, y, w, h) = self.annotations[source_label]
        img = Image.open(img_path).convert("RGB")
        img = img.crop((x, y, x + w, y + h))
        if self.transforms is not None:
            img = self.transforms(img)
        l = [0] * len(self.labels)
        l[label] = 1
        l = torch.tensor(l)
        return img, l


def loader(path, img_size, batch_size=1, split='test', shuffle=True):

    transforms_list = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = DataLoader(
        path + "/" + split,
        transforms=transforms_list,
        shuffle=shuffle,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )
    return loader