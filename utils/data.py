import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

import json
import os

import numpy as np

import cv2

import random


class Yolov7Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.labels = [label for label in load_json(f"{cfg['labels_dir']}det_{cfg['split']}.json") if random.random() < cfg['percent_to_use']]      
        self.labeled_images = [label['name'] for label in self.labels]

    def __len__(self):
        return len(self.labels)

        
    def load_label(self, index):
        boxes = []
        classes = []

        if True:
            label = self.labels[index]

            for object in label['labels']:
                boxes.append([object['box2d']['x1'], object['box2d']['y1'], object['box2d']['x2'], object['box2d']['y2']])
                classes.append(self.cfg['label_to_id'][object['category']])

            boxes = torch.tensor(boxes)
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            boxes = torchvision.ops.box_convert(torch.as_tensor(boxes, dtype=torch.float32), "xyxy", "cxcywh")
            boxes[:, [1, 3]] /= self.cfg['image_shape'][0]
            boxes[:, [0, 2]] /= self.cfg['image_shape'][1]

            classes = np.expand_dims(classes, 1)

            labels_out = torch.hstack(
                (
                    torch.zeros((len(boxes), 1)),
                    torch.as_tensor(classes, dtype=torch.float32),
                    boxes,
                )
            )

            return labels_out


    def load_image(self, index):
        image = cv2.imread(f"{self.cfg['images_dir']}{self.cfg['split']}/{self.labeled_images[index]}")[..., ::-1]
        image = cv2.resize(image, (640, 640))
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = transforms.ToTensor()
        tensor = transform(image) / 255
        tensor.requires_grad = True

        return tensor

    def __getitem__(self, index):
        image = self.load_image(index)
        label = self.load_label(index)

        return image, label       

def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return data

def yolov7_collate_fn(batch):
    images, labels = zip(*batch)
    for i, l in enumerate(labels):
        l[:, 0] = i  # add target image index for build_targets() in loss fn
    return (
        torch.stack(images, 0),
        torch.cat(labels, 0),
    )

def get_dataloader(cfg_path):
    cfg = load_json(cfg_path)

    dataset = Yolov7Dataset(cfg)
    
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = yolov7_collate_fn)

    return dataloader


if __name__ == '__main__':
    cfg = load_json('cfg/dataset.json')
    dataset = Yolov7Dataset(cfg)

    dataloader = DataLoader(dataset, batch_size = 5, shuffle = True, collate_fn = yolov7_collate_fn)

    for i, data in enumerate(dataloader):
        print(data[1])


