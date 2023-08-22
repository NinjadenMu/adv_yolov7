import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

import json
import os

import numpy as np

import cv2


class Yolov7Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.labels = load_json(f"{cfg['labels_dir']}det_{cfg['split']}.json")
        self.labeled_images = [label['name'] for label in self.labels]

    def __len__(self):
        return len(self.labels)
    
    def bdd_to_yolo(self, object, image_shape, label_to_id):
        x_center = ((int(object['box2d']['x2']) - int(object['box2d']['x1'])) / 2 + int(object['box2d']['x1'])) / image_shape[0]
        y_center = ((int(object['box2d']['y2']) - int(object['box2d']['y1'])) / 2 + int(object['box2d']['y1'])) / image_shape[1]

        width = (int(object['box2d']['x2']) - int(object['box2d']['x1'])) / image_shape[0]
        height = (int(object['box2d']['y2']) - int(object['box2d']['y1'])) / image_shape[1]

        return [0, label_to_id[object['category']], x_center, y_center, width, height]
        
    def load_label(self, index):
        try:
            label = self.labels[index]

            objects = []
            for object in label['labels']:
                objects.append(self.bdd_to_yolo(object, self.cfg['image_shape'], self.cfg['label_to_id']))

            return torch.tensor(objects)
        
        except:
            return torch.zeros((0, 6))

    def load_image(self, index):
        image = cv2.imread(f"{self.cfg['images_dir']}{self.cfg['split']}/{self.labeled_images[index]}")
        image = cv2.resize(image, (640, 640))

        transform = transforms.ToTensor()
        tensor = transform(image)[None, :]
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


if __name__ == '__main__':
    cfg = load_json('cfg/dataset.json')
    dataset = Yolov7Dataset(cfg)

    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

    for i, data in enumerate(dataloader):
        pass


