import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

import numpy as np
import cv2
from PIL import Image

import matplotlib  
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from yolov7 import create_yolov7_model
from yolov7 import create_yolov7_loss
from yolov7.trainer import filter_eval_predictions
from yolov7.plotting import show_image

from utils.data import get_dataloader

import json

from tqdm import tqdm

import torchvision

from PIL import Image

import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_yolov7_model('yolov7').to(device)
model.eval()

loss_func = create_yolov7_loss(model, image_size = 640, box_loss_weight = 0, cls_loss_weight = 0, obj_loss_weight = 1, ota_loss = True)
loss_func.to(device)
dataloader = get_dataloader('cfg/dataset.json')

patch = torch.rand([1, 3, 640, 640], requires_grad = True)
init_patch = patch.detach().clone()

optimizer = torch.optim.SGD([patch], lr=1000, momentum=0.9)

loop = tqdm(dataloader)

for i in range(1):
    for i, data in enumerate(loop):
        optimizer.zero_grad()

        inputs, labels = data
        outputs = model(patch)

        loss = loss_func(fpn_heads_outputs = outputs, targets = labels, images = patch)[0]
        loss.backward()

        optimizer.step()

torchvision.utils.save_image(patch, '/Users/jaden/Dev/isef/adv_yolov7/patches/patch.png')
torchvision.utils.save_image(init_patch, '/Users/jaden/Dev/isef/adv_yolov7/patches/init_patch.png')
print(torch.norm(init_patch - patch))




