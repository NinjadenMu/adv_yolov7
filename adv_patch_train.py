import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchvision import datasets, transforms

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

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_yolov7_model('yolov7').to(device)
model.eval()

class Patch:
    def __init__(self, loss_target = 'obj', dim = [3, 64, 64], targeted = False, target = None):
        self.loss_target = loss_target
        self.dim = dim
        self.patch = torch.rand(dim, requires_grad = True)
        self.targeted = targeted
        self.target = target

class Applier:
    def __init__(self):
        pass


image = cv2.imread('data/images/100k/val/b1c9c847-3bda4659.jpg')
image = cv2.resize(image, (640, 640))
transform = transforms.ToTensor()
tensor = transform(image)[None, :]
print(image.shape)

import time

start = time.perf_counter()
frames = 0
while time.perf_counter() - start < 10:
    output = model(tensor)
    frames += 1
print(frames)

preds = model.postprocess(output)
nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.25)
#print(nms_predictions[0])

targets = []

for pred in nms_predictions[0]:
    targets.append([0, pred[-1], pred[0] / 1280, pred[1] / 720, pred[2] / 1280, pred[3] / 720])
#print(targets)
targets = torch.tensor([targets])
#print(targets)

#show_image(image[...,::-1], nms_predictions[0][:, :4].tolist(), nms_predictions[0][:, -1].tolist())



def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return data

dataloader = get_dataloader('cfg/dataset.json')

patch = Patch()
#print((225 * patch.patch).permute(1, 2, 0).detach().numpy())
im = Image.fromarray((225 * patch.patch).permute(1, 2, 0).detach().numpy().astype(np.uint8))
im.save("patches/patch.jpg")

optimizer = torch.optim.SGD([patch.patch], lr=1, momentum=0.9)

loss_func = create_yolov7_loss(model, image_size = 640, box_loss_weight = 0, cls_loss_weight = 0, obj_loss_weight = 1, ota_loss = True)
loss_func.to(device)
output = model(tensor[0])
print(targets)
print(loss_func(fpn_heads_outputs = output, targets = targets[0], images = tensor[0])[0])

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loop = tqdm(dataloader)
    for i, data in enumerate(loop):
        # Every data instance is an input + label pair
        inputs, labels = data
        #print(inputs[0].shape)
        inputs = inputs[0]
        
        inputs[0, :3, 0:64, 0:64] = patch.patch

        labels = labels[0]
        #print(inputs.shape, labels.shape)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = -1 * loss_func(fpn_heads_outputs = outputs, targets = labels, images = inputs)[0]
        #print('loss')
        #print(loss)
        loss.backward()

        running_loss += loss[0]

        # Adjust learning weights
        optimizer.step()

    return running_loss / len(loop)
"""
og_patch = patch.patch
for i in range(3):
    print(train_one_epoch(0, 0))
    print(torch.mean(patch.patch - og_patch))
"""

image = cv2.imread('data/images/100k/val/b1c9c847-3bda4659.jpg')
image = cv2.resize(image, (640, 640))
transform = transforms.ToTensor()
tensor = transform(image)
inputs = torch.stack((tensor, tensor, tensor))

print(inputs.shape)
#outputs = model(inputs)

labels_out = torch.hstack((torch.zeros((1, 1)),torch.as_tensor([[1]], dtype=torch.float32),torch.tensor([[0, 0], [0, 0]])))

#loss = loss_func(fpn_heads_outputs = outputs, targets = labels_out, images = inputs)

im = Image.fromarray((225 * patch.patch).permute(1, 2, 0).detach().numpy().astype(np.uint8))
im.save("patches/patch_trained.jpg")

"""
cv2.imshow('Image', (225 * patch.patch).permute(1, 2, 0).detach().numpy().astype(np.uint8))
cv2.waitKey(0)
"""


