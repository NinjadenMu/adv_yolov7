import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt

from yolov7 import create_yolov7_model
from yolov7 import create_yolov7_loss
from yolov7.trainer import filter_eval_predictions


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_yolov7_model('yolov7').to(device)
model.eval()

epsilons = [0, .05, .1, .15, .2, .25, .3]


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


model.zero_grad()

image = cv2.imread('data/images/images/100k/val/b1c9c847-3bda4659.jpg')
image = cv2.resize(image, (640, 640))

transform = transforms.ToTensor()
tensor = transform(image)[None, :]
tensor.requires_grad = True

loss_func = create_yolov7_loss(model)
loss_func.to(device)

targets = torch.tensor([[0, 0, 0.7, 0.7, 0.1, 0.1]])

output = model(tensor)
preds = model.postprocess(output)
nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.25)

loss = loss_func(fpn_heads_outputs = output, targets = targets, images = tensor)
print(loss)
loss[0].backward()

perturbed_tensor = fgsm_attack(tensor, 0.3, tensor.grad)
perturbed_tensor = denorm(perturbed_tensor, torch.mean(tensor), torch.std(tensor))

print(torch.mean(perturbed_tensor - tensor))

perturbed_output = model(perturbed_tensor)
perturbed_preds = model.postprocess(output)
perturbed_nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.25)

perturbed_loss = loss_func(fpn_heads_outputs = perturbed_output, targets = targets, images = perturbed_tensor)
print(perturbed_loss)

cv2.imshow('Image', image)
cv2.waitKey(0)