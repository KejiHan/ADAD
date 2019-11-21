from __future__ import division
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import cv2
import os

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=1024, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=400, shuffle=False, **kwargs)
for batch_idx, (data, target) in enumerate(test_loader):
    print ('Here coming {}_th batch'.format(batch_idx))
    if batch_idx==0:
        ori_data=data
        ori_target=target
    else:
        ori_data=torch.cat((ori_data, data), 0)
        ori_target=torch.cat((ori_target, target), 0)
ori_target=ori_target.numpy()
ori_data=ori_data.numpy()
np.save('./data/ori_data.npy', ori_data)
np.save('./data/ori_label.npy', ori_target)