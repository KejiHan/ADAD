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
from save_model import sour_cls, test_loader, source_m, classifier

from framwork import ADDA_framwork, s_m, t_m, dis0, dis1, dis2, classifier
from adv_training import model, adv_example, test_loader, source_m

#model1=ADDA_framwork()
model1=torch.load('/home/hankeji/Desktop/new_model/adda_35.pkl')
model10=model1.source_m
model11=model1.classifier
model2=model
modle2=torch.load('/home/hankeji/Desktop/new_model/adv_trianing_35.pkl')
label=torch.load('/home/hankeji/Desktop/ADDA/data/label.pkl')
def compare():
    n_adda=0
    n_adv_training=0
    for i in range(adv_example.size()[0]):
        if i>7999:
            data=adv_example[i]
            target=label[i]
            data, target= Variable(data), Variable(target)

            output0=model10(data)
            output0=model11(output0)
            pred0=output0.data.max(1)[1]
            n_adda+=pred0.eq(target.data).cpu().sum()

            output1=model2(data)
            pred1=output1.data.max(1)[1]
            n_adv_training+= pred1.eq(target.data).cpu.sum()
    print ('ADDA accuracy is {:.4f}'.format(n_adda/2000))
    print ('Adv_training accuracy is {:.4f}'.format(n_adv_training/2000))


if __name__=='__main__':
    compare()