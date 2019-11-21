from __future__ import division
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import time
import cv2
import os
import random


def coral_loss(source, target):
    sm=(source.mean(1).view(-1,1).repeat(1,source.size()[1])-source)
    #print(source.mean(1).size())
    sc=torch.mm(torch.t(sm),sm)

    tm = target.mean(1).view(-1,1).repeat(1,target.size()[1])-target
    tc = torch.mm(torch.t(tm), tm)
    
    loss=torch.mean(torch.abs(sc-tc))
    return loss
def mmd_loss(source, target):
    sm=source.mean(0)
    tm=target.mean(0)
    return torch.mean(torch.abs(sm-tm))

def margin_loss(label, features, num_classes, alpha=0.1,marker=0,centers_old=0):
    eps=1e-3
    lenf_features=features.size()[1]
    centers=Variable((torch.zeros(num_classes,lenf_features)))
    
    
    unique_n=[]
    unique_n=np.asarray(unique_n,np.float32)
    tmp_label = Variable(torch.from_numpy(np.zeros((len(label),1),np.int64)))
    for i in range(num_classes):
        if i==0:
            unique_n=tmp_label.data.eq(label.data).sum()
        else:
            unique_n=np.vstack((unique_n,tmp_label.data.eq(label.data).sum()))
        tmp_label=tmp_label+1
    for j in range(len(features)):
        centers[int(label.cpu().data.numpy()[j])].data+=features[j].data
    unique_n=Variable(torch.from_numpy(unique_n))
    unique_nn=Variable(torch.from_numpy(np.asarray(unique_n.clone().repeat(1, lenf_features).data.cpu().numpy(),np.float32)))
    
    centers_cur=(centers)/(unique_nn+eps )
    if marker==0:
        centers_old=centers_cur
    else:
        centers_old=centers_old-alpha*((centers_old*unique_nn-centers)/(unique_nn+1))
    label = label.view(-1)
   
    label=Variable(torch.from_numpy(np.asarray(label.data.cpu().numpy(),np.int64))).view(-1,)
    
    center_feature = torch.index_select(centers_old,0, label)
    #print(center_feature.size())
    center_dist_norm = features - center_feature
    
    label=label.cpu().data.numpy()
    random.shuffle(label)
    label=Variable(torch.from_numpy(label))
    loss=0
    for i in range(9):
        label=torch.fmod(label+1,num_classes)
        center_feature_shuffle = torch.index_select(centers_old,0, label)
        center_dist_shuffle = features-center_feature_shuffle
        loss=loss+torch.norm(center_dist_shuffle)
    loss=F.softplus(9*torch.norm(center_dist_norm,1)-loss)
    
    return centers_old, loss
if __name__=='__main__':
    a=Variable(torch.ones(12,1))
    b=Variable(torch.ones(12,10))
    print(type(a.data))
    loss,a=margin_loss(a,b,10)
    print(loss)