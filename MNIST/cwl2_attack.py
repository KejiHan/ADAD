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
import os
import cv2
#from data_process import data_process
#from train_mnist import  FC
from train_mnist import classifier, Net
batch_size=128
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

f= torch.load('./model/mnist.pkl')
f.eval()
from attack_carlini_wagner_l2 import AttackCarliniWagnerL2

def namestr(obj, namespace):# get var name
    return [name for name in namespace if namespace[name] is obj]

def attack(dataset):
    adversary=AttackCarliniWagnerL2(max_steps=1000)
    adv_data=[]
    for batch_idx,(data, target) in enumerate(dataset):
        data, target=data.cuda(), target.cuda()
        tmp_adv_data=adversary.run(f,data, (target),batch_idx)

        if batch_idx==0:
            adv_data=tmp_adv_data
        else:
            adv_data=np.vstack((adv_data, tmp_adv_data))

    np.save('./data/c&w_attack_test.npy', adv_data)



def test_accuarcy(cw_data):
    start=0
    acc_cw=0
    acc_ori=0
    for batch_idx, (data, target) in enumerate(test_loader):
        size=len(data)
        end=start+size
        tmp_data=cw_data[start:end]
        tmp_data=torch.from_numpy(tmp_data)

        tmp_data, data, target=tmp_data.cuda(), data.cuda(), target.cuda()
        tmp_data=Variable(tmp_data, requires_grad=True)
        data=Variable(data, requires_grad=True)
        target=Variable(target,requires_grad=True)

        output_ori=f(data)
        pred_ori=output_ori.data.max(1)[1]
        acc_ori+=pred_ori.eq(target.data).cpu().sum()

        output_cw=f(tmp_data)
        pred_cw=output_cw.data.max(1)[1]
        acc_cw+=pred_cw.eq(target.data).cpu().sum()

        start=end
        print("Here coming {}/{}".format(end,len(test_loader.dataset)))
    print("Ori accuracy is: {}".format(acc_ori/len(test_loader.dataset)))
    print('CW accuracy is: {}'.format(acc_cw/len(test_loader.dataset)))
    return 0



if __name__=='__main__':
    attack(test_loader)
    '''
    data=np.load('./data/c&w_attack.npy')
    print(data.shape)
    data=np.reshape(data, (-1,1,28,28))
    for i in range(5):
        cv2.imshow('kk', data[i,0,:,:])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cw_data=data
    test_accuarcy(cw_data)
    '''