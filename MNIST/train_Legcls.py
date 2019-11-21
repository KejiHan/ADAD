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
import random
from torchvision.utils import save_image
from MagNet import AE0

from train_mnist import classifier
from GAN_to_generate import Detector
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hankeji/Desktop/papercode/tmp/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=1024, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./tmp', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=400, shuffle=False, **kwargs)


def train_leg_cl(epoch):
    acc=0
    for i, (data, target) in enumerate(train_loader):
        print('Here coming {}_th epoch: {}_th batch'.format(epoch, i))
        # Configure input
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)

        fake_imgs = LegMap(data)  # .detach()

        if i % 30 == 0:
            imgf = fake_imgs.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            # print(imgf)
            imgr = data.data.cpu().numpy()[0]

            imgf = np.reshape(imgf, (28, 28))
            imgr = np.reshape(imgr, (28, 28))
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        output=cls(fake_imgs)
        optim_LegMap.zero_grad()
        output=F.log_softmax(output)
        loss=F.nll_loss(output, target)
        loss.backward()

        #optim_cls.step()
        optim_LegMap.step()

        pred=output.data.max(1)[1]
        acc+=pred.eq(target.data).cpu().sum()
    acc=acc/len(train_loader.dataset)
    print('Test accuracy is {:.4f}'.format(acc))
    return acc

def b1(LegMap, cls):

    correct = 0
    for data, target in test_loader:

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = cls(LegMap(data))

        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    return correct / len(test_loader.dataset)

if __name__=='__main__':

    f= torch.load('./model/mnist.pkl')
    LegMap=torch.load('./model/AE/AE0.pkl')

    cls=f.cuda()
    optim_LegMap=optim.Adam(LegMap.parameters(), lr=1e-3, weight_decay=1e-4)
    #optim_cls=optim.Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)

    epoches=20
    arr=[]
    arr=np.asarray(arr)
    for i in range(epoches):
        tmp_acc=train_leg_cl(i)
        tmp_tacc=b1(LegMap, cls)
        arr=np.append(arr, i)
        arr=np.append(arr, tmp_acc)
        arr=np.append(arr, tmp_tacc)
        if i==58:
            torch.save(LegMap, './model/LegMap_58.pkl')
            torch.save(cls, './model/cls_58.pkl')
    torch.save(LegMap,'./model/LegMap.pkl')
    torch.save(cls, './model/cls.pkl')
    arr=np.reshape(arr, (-1, 3))


    from pylab import *
    plt1, = plt.plot(arr[:, 0], arr[:, 1], 'b', label='train')
    plt2, = plt.plot(arr[:, 0], arr[:, 2], 'k', label='test')
    plt.legend(handles=[plt1, plt2])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('(epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
