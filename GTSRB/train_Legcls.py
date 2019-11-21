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
from search_networks import model0
from MagNet import AE0,AE1,AE3,AE4
f=torch.load('./model/GTSRB_submodels_30_0.pkl')

batch_size=2048

kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.GTSRB('./tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.GTSRB('./tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)



def train_leg_cl(epoch):
    acc=0
    for i, (data, target) in enumerate(train_loader):
        print('Here coming evaluating {}_th epoch: {}_th batch'.format(epoch, i))
        # Configure input
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)

        fake_imgs = LegMap(data)  # .detach()

        if i % 30 == 0:
            imgf = fake_imgs.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = data.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        output=cls((fake_imgs))
        pred = output.data.max(1)[1]
        acc += pred.eq(target.data).cpu().sum()


        optim_LegMap.zero_grad()
        loss=F.nll_loss(output, target)#+torch.norm(data-fake_imgs,2)
        print(loss.data.cpu().numpy())
        loss.backward()
        optim_LegMap.step()


    acc=acc/len(train_loader.dataset)
    print('Training accuracy is {:.4f}'.format(acc))
    return acc

def b1(LegMap, cls, adv_examples):
    start=0
    ori_correct = 0
    adv_correct=0
    for data, target in test_loader:

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        end = start + data.size()[0]
        print('Here coming {}/{}'.format(end, len(test_loader.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start=end

        output = cls(LegMap(data))
        pred = output.data.max(1)[1] # get the index of the max log-probability
        ori_correct += pred.eq(target.data).cpu().sum()

        output_adv= cls(LegMap(adv_data))
        pred = output_adv.data.max(1)[1]  # get the index of the max log-probability
        adv_correct += pred.eq(target.data).cpu().sum()
    print('Test ori accuracy is {}'.format(ori_correct / len(test_loader.dataset)))
    print('Test adv accuracy is {}'.format(adv_correct / len(test_loader.dataset)))
    return adv_correct / len(test_loader.dataset)

if __name__=='__main__':
    #cls=torch.load('./model/cls_GTSRB_co_58.pkl')
    #LegMap=torch.load('./model/LegMap_GTSRB_co_58.pkl')
    #adv_examples=np.load('./data/c&w_attack_GTSRB_test_128.npy')

    LegMap=torch.load('./model/GTSRB_AE0.pkl')
    cls=f
    optim_LegMap=optim.Adam(LegMap.parameters(), lr=1e-3, weight_decay=1e-4)
    #optim_cls=optim.Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)

    epoches=30
    arr=[]
    arr=np.asarray(arr)
    for i in range(epoches):
        tmp_acc=train_leg_cl(i)
        tmp_tacc=b1(LegMap, cls, adv_examples)
        arr=np.append(arr, i)
        arr=np.append(arr, tmp_acc)
        arr=np.append(arr, tmp_tacc)
    torch.save(LegMap,'./model/LegMap_GTSRB_sep1.pkl')
    #torch.save(cls, './model/cls_GTSRB_co.pkl')

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
