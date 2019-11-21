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
from train_mnist import classifier

from autoencoder_discriminator import Genetor
from WGAN import Generator1

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=1000, shuffle=False, **kwargs)

cw_data=np.load('./data/c&w_attack.npy')
cw_data=np.transpose(cw_data, (0,3,2,1))
def search_threshold():
    start=0
    loss_ori=0
    loss_auto=0
    dif=[]
    dif=np.asarray(dif, np.float32)
    for batch_idx, (data, target) in enumerate(train_loader):
        size=len(data)
        end=start+size
        print("Here coming data {}/{}".format(end, len(train_loader.dataset)))
        tmp_cw_data=cw_data[start:end]
        tmp_cw_data=torch.from_numpy(tmp_cw_data)

        print(tmp_cw_data[0].max())
        print(data[0].max())

        tmp_cw_data=tmp_cw_data.cuda()
        data,target=data.cuda(),  target.cuda()
        tmp_cw_data=Variable(tmp_cw_data)
        data, target=Variable(data), Variable(target)

        tmp_data=g(data)
        tmp_tmp_cw_data=g(tmp_cw_data)

        loss_ori=torch.norm(data-tmp_data,2).cpu().data.sum()
        #print(type(loss_ori))
        loss_auto=torch.norm(tmp_cw_data-tmp_tmp_cw_data,2).cpu().data.sum()
        start=end
        dif=np.append(dif, batch_idx)
        dif=np.append(dif, loss_ori)
        dif=np.append(dif, loss_auto)
    #print('Mean loss_ori is {} | Mean loss_auto is {}'.format(loss_ori/len(train_loader.dataset), \
    #                                                          loss_auto/len(train_loader.dataset)))
    return dif


if __name__=='__main__':
    g=torch.load('./model/auto_dis_without_classloss.pkl')
    #g=torch.load('./model/wgan_mnist_generator.pkl')
    g.eval()
    dif=search_threshold()
    dif=np.reshape(dif, (-1, 3))
    from pylab import *

    plt1, = plt.plot(dif[:, 0], dif[:, 1], 'b', label='ori')
    plt2, = plt.plot(dif[:, 0], dif[:, 2], 'k', label='auto')
    plt.legend(handles=[plt1, plt2])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel( 'Batch Number', fontsize=15)
    plt.ylabel('Difference of Per Batch', fontsize=15)
    ax = plt.axes()
    plt.show()
