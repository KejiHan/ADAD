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
    batch_size=100, shuffle=True, **kwargs)
f= torch.load('./model/mnist.pkl')
class Genetor(nn.Module):
    def __init__(self):
        super(Genetor, self).__init__()
        self.con1 = nn.Conv2d(1, 20, 5)
        self.con2 = nn.Conv2d(20, 10, 5)
        self.con3=nn.Conv2d(10,20,5)
        self.con4=nn.Conv2d(20,32,5)
        self.fc1 = nn.Linear(4608, 2304)
        self.fc2 = nn.Linear(2304, 4608)

        self.recon1 = nn.ConvTranspose2d(32, 20, 5)
        self.recon2 = nn.ConvTranspose2d(20, 10, 5)
        self.recon3=nn.ConvTranspose2d(10,20,5)
        self.recon4=nn.ConvTranspose2d(20,1, 5)

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x=self.con3(x)
        x=self.con4(x)
        #print (x.size())
        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = self.fc2(x)
        #print (x.size())
        x = x.view(-1, 32, 12, 12)
        x = self.recon1(x)
        x = self.recon2(x)
        x=self.recon3(x)
        x=self.recon4(x)
        #print(x.size())
        return x
def train_auto(epoch):
    n_auto = 0
    n_ori = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.cuda(),target.cuda()
        data, target=Variable(data, requires_grad=True),Variable(target)
        print('Here coming {}_th epcoch: {}_th batch'.format(epoch, batch_idx))

        output_ori=f(data)
        pred_ori=output_ori.data.max(1)[1]
        n_ori+=pred_ori.eq(target.data).cpu().sum()

        en_data=g(data)
        output=f(en_data)
        pred=output.data.max(1)[1]
        n_auto+=pred.eq(target.data).cpu().sum()
        loss1=F.nll_loss(output, target)
        loss2=torch.norm(en_data-data, 2)

        loss=loss2+loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(n_ori, n_auto)
    return n_ori/len(train_loader.dataset), n_auto/len(train_loader.dataset)


if __name__=='__main__':
    epoches=50
    g=Genetor().cuda()
    optimizer=optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
    acc=[]
    acc=np.asarray(acc, np.float)
    for i in range(epoches):
        tmp_acc_ori, tmp_acc_auto=train_auto(i)
        acc=np.append(acc, i)
        acc=np.append(acc, tmp_acc_ori)
        acc=np.append(acc, tmp_acc_auto)
    torch.save(g, './model/auto_dis_without_classloss.pkl')
    acc=np.reshape(acc, (-1, 3))
    from pylab import *
    plt1, = plt.plot(acc[:, 0], acc[:, 1], 'b', label='ori')
    plt2, = plt.plot(acc[:, 0], acc[:, 2], 'k', label='auto')
    plt.legend(handles=[plt1, plt2])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('$\phi$ (epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
