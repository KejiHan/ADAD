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
from data_process import data_process
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
adv_data=torch.utils.data.DataLoader(datasets.MYDATA('../tmp', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=100, shuffle=True, **kwargs)
#pth0='/home/hankeji/Desktop/Adversarial Examples/Train-FGSM-0.2.npy'
#adv_example=torch.load('/home/hankeji/Desktop/Adversarial Examples/Cat_Train-FGSM-0.2.pkl')



class target_m(nn.Module):
    def __init__(self):
        super(target_m, self).__init__()
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

class discri0(nn.Module):
    def __init__(self):
        super(discri0, self).__init__()
        self.fc1 = nn.Linear(784, 392)

    def forward(self, x):
        x=x = x.view(-1,784)
        x=self.fc1(x)
        return x


class discri1(nn.Module):
    def __init__(self):
        super(discri1, self).__init__()
        self.fc1 = nn.Linear(392, 196)

    def forward(self, x):
        x = self.fc1(x)
        return x


class discri2(nn.Module):
    def __init__(self):
        super(discri2, self).__init__()
        self.fc1 = nn.Linear(196, 50)

    def forward(self, x):
        x = self.fc1(x)
        return x

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.norm1 = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(10)
        self.norm3 = nn.BatchNorm2d(20)
        self.norm4 = nn.BatchNorm1d(320)
        self.norm5 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.norm1(x)), 2))
        # print (x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.norm2(x))), 2))
        x=x.view(-1, 320)
        x=self.fc1(self.norm4(x))
        x=self.fc2(self.norm5(x))
        return F.log_softmax(x)

en=target_m()
clas=classifier()
class train_m(nn.Module):
    def __init__(self):
        super(train_m,self).__init__()
        self.m=en
        self.clas=clas
    def forward(self, x):
        x0=self.m(x)
        y=self.clas(x0)
        return x0, F.log_softmax(y)

def train_encoder(epoch):
    n=0
    for batch_idx, (data, target) in enumerate(train_loader):
        print ('Trian _encoder: here coming {}_epoch: {}_th batch'.format(epoch, batch_idx))
        if torch.cuda.is_available():
            data, target= data.cuda(), target.cuda()
            data, target=Variable(data, requires_grad=True), Variable(target)
        x0, output=model(data)
        if batch_idx%10==0:
            tmp=x0.data.cpu().numpy()[0]
            tmp=np.reshape(tmp,(28,28))
            cv2.imshow('loss', tmp)
            cv2.waitKey(200)
            #cv2.destroyAllWindows()
        pred=output.data.max(1)[1]
        n+=pred.eq(target.data).cpu().sum()

        #print loss1.size()
        optimizer.zero_grad()
        loss= F.nll_loss(output, target)
        loss.backward()
        #print data.grad
        optimizer.step()
    print ('Train accuracy is {:.4f}'.format(n/len(train_loader.dataset)))
    return n/len(train_loader.dataset)

def train_encoder_loss(epoch):
    n=0
    for batch_idx, (data, target) in enumerate(train_loader):
        print ('Train_encoder_loss: here coming {}_epoch: {}_th batch'.format(epoch, batch_idx))
        if torch.cuda.is_available():
            data, target= data.cuda(), target.cuda()
            data, target=Variable(data, requires_grad=True), Variable(target)
        x0, output=model0(data)
        if batch_idx%10==0:
            tmp=x0.data.cpu().numpy()[0]
            tmp=np.reshape(tmp,(28,28))
            cv2.imshow('loss', tmp)
            cv2.waitKey(200)
            cv2.destroyAllWindows()
        pred=output.data.max(1)[1]
        n+=pred.eq(target.data).cpu().sum()
        loss1 = (data - x0).sum(0).view(28, 28)
        loss1 = torch.norm(loss1, 2)
        #loss2 = torch.norm((data-x0), 2)
        optimizer0.zero_grad()
        loss= F.nll_loss(output, target)+loss1
        loss.backward()
        optimizer0.step()
    print ('Train accuracy is {:.4f}'.format(n/len(train_loader.dataset)))
    return n/len(train_loader.dataset)


if __name__=='__main__':
    model=train_m()
    print (id(model))
    model.cuda()
    optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model0 =train_m()
    print (id(model0))
    model0.cuda()
    optimizer0 = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    acc=[]
    acc=np.asarray(acc, np.float)
    for i in range(100):
        tmp_acc=train_encoder(i)
        tmp_acc0=train_encoder_loss(i)
        acc=np.append(acc, i)
        acc=np.append(acc, tmp_acc)
        acc = np.append(acc, tmp_acc0)
    acc=np.reshape(acc, (-1, 3))
    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2, color='k', marker='D', label='ori')
    plt1, = plt.plot(acc[:, 0], acc[:, 2], linewidth=2, color='b', marker='o', label='2-norm_loss')
    plt.legend(handles=[plt0, plt1])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('$\phi$ (epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()




