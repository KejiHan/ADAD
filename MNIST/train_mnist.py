#-*-coding:utf-8-*-
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
#import cv2
import os
from models import model3
from atda import margin_loss,mmd_loss,coral_loss

batch_size=512
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
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3)
        self.conv3 = nn.Conv2d(30, 10, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(90, 30)
        self.fc2 = nn.Linear(30, 10)
        self.norm1=nn.BatchNorm2d(1)
        self.norm2=nn.BatchNorm2d(30)
        self.norm3=nn.BatchNorm2d(20)
        self.norm4=nn.BatchNorm1d(90)
        self.norm5=nn.BatchNorm1d(30)

    def forward(self, x):
        x=self.norm1(x)
        x=self.conv1(x)
        x = F.max_pool2d(x, 2)
        x=self.norm3(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x = F.max_pool2d(x, 2)                                                                
        x=self.conv3(x)
        x=self.conv2_drop(x)
        #print(x.size())
        x=x.view(-1,90)
        x=self.fc1(self.norm4(x))
        x=self.fc2(self.norm5(x))
        #print(x.size())
        return x

def train_atda(epoch):
    n=0
    model.train()
    start=0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        end=start+len(data)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data,requires_grad=True), Variable(target)
        if batch_idx==0:
            marker=0
            centers=0
        optimizer.zero_grad()
        output = model(data)
        output=F.log_softmax(output)
        
        loss_ori=F.nll_loss(output, target)
        loss_ori.backward()
        adv_data=data+0.1*torch.sign(data.grad)
        adv_data=Variable(adv_data.data,requires_grad=True)
        pred = output.data.max(1)[1]
        n += pred.eq(target.data).cpu().sum()
        
        optimizer.zero_grad()
        adv_output=model(adv_data)
        adv_output=F.log_softmax(adv_output)
        adv_label=adv_output.max(1)[1]

        output = model(data)
        output = F.log_softmax(output)
        output_cat = torch.cat((output, adv_output), 0)
        #output_cat=Variable(output_cat.data,requires_grad=True)
        label_cat=torch.cat((target,adv_label),0)
        centers,margin_loss1=margin_loss(label_cat, output_cat, num_classes=10, alpha=0.1,marker=marker,centers_old=centers)
        marker=marker+1
        
        
        loss1 = F.nll_loss(output, target)+F.nll_loss(adv_output, adv_label)+1/3*1e-3*(margin_loss1+mmd_loss(output,adv_output)+\
                                                                                  coral_loss(adv_output,output))#margin_loss1\
                                                            # +coral_loss(output,adv_output))
        loss1.backward()
        optimizer.step()
        if batch_idx % 10==0:
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                epoch, end, len(train_loader.dataset),
               loss_ori.data[0]))
        start=end
    tmp_acc=100*n / len(train_loader.dataset)
    print('\n')
    print('Train accuracy is {:.4f}'.format(tmp_acc))

    return tmp_acc


def train_mnist(epoch):
    n = 0
    model.train()
    start = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        end = start + len(data)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        if batch_idx == 0:
            marker = 0
            centers = 0
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output)

        loss_ori = F.nll_loss(output, target)
        loss_ori.backward()
        adv_data = data + 0.1 * torch.sign(data.grad)
        adv_data = Variable(adv_data.data, requires_grad=True)
        pred = output.data.max(1)[1]
        n += pred.eq(target.data).cpu().sum()

        optimizer.zero_grad()
        adv_output = model(adv_data)
        adv_output = F.log_softmax(adv_output)
        adv_label = adv_output.max(1)[1]

        output = model(data)
        output = F.log_softmax(output)
        output_cat = torch.cat((output, adv_output), 0)
        # output_cat=Variable(output_cat.data,requires_grad=True)
        label_cat = torch.cat((target, adv_label), 0)
        centers, margin_loss1 = margin_loss(label_cat, output_cat, num_classes=10, alpha=0.1, marker=marker,
                                            centers_old=centers)
        marker = marker + 1

        loss1 = F.nll_loss(output, target) + F.nll_loss(adv_output, adv_label) + 1 / 3 * 1e-3 * (
                    margin_loss1 + mmd_loss(output, adv_output) + \
                    coral_loss(adv_output, output))  # margin_loss1\
        # +coral_loss(output,adv_output))
        loss1.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                epoch, end, len(train_loader.dataset),
                loss_ori.data[0]))
        start = end
    tmp_acc = 100 * n / len(train_loader.dataset)
    print('\n')
    print('Train accuracy is {:.4f}'.format(tmp_acc))

    return tmp_acc
def b1(epoch):
    model.eval()
    #test_loss = 0
    correct = 0
    for data, target in test_loader:

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
     
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    return 100* correct / len(test_loader.dataset)


if __name__=='__main__':

    epoches=15
    acc=[]
    acc=np.asarray(acc, np.float32)
    model=classifier().cuda()
    #model=Net().cuda()
    optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for i in range(epoches):
        train_acc=train_mnist(i)
        acc=np.append(acc,i)
        acc=np.append(acc, train_acc)
        test_acc=b1(i)
        acc=np.append(acc, test_acc)
        print ('Test accuracy is {:.4f}'.format(test_acc))
        print ('\n' * 2)
    torch.save(model, './model/mnist_daat.pkl')
    acc=np.reshape(acc, (-1, 3))
    #np.save('./data/mnist_Net_train_', acc)

    #acc=np.load('./data/mnist_Net_train_acc.npy')
    #print(acc[:,1])
    from pylab import *
    plt1, = plt.plot(acc[:, 0], acc[:, 1], 'b', label='train')
    plt2, = plt.plot(acc[:, 0], acc[:, 2], 'k', label='test')
    plt.legend(handles=[plt1, plt2])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('(epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
