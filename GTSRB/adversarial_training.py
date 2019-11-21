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

from MagNet import AE0,AE1,AE2,AE3,AE4,AE5
#from AdvGAN import Detector
#from train_Legcls import b1
from search_networks import model0, model2


batch_size=1000


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

def at(epoch, adv_examples):
    n = 0
    n_adv = 0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).view(-1,)
        end = start + data.size()[0]
        print('Here coming {} epoch | {}/{}'.format(epoch, end, len(train_loader.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start = end

        output_ori = cls(LegMap(data))
        loss = F.nll_loss(output_ori, target)
        pred = output_ori.data.max(1)[1]
        n += pred.eq(target.data).cpu().sum()
        optim_LegMap.zero_grad()
        optim_cls.zero_grad()
        loss.backward()
        optim_cls.step()
        optim_LegMap.step()


        output_adv = cls((adv_data))
        pred_adv = output_adv.data.max(1)[1]
        n_adv += pred_adv.eq(target.data).cpu().sum()
        loss = F.nll_loss(output_adv, target)
        optim_LegMap.zero_grad()
        optim_cls.zero_grad()
        loss.backward()
        optim_cls.step()
        optim_LegMap.step()

def eat(epoch):


    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target).view(-1,)
        end = start + data.size()[0]
        print('Here coming {} epoch | {}/{}'.format(epoch, end, len(train_loader.dataset)))
        start=end

        output1=model(data)
        output1=F.log_softmax(output1,dim=1)
        loss1=F.nll_loss(output1,target)
        loss1.backward()
        adv1=data+0.1*torch.sign(data.grad)
        adv1=Variable(adv1.data,requires_grad=True)

        output2 = submddel(data)
        output2 = F.log_softmax(output2,dim=1)
        loss2 = F.nll_loss(output2, target)
        loss2.backward()
        adv2 = data + 0.1 * torch.sign(data.grad)
        adv2=Variable(adv2.data,requires_grad=True)

        output0=model(data)
        output0=F.log_softmax(output0,dim=1)
        output01 = model(adv1)
        output01 = F.log_softmax(output01,dim=1)
        output02 = model(adv2)
        output02 = F.log_softmax(output02,dim=1)

        loss=F.nll_loss(output0,target)+F.nll_loss(output01, target)+F.nll_loss(output02,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 0


def b1(cls,dataset, adv_examples):
    start=0
    ori_correct = 0
    adv_correct=0
    for batch_idx, (data, target) in enumerate(dataset):
        if batch_idx>2:
            break
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target).view(-1,)
        end = start + data.size()[0]
        print('Here coming {}/{}'.format(end, len(dataset.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start=end

        img_ori = data[0].data.cpu().numpy()
        img_ori = np.transpose(img_ori, (1, 2, 0))
        img_rec = adv_data[0].data.cpu().numpy()
        img_rec = np.transpose(img_rec, (1, 2, 0))
        img = np.hstack((img_ori, img_rec))
        cv2.imshow('kk', img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        output =cls((data))
        output=F.log_softmax(output,dim=1)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        ori_correct += pred.eq(target.data).cpu().sum()

        output_adv= cls((adv_data))
        output_adv=F.log_softmax(output_adv,dim=1)
        pred = output_adv.data.max(1)[1]  # get the index of the max log-probability
        adv_correct += pred.eq(target.data).cpu().sum()
        #print(adv_correct)
    print('Test ori accuracy is {:.4f}'.format(100*ori_correct /3000))
    print('Test adv accuracy is {:.4f}'.format(100*adv_correct /3000))
    return ori_correct / len(dataset.dataset),adv_correct / len(dataset.dataset)
if __name__=='__main__':
    
    '''
    LegMap =AE0().cuda() #torch.load('./model/LegMap_GTSRB_sep.pkl')
    cls =model0().cuda()# torch.load('./model/GTSRB_submodels_30_0.pkl')
    optim_LegMap = optim.Adam(LegMap.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_cls = optim.Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)

    adv_examples1 = np.load('./data/c&w_attack_GTSRB_train_128.npy')
    #adv_examples2 = np.load('./data/Adv_GAN_GTSRB_fake_train.npy')
    adv_examples3 = np.load('./data/FGSM_GTSRB_fake_0.1_train.npy')
    adv_examples_list=[adv_examples1, adv_examples3]
    adv_examples_name_list=['c&w', 'FGSM']
    for i in range(1):
        adv_examples=adv_examples_list[i]
        for j in range(30):
            at(j, adv_examples)
        torch.save(cls, './model_at_cls_newauto_'+adv_examples_name_list[i]+'.pkl')
        torch.save(LegMap, './model_at_LegMap_newauto_' + adv_examples_name_list[i] + '.pkl')
    '''
    cls= torch.load('./model/GTSRB_eat.pkl').cuda()
    cls.eval()
    #cls=torch.load('./model_at_cls_new_FGSM.pkl')
    #cls = torch.load('./model/cls_GTSRB_co_58.pkl')
    #LegMap = torch.load('./model/LegMap_GTSRB_co_58.pkl')
    #cls = torch.load('./model/cls_GTSRB_co_58.pkl')
    #LegMap = torch.load('./model/LegMap_GTSRB_co_58.pkl')
    #adv_examples = np.load('./data/FGSM_GTSRB_fake_0.05_test.npy')
    #adv_examples = np.load('./data/c&w_attack_GTSRB_onebatch.npy')
    adv_examples = np.load('./data/c&w_attack_GTSRB_test.npy')
    #cls=torch.load('./model_at_cls_newauto_FGSM.pkl')
    #LegMap=torch.load('./model_at_LegMap_newauto_FGSM.pkl')
    b1(cls, test_loader, adv_examples)
    #################
    #     EAT       #
    #################
    '''
    model=torch.load('./model/GTSRB.pkl').cuda()
    optimizer=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    submddel=torch.load('./model/GTSRB2.pkl').cuda()
    epochs=20
    for i in range(epochs):
        eat(i)
    torch.save(model,'./model/GTSRB_eat.pkl')
    '''