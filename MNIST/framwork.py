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
from data_process import data_process
from MagNet import AE0
from WGAN import train_wgan
from train_mnist import classifier
from AdvGAN import Detector

from WGAN import train_wgan
from train_mnist import classifier
#from train_Legcls import cls, LegMap
batch_size=1024
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

#pth0='/home/hankeji/Desktop/Adversarial Examples/Train-FGSM-0.2.npy'
#adv_example=np.load('./data/Adv_GAN_fake.npy')
#adv_example=np.load('./data/FGSM_fake_0.2.npy')

#adv_example=np.reshape(adv_example,(-1,1,28,28))
#print(data.shape)

#adv_example=np.transpose(data,(0,3,2,1))
'''
for i in range(len(adv_example)):
    cv2.imshow('kk',adv_example[i,0, :,:])
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
'''

#print(adv_example.shape)

def generate_fake(data, grad):
    return data + (grad)


def D_loss(logits_real, logits_fake):
    bec = nn.BCEWithLogitsLoss()
    true_labels = Variable(torch.ones(logits_real.size()), requires_grad=True).cuda()
    real_image_loss = bec(logits_real, true_labels)
    fake_image_loss = bec(logits_fake, 1 - true_labels)
    return real_image_loss + fake_image_loss


def G_loss(logits_fake):
    bec = nn.BCEWithLogitsLoss()
    true_labels = Variable(torch.ones(logits_fake.size()), requires_grad=True).cuda()
    loss = bec(logits_fake, true_labels)
    return loss


def train_main(epoch):
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #print batch_idx
        print('Here coming {}_th epoch: {}_th batch'.format(epoch, batch_idx))
        end=start+data.size()[0]
        adv_e=adv_examples[start:end]
        adv_e=torch.from_numpy(adv_e)
        adv_e=adv_e.cuda()
        adv_e=Variable(adv_e, requires_grad=True)
        data, target=data.cuda(), target.cuda()
        data,target=Variable(data, requires_grad=True), Variable(target)
        start=end

        advmap_output = AdvMap(adv_e)
        logits_fake = det(advmap_output)
        logits_real = det(LegMap(data))


        loss_D = D_loss(logits_real, logits_fake)
        #loss_D=(torch.mean(logits_real)-torch.mean(logits_fake))*1e6
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        #for p in det.parameters():
        #   p.data.clamp_(-1, 1)

        '''
        if batch_idx % 30 == 0:
            imgf = adv_e.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1, 2, 0)

            imgr = data.data.cpu().numpy()[0]
            imgr = imgr.transpose(1, 2, 0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        '''
        data_fake0 = AdvMap(adv_e)
        legmap_output = LegMap(data)
        loss1 = G_loss(det(data_fake0))
        #loss1=-torch.mean(det(data_fake0))

        cls_output = cls(data_fake0)
        cls_output=F.log_softmax(cls_output)
        cls_loss = F.nll_loss(cls_output, target)

        loss_G =cls_loss+loss1 + 1e-3 * torch.norm(data_fake0 - legmap_output, 2)
        pred = cls_output.data.max(1)[1]
        n_adv += pred.eq(target.data).cpu().sum()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

    print(n_adv/len(train_loader.dataset))
    return n_adv/len(train_loader.dataset)



def train_main1(epoch):
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #print batch_idx
        print('Here coming {}_th epoch: {}_th batch'.format(epoch, batch_idx))
        end=start+data.size()[0]
        adv_e=adv_examples[start:end]
        adv_e=torch.from_numpy(adv_e)
        adv_e=adv_e.cuda()
        adv_e=Variable(adv_e, requires_grad=True)
        data, target=data.cuda(), target.cuda()
        data,target=Variable(data, requires_grad=True), Variable(target)
        start=end

        advmap_output = AdvMap(adv_e)
        logits_fake = det(advmap_output)
        logits_real = det((data))


        loss_D = D_loss(logits_real, logits_fake)
        #loss_D=(torch.mean(logits_real)-torch.mean(logits_fake))*1e6
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        for p in det.parameters():
           p.data.clamp_(-1, 1)

        '''
        if batch_idx % 30 == 0:
            imgf = adv_e.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1, 2, 0)

            imgr = data.data.cpu().numpy()[0]
            imgr = imgr.transpose(1, 2, 0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        '''
        data_fake0 = AdvMap(adv_e)
        legmap_output = (data)
        loss1 = G_loss(det(data_fake0))
        #loss1=-torch.mean(det(data_fake0))

        cls_output = cls(data_fake0)
        cls_output=F.log_softmax(cls_output)
        cls_loss = F.nll_loss(cls_output, target)

        loss_G =cls_loss+loss1 + 1e-3 * torch.norm(data_fake0 - legmap_output, 2)
        pred = cls_output.data.max(1)[1]
        n_adv += pred.eq(target.data).cpu().sum()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

    print(n_adv/len(train_loader.dataset))
    return n_adv/len(train_loader.dataset)

from MagNet import det_batch_models,det_batch,threshold,reformer

def ADA_main(model_list,thres, dataset, data_adv, evl='adv'):
    n = 0
    n_adv = 0
    thres = thres
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        print(
            'Checking eval_dataset-->{}/{} batch'.format(batch_idx, np.ceil(len(dataset.dataset) / batch_size)))
        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        tmp_adv = data_adv[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)


        if evl=='leg':
            eval_data=data
        else:
            eval_data=tmp_adv
        res = det_batch_models(model_list, thres, eval_data)
        if res > 0:
            tmp_adv=AdvMap(tmp_adv)
            output=cls(tmp_adv)
            pred=output.data.max(1)[1]
            n_adv+=pred.eq(target.data).cpu().sum()
        else:
            tmp_adv = LegMap(data)
            output = cls(tmp_adv)
            pred = output.data.max(1)[1]
            n += pred.eq(target.data).cpu().sum()


    acc = (n_adv +n)/ len(dataset.dataset)
    print(acc)
    print('Done!')
    return acc


def b1(dataset):
    ori_corr = 0
    adv_corr=0
    start=0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target =target.cuda()
        data, targe=Variable(data), Variable(target)
        end=start+len(data)
        tmp_adv = dataset[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)


        output = cls(tmp_adv)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        ori_corr += pred.eq(target.data).cpu().sum()

        tmp_adv=AdvMap(tmp_adv)
        output = cls(tmp_adv)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        ori_corr += pred.eq(target.data).cpu().sum()
    print ('Test accuray is {}'.format(100. * ori_corr / len(test_loader.dataset)))
    return 100. * ori_corr / len(test_loader.dataset)

def eval_Map(dataset, adv_example):
    n_ori = 0
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]
        print('Here coming {}_th batch'.format(batch_idx))

        adv_data = adv_example[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start=end
        LegMap_output=LegMap(data)
        AdvMap_output=AdvMap(adv_data)
        #print('LegMap norm is:{}'.format(LegMap_output.max()))
        #print('Ori norm is:{}'.format(data.max()))
        '''
        if batch_idx % 1 == 0:
            imgf =LegMap_output.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            # print(imgf)
            imgr =data.data.cpu().numpy()[0]

            imgf = np.reshape(imgf, (28, 28))
            imgr = np.reshape(imgr, (28, 28))
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        '''

        output = cls(LegMap_output)
        pred = output.data.max(1)[1]
        n_ori += pred.eq(target.data).cpu().sum()

        output=cls(AdvMap_output)
        pred=output.data.max(1)[1]
        n_adv+=pred.eq(target.data).cpu().sum()

    print('Ori accuracy is {}'.format(n_ori/len(dataset.dataset)))
    print('Adv accuracy is {}'.format(n_adv / len(dataset.dataset)))
    return n_adv / len(dataset.dataset)
if __name__=='__main__':
    '''
    f = torch.load('./model/mnist.pkl')
    cls = f.cuda()
    LegMap = torch.load('./model/LegMap00.pkl')
    # LegMap = torch.load('./model/AE/AE3.pkl')

    #adv_example = np.load('./data/c&w_attack_test.npy')
    #adv_example = adv_example.reshape(-1, 1, 28, 28)
    #adv_example = np.load('./data/FGSM_fake_0.3_test.npy')

    AdvMap = AE0().cuda()#.load('./model/AE/AE0.pkl')
    #AdvMap=torch.load('./model/AdvMap_c&w.pkl')

    with_list=['with', 'without']
    '''
    ################
    # train AdvMap #
    ################
    #f = torch.load('./model/mnist.pkl')
    cls=torch.load('./model/cls.pkl')
    LegMap = torch.load('./model/LegMap.pkl')
    AdvMap = AE0().cuda()  # .load('./model/AE/AE0.pkl')

    # adv_examples = np.load('./data/c&w_attack_test.npy')
    # adv_example = adv_examples.reshape(-1, 1, 28, 28)
    # -print(adv_examples1.shape)
    # adv_example = np.load('./data/FGSM_fake_0.3_test.npy')
    '''
    I_n = 50
    det= Detector().cuda()
    AdvMap=AE0().cuda()
    optim_D = optim.RMSprop(det.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_G = optim.RMSprop(AdvMap.parameters(), lr=1e-3, weight_decay=1e-4)
    for i in range(1):
        acc = []
        acc = np.asarray(acc, np.float)
        adv_examples = adv_examples_list[i]
        for j in range(I_n):
            if k==0:
                a=train_main(j)
            else:
                a=train_main1(j)
            #b = eval_Map(test_loader, adv_example)
            acc = np.append(acc, j)
            acc = np.append(acc, a)
            acc = np.append(acc, b)
        acc = np.reshape(acc, (-1, 3))
        np.save('./data/train_AdvMap_'+with_list[k] + adv_examples_name_list[i] + '_50.npy', acc)
        torch.save(AdvMap, './model/AdvMap_'+with_list[k] + adv_examples_name_list[i] + '_50.pkl')
        torch.save(det, './model/ADtoD_Det_' +with_list[k]+ adv_examples_name_list[i] + '_50.pkl')
    '''
    '''
    I_n = 30
    det = Detector().cuda()
    AdvMap = AE0().cuda()
    optim_D = optim.RMSprop(det.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_G = optim.RMSprop(AdvMap.parameters(), lr=1e-3, weight_decay=1e-4)
    adv_examples = np.load('./data/FGSM_fake_0.2.npy')
    acc = []
    acc = np.asarray(acc, np.float)
    for j in range(I_n):
        a = train_main(j)
        #b = train_main1(j)
        acc = np.append(acc, j)
        acc = np.append(acc, a)
        #acc = np.append(acc, b)
    acc = np.reshape(acc, (-1, 2))
    np.save('./data/train_AdvMap_MNIST_FGSM_with_30.npy', acc)
    '''




    ##################
    # test framework #
    ##################
    '''
    adv_example = np.load('./data/Adv_GAN_fake_test.npy')
    #adv_example = np.reshape(adv_example, (-1, 1, 28, 28))

    # rootpth = './model/AE'
    # model_list = get_model_list(rootpth)
    model1 = torch.load('./model/MagNet_30_0.pkl').cuda()
    model2 = torch.load('./model/MagNet_30_1.pkl').cuda()
    model_list = [model1, model2]
    thres=np.load('./data/MagNet_batch_thre_test.npy')

    ADA_main(model_list,thres, test_loader, adv_example)
    #b1(adv_example)
    '''













