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
from MagNet import AE0,AE1,AE2,AE3,AE4,AE5
from AdvGAN import Detector,eval_adv
from search_networks import model0
f=torch.load('./model/GTSRB_submodels_30_0.pkl')
batch_size=1024

kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.GTSRB('./tmp', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.GTSRB('./tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)


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
    n = 0
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]
        print('Here coming {} epoch | {}/{}'.format(epoch, end, len(train_loader.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start = end

        advmap_output=AdvMap(adv_data)
        logits_fake=det(advmap_output)
        logits_real=det(LegMap(data))
        loss_D = torch.mean(logits_real)-torch.mean(logits_fake)
        #loss_D=D_loss(logits_real, logits_fake)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        for p in det.parameters():
           p.data.clamp_(-1, 1)
        '''
        if batch_idx % 30 == 0:
            imgf = advmap_output.data.cpu().numpy()[0]
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

        data_fake0 = AdvMap(adv_data)
        legmap_output =LegMap(data)
        loss1 = -torch.mean(det(data_fake0))

        cls_output = cls((data_fake0))
        cls_output=F.log_softmax(cls_output)
        cls_loss = F.nll_loss(cls_output, target)

        loss_G = loss1+cls_loss+1e-3*torch.norm(data_fake0-legmap_output, 2)
        pred = cls_output.data.max(1)[1]
        n_adv += pred.eq(target.data).cpu().sum()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

    print('Train accuracy is {:.4f}'.format(n_adv/len(train_loader.dataset)))
    return n_adv/len(train_loader.dataset)

def train_main1(epoch):
    n = 0
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]
        print('Here coming {} epoch | {}/{}'.format(epoch, end, len(train_loader.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start = end

        advmap_output=AdvMap(adv_data)
        logits_fake=det(advmap_output)
        logits_real=det((data))
        loss_D = torch.mean(logits_real)-torch.mean(logits_fake)
        #loss_D=D_loss(logits_real, logits_fake)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        for p in det.parameters():
           p.data.clamp_(-1, 1)
        '''
        if batch_idx % 30 == 0:
            imgf = advmap_output.data.cpu().numpy()[0]
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

        data_fake0 = AdvMap(adv_data)
        loss1 = -torch.mean(det(data_fake0))
        cls_output = cls((data_fake0))
        cls_output=F.log_softmax(cls_output)
        cls_loss = F.nll_loss(cls_output, target)

        loss_G = loss1+cls_loss+1e-3*torch.norm(data_fake0-data, 2)
        pred = cls_output.data.max(1)[1]
        n_adv += pred.eq(target.data).cpu().sum()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()


    print('Train accuracy is {:.4f}'.format(n_adv/len(train_loader.dataset)))
    return n_adv/len(train_loader.dataset)

from MagNet import det_batch_models,det_batch,threshold,reformer,main_batch


def ADA_main(model_list,thres, dataset, data_adv, evl='leg'):
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
        start=end

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
    acc = (n_adv +n)/len(dataset.dataset)
    print(acc)
    print('Done!')
    return acc

def b1(dataset):
    cls.eval()
    correct = 0
    start=0
    for batch_idx, (data, target) in enumerate(test_loader):
        target =target.cuda()
        print(len(data))
        end=start+len(data)
        tmp_adv = dataset[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()

        tmp_adv = Variable(tmp_adv)
        target = Variable(target)

        output = cls(tmp_adv)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    print ('Test accuray is {}'.format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def eval_cls(dataset):
    n = 0
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]
        print('Here coming {}/{}'.format(end, len(dataset.dataset)))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start=end


        output=cls(LegMap(adv_data))
        pred=output.data.max(1)[1]
        n+=pred.eq(target.data).cpu().sum()
    print('Eval accuracy is {}'.format(n/len(dataset.dataset)))
    return 0

def eval_Map(dataset, adv_examples):
    n_ori = 0
    n_adv=0
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]
        print('Here coming {}_th batch'.format(batch_idx))

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start=end
        LegMap_output=LegMap(adv_data)
        print('LegMap norm is:{}'.format(torch.norm(LegMap_output, 2).data.cpu().numpy()))
        print('Ori norm is:{}'.format(torch.norm(data, 2).data.cpu().numpy()))
        AdvMap_putput=AdvMap(data)
        output = cls(LegMap_output)
        pred = output.data.max(1)[1]
        n_ori += pred.eq(target.data).cpu().sum()
        '''
        if batch_idx % 30 == 0:
            imgf =LegMap_output.data.cpu().numpy()[0]
            #imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = AdvMap_putput.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)
            #imgr=imgr/imgr.max()
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        '''

        output=cls(AdvMap_putput)
        pred=output.data.max(1)[1]
        n_adv+=pred.eq(target.data).cpu().sum()

    print('Ori accuracy is {}'.format(n_ori/len(dataset.dataset)))
    print('Adv accuracy is {}'.format(n_adv / len(dataset.dataset)))
    return n_adv / len(dataset.dataset)
def rap(adv_examples):
    n = 0
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        end = start + data.size()[0]

        adv_data = adv_examples[start:end]
        adv_data = torch.from_numpy(adv_data)
        adv_data = adv_data.cuda()
        adv_data = Variable(adv_data)
        start = end
        if batch_idx>0:
            break

        data_ori=LegMap(data)
        data_adv=LegMap(adv_data)

        data=data.data.cpu().numpy()
        adv_data=adv_data.data.cpu().numpy()
        data_ori=data_ori.data.cpu().numpy()
        data_adv=data_adv.data.cpu().numpy()
        np.save('./data/eval_per_ori_data.npy',data)
        np.save('./data/eval_per_adv_data.npy', adv_data)
        np.save('./data/eval_per_data_ori.npy', data_ori)
        np.save('./data/eval_per_data_adv.npy', data_adv)
        print('Done!!!')

if __name__=='__main__':
    #cls =torch.load('./model/GTSRB_submodels_0.pkl')
    cls= torch.load('./model/GTSRB_submodels_30_0.pkl')
    LegMap = torch.load('./model/LegMap_GTSRB_sep1.pkl')
    #LegMap = torch.load('./model/GTSRB_AE0.pkl')
    #AdvMap = torch.load('./model/GTSRB_AdvMap0_FGSM_200.pkl')
    #AdvMap = torch.load('./model/GTSRB_AdvMap0_c&w_200.pkl')

    #adv_examples1 = np.load('./data/c&w_attack_GTSRB_train_128.npy')
    #adv_examples2 = np.load('./data/Adv_GAN_GTSRB_fake_train.npy')
    #adv_examples3 = np.load('./data/FGSM_GTSRB_fake_0.05_train.npy')
    #adv_examples_list=[adv_examples1, adv_examples2, adv_examples3]
    #adv_examples_name_list = ['c&w', 'AdvGAN', 'FGSM']

    #adv_example = np.load('./data/FGSM_GTSRB_fake_0.1_test.npy')
    #adv_example = np.load('./data/Adv_GAN_GTSRB_fake_test.npy')
    #adv_example = np.load('./data/c&w_attack_GTSRB_test_128.npy')
    #adv_example = np.load('./data/c&w_attack_GTSRB_train_128.npy')
    #eval_Map(test_loader, adv_example)

    #
    #eval_Map(test_loader, adv_example)
    #AdvMap=torch.load('./model/GTSRB_AdvMap_FGSM.pkl')
    #AdvMap = torch.load('./model/GTSRB_AdvMap_c&w_120.pkl')
    #AdvMap = torch.load('./model/GTSRB_AdvMap_c&w_80.pkl')
    #det=torch.load('./model/GTSRB_Det_AdvMap_c&w_80.pkl')
    
    #########################
    # advdersarial training #
    #########################
    '''
    acc = []
    acc = np.asarray(acc, np.float)
    optim_D = optim.RMSprop(cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_G = optim.RMSprop(AdvMap.parameters(), lr=1e-3, weight_decay=1e-4)
    for i in range(70):
        a=eval_advMap(i, train_loader)
        acc = np.append(acc, i)
        acc = np.append(acc, a)
    acc = np.reshape(acc, (-1, 2))
    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2, color='k', marker='D', label='ori')
    # plt1, = plt.plot(acc[:, 0], acc[:, 2], linewidth=2, color='b', marker='o',label='adv')
    plt.legend(handles=[plt0])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=10, fontweight='bold')
    plt.xlabel('(epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
    #eval_cls(train_loader)
    #eval_adv(f,train_loader,adv_examples)
    '''
    ################
    # train AdvMap #
    ################
    '''
    adv_examples1 = np.load('./data/c&w_attack_GTSRB_train_128.npy')
    #adv_examples2 = np.load('./data/Adv_GAN_GTSRB_fake_train.npy')
    adv_examples3 = np.load('./data/FGSM_GTSRB_fake_0.1_train.npy')
    adv_examples_list = [adv_examples1, adv_examples3]
    adv_examples_name_list = ['c&w', 'FGSM']
    adv_example = np.load('./data/c&w_attack_GTSRB_test_128.npy')
    for j in range(2):
        AdvMap =AE0().cuda()# torch.load('./model/GTSRB_AE0.pkl')
        det=Detector().cuda()
        I_n = 50
        optim_D = optim.RMSprop(det.parameters(), lr=1e-3, weight_decay=1e-4)
        optim_G = optim.RMSprop(AdvMap.parameters(), lr=1e-3, weight_decay=1e-4)
        adv_examples=adv_examples_list[j]
        acc = []
        acc = np.asarray(acc, np.float)
        for i in range(I_n):
            a = train_main(i)
            b = eval_Map(test_loader, adv_example)
            acc = np.append(acc, i)
            acc = np.append(acc, a)
            acc = np.append(acc, b)
        acc = np.reshape(acc, (-1, 3))
        np.save('./data/GTSRB_with_'+adv_examples_name_list[j]+'_50.npy', acc)
        torch.save(AdvMap, './model/GTSRB_AdvMap_with_'+adv_examples_name_list[j]+'_50.pkl')
        torch.save(det, './model/GTSRB_Det_AdvMap_with'+adv_examples_name_list[j]+'_50.pkl')
    '''
    I_n = 30
    det = Detector().cuda()
    AdvMap = AE0().cuda()
    optim_D = optim.RMSprop(det.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_G = optim.RMSprop(AdvMap.parameters(), lr=1e-3, weight_decay=1e-4)
    adv_examples = np.load('./data/FGSM_GTSRB_fake_0.1_train.npy')
    acc = []
    acc = np.asarray(acc, np.float)
    for j in range(I_n):
        a = train_main(j)
        b = train_main1(j)
        acc = np.append(acc, j)
        acc = np.append(acc, a)
        #acc = np.append(acc, b)
        acc = np.reshape(acc, (-1, 3))
    np.save('./data/train_AdvMap_MNIST_FGSM_50.npy', acc)


    #acc0=np.load('./data/GTSRB_without_legmap.npy')
    #acc1=np.load('./data/GTSRB_with_legmap.npy')
    '''
    import matplotlib.pyplot as plt
    from pylab import *
    plt0, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2, color='k', marker='D', label='train')
    plt1, = plt.plot(acc[:, 0], acc[:, 2], linewidth=2, color='b', marker='o',label='test')
    plt.legend(handles=[plt0, plt1])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=10, fontweight='bold')
    plt.xlabel( '(epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''


    ##################
    # test framework #
    ##################
    '''
    #adv_example = np.load('./data/Adv_GAN_fake_test.npy')

    # rootpth = './model/AE'
    # model_list = get_model_list(rootpth)
    adv_examples = np.load('./data/FGSM_GTSRB_fake_0.05_test.npy')
    #adv_examples = np.load('./data/Adv_GAN_GTSRB_fake_test.npy')
    #adv_examples = np.load('./data/c&w_attack_GTSRB_test_128.npy')
    model1 = torch.load('./model/MagNet_30_GTSRB_0.pkl').cuda()
    model2 = torch.load('./model/MagNet_30_GTSRB_1.pkl').cuda()
    model_list = [model1, model2]
    thres=np.load('./data/MagNet_GTSRB_batch_thre_l2_train.npy')
    ADA_main(model_list,thres, test_loader, adv_examples)
    '''
    ###############
    #   eval map  #
    ###############

    #rap(adv_examples)

























