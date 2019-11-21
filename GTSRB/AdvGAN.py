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
from MagNet import AE0
from torchvision.utils import save_image
from search_networks import model0

#f=torch.load('./model/GTSRB_submodels_30_0.pkl')
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




class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv3 = nn.Conv2d(20, 10, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 1)
        self.norm1 = nn.BatchNorm2d(3)
        self.norm2 = nn.BatchNorm2d(10)
        self.norm3 = nn.BatchNorm2d(20)
        self.norm4 = nn.BatchNorm1d(500)
        self.norm5 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.norm1(x)), 2))
        # print (x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.norm2(x))), 2))
        #print(x.size())
        x = x.view(-1, 500)
        x = self.fc1(self.norm4(x))
        x = self.fc2(self.norm5(x))
        return x





def generate_fake(data, grad):
    return torch.clamp(data + (grad), 0,1)


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


def train_AdvGAN(epoch, i):
    #epoch = epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        print('Here coming {}_th epoch: {}_ th batch'.format(epoch, batch_idx))
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        #print
        #print(torch.norm(data, 2))
        grad = G_(data)
        data_fake = generate_fake(data, grad)
        # data_fake=Variable(data_fake.data, requires_grad=True)
        #print(torch.norm(data,2))
        logits_fake = D_(data_fake)
        #print(torch.norm(data, 2))

        if batch_idx % 30 == 0:
            imgf = data_fake.data.cpu().numpy()[0]
            #imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = data.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)
            #imgr=imgr/imgr.max()
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        save_image(data_fake.data[:25], 'images/%d.png' % epoch, nrow=5, normalize=True)

        #det_loss = D_loss(logits_real, logits_fake)
        loss_D = torch.mean(logits_fake)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        for p in D_.parameters():
            p.data.clamp_(-1, 1)

        gen_imgs=G_(data)
        data_fake0=generate_fake(gen_imgs, data)
        c_output = f(data_fake0)
        c_output = F.softmax(c_output)
        #a = torch.norm(gen_imgs - data, 2).data.cpu().numpy()
        gen_loss = -torch.mean(D_(data_fake0)) + 1e-10*torch.norm(data_fake0 - data, 2)

        const = (torch.norm(gen_imgs, 2)-i)
        print(torch.norm(gen_imgs,2).data.cpu().numpy())
        Hinge_loss = torch.max(Variable(torch.zeros((1)).cuda()), const)
        loss = gen_loss+Hinge_loss -100*F.cross_entropy(c_output, target)
        print(loss.data.cpu().numpy())
        # loss1=F.cross_entropy(c_output, (target+1)%10)

        optim_G.zero_grad()
        loss.backward()
        optim_G.step()


    return G_, D_

def AdvGAN_G(G_):
    for batch_idx, (data, target) in enumerate(test_loader):
        print('Here coming  {}_ th batch'.format(batch_idx))
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)

        grad = G_(data)
        data_fake = generate_fake(data, grad)

        '''
        if batch_idx % 30 == 0:
            imgf = data_fake.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = data.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        '''
        data_fake = data_fake.data.cpu()
        if batch_idx == 0:
            fake = data_fake
        else:
            fake = torch.cat((fake, data_fake), 0)
    fake = fake.numpy()
    np.save('./data/Adv_GAN_GTSRB_fake_train.npy', fake)
    return fake


def FGSM():
    for batch_idx, (data, target) in enumerate(train_loader):
        print('Here coming  {}_ th batch'.format(batch_idx))
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target).view(-1,)
        output = f(data)
        loss=F.nll_loss(output, target)
        loss.backward()
        grad=0.1*torch.sign(data.grad)
        data_fake =data+grad# generate_fake(data, grad)


        if batch_idx % 1 == 0:
            imgf = data_fake.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = data.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        data_fake = data_fake.data.cpu()
        if batch_idx == 0:
            fake = data_fake
        else:
            fake = torch.cat((fake, data_fake), 0)
    fake = fake.numpy()
    np.save('./data/FGSM_GTSRB_fake_0.05_test.npy', fake)
    return fake

def b1(G,epoch):
    f.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)

        #output = f(data)
        # test_loss = F.nll_loss(output, target)
        # test_loss.backward()
        # print (data.size(), G(data).size())
        data_adv = generate_fake(data, G(data))

        output = f(data_adv)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        # if batch_idx%10==0:
        #    tmp_image=data.data.cpu().numpy()[0]
        #    tmp_image=np.reshape(tmp_image, (28, 28))
        #    label=target.data.cpu().numpy()[0]
        # cv2.imshow('Fake label is: '+str(pred.cpu().numpy()[0])+'  Ground-truth label is:'+str(label), tmp_image)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        correct += pred.eq(target.data).cpu().sum()
    print ('Test accuray is {}'.format( correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def test_det(model):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.cuda(),target.cuda()
        data, target=Variable(data, requires_grad=True),Variable(target)

        loss=model(data)
        if batch_idx>0:
            print('done!')
            break

def eval_adv(f,dataset,adv_examples):
    n=0
    start=0
    for batch_idx, (data, target) in enumerate(dataset):
        data, target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        end=start+data.size()[0]
        print('Here coming {}/{}'.format(end,len(dataset.dataset)))

        adv_data=adv_examples[start:end]
        adv_data=torch.from_numpy(adv_data)
        adv_data=adv_data.cuda()
        adv_data=Variable(adv_data)

        if batch_idx % 1== 0:
            imgf = adv_data.data.cpu().numpy()[0]
            imgf = imgf / np.max(imgf)
            imgf = imgf.transpose(1,2,0)

            imgr = data.data.cpu().numpy()[0]
            imgr =imgr.transpose(1,2,0)

            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        output=f(adv_data)
        pred=output.data.max(1)[1]
        n+=pred.eq(target.data).cpu().sum()
        start=end
    acc=n/len(dataset.dataset)
    print('Testing accuracy is {}'.format(acc))
    return acc






if __name__ == '__main__':
    #adv_examples=np.load('./data/FGSM_GTSRB_fake_0.05.npy')
    #eval_adv(train_loader,adv_examples)
    '''
    I_train =30
    I_search = 1

    #D_ = torch.load('./model/Detector_vgg16_98.pkl')
    G_ =AE0().cuda()
    D_=Detector().cuda()
    optim_D = optim.RMSprop(D_.parameters(), lr=1e-3, weight_decay=1e-4)
    optim_G = optim.RMSprop(G_.parameters(), lr=1e-3, weight_decay=1e-4)
    acc = []
    acc = np.asarray(acc, np.float)
    for j in range(I_search):
        for i in range(I_train):
            G, D = train_AdvGAN(i, 120)

            #time.sleep(1)
            #G = torch.load('./model/Generator_vgg16.pkl')
            fake=AdvGAN_G(G)
            tmp_acc =  eval_adv(test_loader, fake)
            acc = np.append(acc, i)
            acc = np.append(acc, tmp_acc)
    torch.save(G, './model/Generator_GTSRB_12.pkl')
    torch.save(D, './model/Detector_GTSRB_12.pkl')
    acc = np.reshape(acc, (I_train, 2))

    from pylab import *
    plt0, = plt.plot(acc[:,0], acc[:,1])
    plt.legend(handles=[plt0])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('(epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
    #np.save('./data/GAN_c_0to500_vgg16.npy', acc)
    '''


    '''
    acc=np.load('./data/GAN_c_0to500_vgg16.npy')

    from pylab import *

    plt0, = plt.plot(acc[0][:, 0], acc[0][:, 1], 'b',marker='|', label='100')
    plt1, = plt.plot(acc[1][:, 0], acc[1][:, 1], 'b',marker='1', label='110')
    plt2, = plt.plot(acc[2][:, 0], acc[2][:, 1], 'b',marker='2', label='12')
    plt3, = plt.plot(acc[3][:, 0], acc[3][:, 1], 'g',marker='3', label='130')
    plt4, = plt.plot(acc[4][:, 0], acc[4][:, 1], 'g',marker='4', label='140')
    plt5, = plt.plot(acc[5][:, 0], acc[5][:, 1], 'g', marker='1',label='150')
    plt6, = plt.plot(acc[6][:, 0], acc[6][:, 1], 'g',marker='2', label='160')
    plt7, = plt.plot(acc[7][:, 0], acc[7][:, 1], 'k', marker='3',label='170')
    plt8, = plt.plot(acc[8][:, 0], acc[8][:, 1], 'k',marker='4', label='180')
    plt9, = plt.plot(acc[9][:, 0], acc[9][:, 1], 'k',marker='1', label='190')
    plt10, = plt.plot(acc[10][:, 0], acc[10][:, 1], 'r',marker='2', label='200')
    plt11, = plt.plot(acc[11][:, 0], acc[11][:, 1], 'r', marker='3',label='210')
    plt12, = plt.plot(acc[12][:, 0], acc[12][:, 1], 'r', marker='4',label='220')
    plt13, = plt.plot(acc[13][:, 0], acc[13][:, 1], 'r', marker='1',label='230')
    plt14, = plt.plot(acc[14][:, 0], acc[14][:, 1], 'r',marker='2', label='240')
    plt15, = plt.plot(acc[15][:, 0], acc[15][:, 1], 'y', marker='3',label='250')
    plt16, = plt.plot(acc[16][:, 0], acc[16][:, 1], 'y', marker='4',label='260')
    plt17, = plt.plot(acc[17][:, 0], acc[17][:, 1], 'y',marker='1', label='270')
    plt18, = plt.plot(acc[18][:, 0], acc[18][:, 1], 'y',marker='2', label='280')
    plt19, = plt.plot(acc[19][:, 0], acc[19][:, 1], 'm', marker='3',label='290')
    plt20, = plt.plot(acc[20][:, 0], acc[20][:, 1], 'm', marker='4',label='300')
    plt21, = plt.plot(acc[21][:, 0], acc[21][:, 1], 'm', marker='1',label='310')
    plt.legend(handles=[plt0,plt1,plt2,plt3,plt4,plt5,plt6,plt7,plt8,plt9,plt10,
                        plt11,plt12,plt12,plt14,plt15, plt16,plt17,plt18,plt19,plt20,
                        plt21])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('$\phi$ (epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''
    #torch.save(G, './model/Generator_GTSRB_20.pkl')
    #G_=torch.load( './model/Generator_GTSRB_12.pkl')
    f = torch.load('./model/GTSRB.pkl')
    f.eval()
    #fake=AdvGAN_G(G_)
    FGSM()
    #adv_example=np.load('./data/FGSM_GTSRB_fake_0.1_train.npy')
    #eval_adv(f,train_loader, adv_example)



