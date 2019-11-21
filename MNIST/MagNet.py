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
from train_mnist import classifier
from torchvision.utils import save_image
batch_size=200
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
#adv_example=torch.load('/home/hankeji/Desktop/Adversarial Examples/Cat_Train-FGSM-0.2.pkl')
#f= torch.load('./model/mnist.pkl')


class AE0(nn.Module):
    def __init__(self):
        super(AE0, self).__init__()
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

class AE1(nn.Module):#with dropout layers
    def __init__(self):
        super(AE1, self).__init__()
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
        x=F.dropout(x, p=0.5)
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

class AE2(nn.Module):# BN layers
    def __init__(self):
        super(AE2, self).__init__()
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
        self.norm1=nn.BatchNorm2d(1)
        self.norm2=nn.BatchNorm2d(20)
        self.norm3=nn.BatchNorm2d(10)
        self.norm4=nn.BatchNorm2d(32)
        self.norm5=nn.BatchNorm1d(4608)
        self.norm6=nn.BatchNorm1d(2304)

    def forward(self, x):
        x = self.con1(self.norm1(x))
        x = self.con2(self.norm2(x))
        #x=F.dropout(x, p=0.5)
        x=self.con3(self.norm3(x))
        x=self.con4(self.norm2(x))
        x = x.view(-1, 4608)
        x = self.fc1(self.norm5(x))
        x = self.fc2(self.norm6(x))
        #print (x.size())
        x = x.view(-1, 32, 12, 12)
        x = self.recon1(self.norm4(x))
        x = self.recon2(self.norm2(x))
        x=self.recon3(self.norm3(x))
        x=self.recon4(self.norm2(x))
        #print(x.size())
        return x

class AE3(nn.Module):#change number of channels
    def __init__(self):
        super(AE3, self).__init__()
        self.con1 = nn.Conv2d(1, 16, 5)
        self.con2 = nn.Conv2d(16, 32, 5)
        self.con3=nn.Conv2d(32,16,5)
        self.con4=nn.Conv2d(16,8,5)
        self.fc1 = nn.Linear(1152, 2304)
        self.fc2 = nn.Linear(2304, 1152)

        self.recon1 = nn.ConvTranspose2d(8, 16, 5)
        self.recon2 = nn.ConvTranspose2d(16, 32, 5)
        self.recon3=nn.ConvTranspose2d(32,16,5)
        self.recon4=nn.ConvTranspose2d(16,1, 5)
        self.norm1=nn.BatchNorm2d(1)
        self.norm2=nn.BatchNorm2d(16)
        self.norm3=nn.BatchNorm2d(32)
        self.norm4=nn.BatchNorm2d(8)
        self.norm5=nn.BatchNorm1d(1152)
        self.norm6=nn.BatchNorm1d(2304)

    def forward(self, x):
        x = self.con1(self.norm1(x))
        x = self.con2(self.norm2(x))
        x=F.dropout(x, p=0.5)
        x=self.con3(self.norm3(x))
        x=self.con4(self.norm2(x))
        x = x.view(-1, 1152)
        x = self.fc1(self.norm5(x))
        x = self.fc2(self.norm6(x))
        #print (x.size())
        x = x.view(-1, 8, 12, 12)
        x = self.recon1(self.norm4(x))
        x = self.recon2(self.norm2(x))
        x=self.recon3(self.norm3(x))
        x=self.recon4(self.norm2(x))
        #print(x.size())
        return x

class AE4(nn.Module):#without FC layers
    def __init__(self):
        super(AE4, self).__init__()
        self.con1 = nn.Conv2d(1, 16, 5)
        self.con2 = nn.Conv2d(16, 32, 5)
        self.con3=nn.Conv2d(32,16,5)
        self.con4=nn.Conv2d(16,8,5)
        self.recon1 = nn.ConvTranspose2d(8, 16, 5)
        self.recon2 = nn.ConvTranspose2d(16, 32, 5)
        self.recon3=nn.ConvTranspose2d(32,16,5)
        self.recon4=nn.ConvTranspose2d(16,1, 5)
        self.norm1=nn.BatchNorm2d(1)
        self.norm2=nn.BatchNorm2d(16)
        self.norm3=nn.BatchNorm2d(32)
        self.norm4=nn.BatchNorm2d(8)
        self.norm5=nn.BatchNorm1d(1152)
        self.norm6=nn.BatchNorm1d(2304)

    def forward(self, x):
        x = self.con1(self.norm1(x))
        x = self.con2(self.norm2(x))
        x=F.dropout(x, p=0.5)
        x=self.con3(self.norm3(x))
        x=self.con4(self.norm2(x))
        x = self.recon1(self.norm4(x))
        x = self.recon2(self.norm2(x))
        x=self.recon3(self.norm3(x))
        x=self.recon4(self.norm2(x))
        #print(x.size())
        return x

class AE5(nn.Module):

    def __init__(self):
        super(AE5, self).__init__()
        size = 784
        self.fc1=nn.Linear(size, size*2)
        self.fc2=nn.Linear(size*2, size*2)
        self.fc3=nn.Linear(size*2, size)
        self.norm1=nn.BatchNorm1d(size)
        self.norm2=nn.BatchNorm1d(size*2)
    def forward(self, x):
        x=x.view(x.size()[0],-1)
        x=self.fc1(self.norm1(x))
        x=F.relu(x)
        x=F.dropout(x, p=0.5)
        x=self.fc2(self.norm2(x))
        x=F.relu(x)
        x=self.fc3(self.norm2(x))
        x=x.view(-1, 1, 28, 28)
        #print(x.size())
        return x

def train_ae(epoch, model_name):
    n_auto = 0
    n_ori = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.cuda(),target.cuda()
        data, target=Variable(data, requires_grad=True),Variable(target)
        print('Here training {}: Coming {}_th epcoch: {}_th batch'.format( model_name, epoch, batch_idx))

        en_data=model(data)
        output=f(en_data)
        pred=output.data.max(1)[1]
        n_auto+=pred.eq(target.data).cpu().sum()

        output1 = f(data)
        pred=output1.data.max(1)[1]
        n_ori+=pred.eq(target.data).cpu().sum()


        #loss1=F.nll_loss(output, target)
        loss2=torch.norm(en_data-data, 2)

        loss=loss2#+loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Ori accuracy is {} | Adv accuracy is {}'.\
          format(n_ori/len(train_loader.dataset), n_auto/len(train_loader.dataset)))
    return  n_auto/len(train_loader.dataset)

def get_model_list(rootpth):
    model_list=[]
    for rootdir, dirpth, files in os.walk(rootpth):
       for file in files:
           tmp_pth=os.path.join(rootdir, file)
           tmp_model=torch.load(tmp_pth)
           tmp_model=tmp_model.cpu()#cuda()
           print(tmp_pth)
           model_list.append(tmp_model)
    return model_list


def compare(dif, thre):
    if dif <=thre:
        return 0
    else:
        return 1


def detector(model_list, thresholds, data):
    #threholds = thresholds(model_list, data_leg, data_adv)
    result=0
    for i in range(len(model_list)):
        data_fake=model_list[i](data)
        dif=torch.norm(data-data_fake, 2)
        dif=dif.data.cpu().numpy()
        result+=compare(dif, thresholds[i])
    return result

def reformer(model_list):
    length=len(model_list)
    rand=random.randint(0, length-1)
    return model_list[rand]





def train_aes(model_list, optimi_list,  epoch):
    lens=len(model_list)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        print('Coming {}_th epcoch: {}_th batch'.format(epoch, batch_idx))


        for i in range(lens):

            tmp_output=model_list[i](data)
            if i==0:
                loss=torch.norm(data-tmp_output,2)
                diff0 = tmp_output
            else:
                loss=loss+torch.norm(data-tmp_output,2)
                diff0=diff0+tmp_output

        for i in range(lens):
            if i==0:
                diff=torch.norm(model_list[i](data)-diff0/lens,2)
            else:

                diff+=torch.norm(model_list[i](data)-diff0/lens,2)

        loss_both=loss+diff*0.4
        for i in range(lens):
            optimi_list[i].zero_grad()
        loss_both.backward()
        for i in range(lens):
            optimi_list[i].step()
    return model_list

def test_accuarcy(cw_data):
    start=0
    acc_cw=0
    acc_ori=0
    for batch_idx, (data, target) in enumerate(train_loader):
        size=len(data)
        end=start+size
        tmp_data=cw_data[start:end]
        tmp_data=torch.from_numpy(tmp_data)

        tmp_data, data, target=tmp_data.cuda(), data.cuda(), target.cuda()
        tmp_data=Variable(tmp_data)
        data=Variable(data)
        target=Variable(target)
        tmp_data=ref(tmp_data)


        output_ori=f(data)
        pred_ori=output_ori.data.max(1)[1]
        acc_ori+=pred_ori.eq(target.data).cpu().sum()

        output_cw=f(tmp_data)
        pred_cw=output_cw.data.max(1)[1]
        acc_cw+=pred_cw.eq(target.data).cpu().sum()

        start=end
        print("Here coming {}/{}".format(end,len(train_loader.dataset)))
    print("Ori accuracy is: {}".format(acc_ori/len(train_loader.dataset)))
    print('CW accuracy is: {}'.format(acc_cw/len(train_loader.dataset)))
    return 0


def optimizer_list(model_list):
    lens=len(model_list)
    optim_list=[]
    for i in range(lens):
        optim_list.append(optim.Adam(model_list[i].parameters(),lr=1e-3, weight_decay=1e-4))
    return optim_list


def threshold(model2, train_loader):
    thre=[]
    thre=np.asarray(thre)

    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print('Checking eval_dataset-->{}/{} batch'.format(batch_idx+1, np.ceil(len(train_loader.dataset) / batch_size)))
        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if batch_idx>200:
            break
        tmp_adv =adv_example[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)

        #diff_leg=-torch.norm(model2(data), 2).data.cpu().numpy()+torch.norm(data, 2).data.cpu().numpy()
        diff_leg=torch.norm(model2(data)-data,2).data.cpu().numpy()
        #diff_adv=-torch.norm(model2(tmp_adv),2).data.cpu().numpy()+torch.norm(tmp_adv, 2).data.cpu().numpy()
        #print(diff_adv)
        diff_adv=torch.norm(model2(tmp_adv)-tmp_adv,2).data.cpu().numpy()
        thre=np.append(thre, batch_idx)
        thre=np.append(thre, diff_leg)
        thre=np.append(thre, diff_adv)
    thre=np.reshape(thre,(-1,3))
    print('Var of leg is {} | Var of adv is {}'.format(thre[:,1].var(), thre[:,2].var()))
    return thre#[:,1].max()#+3*np.sqrt(thre[:,1].var())


def det_batch(model, thre, data):
    diff_leg =torch.norm(model(data)-data,2).data.cpu().numpy()
    result=compare(diff_leg, thre)
    return result

def det_batch_models(model_list,thres, data):
    res=0
    for i in range(len(model_list)):
        res+=det_batch(model_list[i], thres[i],data)
    return res

def threshold_kl(cls,model, dataset):
    start=0
    thre = []
    thre = np.asarray(thre)
    for batch_idx, (data, target) in enumerate(dataset):
        print('Checking eval_dataset-->{}/{} batch'.format(batch_idx+1, np.ceil(len(dataset.dataset) / batch_size)))
        end = start + data.size()[0]
        data = data.cuda()
        data = Variable(data)#, Variable(target)

        tmp_adv = adv_example[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)

        ae_output = cls(data) / 40
        ori_output=cls(model(data))/40
        diff_leg=F.kl_div(ae_output, ori_output).data.cpu().numpy()

        ae_output = cls(model(tmp_adv)) / 40
        ori_output = cls(tmp_adv) / 40
        diff_adv = F.kl_div(ae_output, ori_output).data.cpu().numpy()

        #print(type(batch_idx))
        thre = np.append(thre, batch_idx)
        thre = np.append(thre, diff_leg)
        thre = np.append(thre, diff_adv)
    thre=np.reshape(thre, (-1,3))
    return thre




def main_batch(model_list, thres, dataset, data_adv):
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
        res=det_batch_models(model_list, thres, data)
        if res>0:
            n_adv+=data.size()[0]
        else:
            tmp_model=reformer(model_list)
            data=tmp_model(tmp_adv)
            output = f(data)
            pre = output.data.max(1)[1]
            n += pre.eq(target.data).cpu().sum()
    acc = n / len(dataset.dataset)
    acc_adv = n_adv / len(dataset.dataset)
    print(acc, acc_adv)
    print('Done!')
    return acc, acc_adv


def gen_recon_adv(ae, adv_data):
    start = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        end = start + data.size()[0]
        if end > 200:
            break
        tmp_adv = adv_data[start:end]
        tmp_adv = torch.from_numpy(tmp_adv).cuda()
        tmp_adv = Variable(tmp_adv)
        start = end
        # data, target=data.cuda(), target.cuda()
        # data, target=Variable(data), Variable(target)
        save_image(tmp_adv.data[:25], 'images/mnist_fgsm.eps', nrow=5, normalize=True)
        deoutput = ae(tmp_adv)
        save_image(deoutput.data[:25], 'images/mnist_fgsm_rec.eps', nrow=5, normalize=True)

if __name__=='__main__':
    #f=f.cuda()

    ################
    # train MagNet #
    ################
    '''
    I=30

    dataset=train_loader
    #data_adv=np.load('./data/c&w_attack.npy')
    #data_adv = np.transpose(data_adv, (0, 3, 2, 1))

    
    #model_list=get_model_list(rootpth)
    model1=torch.load('./model/AE/AE0.pkl')
    model2=torch.load('./model/AE/AE3.pkl')
    model_list=[model1.cuda(), model2.cuda()]
    #model_list=[model_list[0], model_list[4]]
    optim_list=optimizer_list(model_list)
    for i in range(I):
        model_list=train_aes(model_list, optim_list, i)
    for i in range(len(model_list)):
        torch.save(model_list[i], './model/MagNet_30_'+str(i)+'.pkl')

    '''
    '''
    #ref=torch.load('./model/AE/AE0.pkl')
    adv_example = np.load('./adv_examples/c&w_attack_19.npy')
    adv_example=np.load('./data/Adv_GAN_fake_test.npy')
    adv_example = np.reshape(adv_example, (-1, 1, 28, 28))

    #rootpth = './model/AE'
    #model_list = get_model_list(rootpth)
    model1=torch.load('./model/MagNet_30_0.pkl').cuda()
    model2=torch.load('./model/MagNet_30_1.pkl').cuda()
    model_list=[model1,model2]
    #thres=detetct(model_list,2)
    #np.save('./data/MagNet_thres.npy', thres)
    thres=np.load('./data/MagNet_thres.npy')
    #print(thres)
    main(model_list,adv_example,train_loader, f)
    '''
    ###############
    #   Test      #
    ###############
    '''
    thre=[]
    thre=np.asarray(thre)
    model1 = torch.load('./model/MagNet_30_0.pkl').cuda()
    model2 = torch.load('./model/MagNet_30_1.pkl').cuda()
    model_list = [model1, model2]
    adv_example = np.load('./adv_examples/c&w_attack_test_19.npy')
    #for i in range(2):
    #    tmp=threshold(model_list[i], test_loader)
    #    thre=np.append(thre,tmp)
    #print(thre)
    #adv_examples = np.load('./adv_examples/c&w_attack_test_19.npy')
    # print(adv_examples.shape)
    # adv_examples = np.load('./data/Adv_GAN_fake_test.npy')
    # print(adv_examples.shape)
    # adv_examples = np.load('./data/FGSM_fake_0.2_test.npy')
    #np.save('./data/MagNet_batch_thres_test.npy',thre)

    thres=np.load('./data/MagNet_batch_thres_test.npy')
    main_batch(model_list, thres, test_loader, adv_example)
    '''
    '''
    #adv_example = np.load('./data/c&w_attack.npy')
    #adv_example = adv_example.reshape(-1, 1, 28, 28)
    adv_example = np.load('./data/FGSM_fake_0.3_test.npy')
    model1=torch.load('./model/detector_rec.pkl')
    acc0=threshold(model1, test_loader)
    #acc0=threshold(model2)
    #print(acc0.shape, acc1.shape)
    #acc=np.hstack((acc1,acc0))
    #print(acc.shape)
    np.save('./data/MNIST_REC_FGSM.npy', acc0)
    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(acc0[:, 0], acc0[:, 1], linewidth=2, color='k', marker='D', label='leg')
    plt1, = plt.plot(acc0[:, 0], acc0[:, 2], linewidth=2, color='b', marker='o', label='adv')
    plt.legend(handles=[plt0, plt1])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=10, fontweight='bold')
    plt.xlabel('batch number', fontsize=15)
    plt.ylabel('$l_{2}$ norm of the difference between input and output', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''
    ######################
    #   Train Detector   #
    ######################
    '''
    I = 30
    model = torch.load('./model/AE/AE0.pkl')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    acc = []
    acc = np.asarray(acc)
    #adv_examples = np.load('./data/c&w_attack_test.npy')
    #adv_examples = adv_examples.reshape(-1, 1, 28, 28)
    for i in range(I):
        tmpacc = train_ae(i, 'AE0')
        acc = np.append(acc, i)
        acc = np.append(acc, tmpacc)
    acc = np.reshape(acc, (-1, 2))
    torch.save(model, './model/detector_rec.pkl')
    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2, color='k', marker='D', label='acc')
    # plt1, = plt.plot(acc0[:, 0], acc0[:, 2], linewidth=2, color='b', marker='o', label='adv')
    plt.legend(handles=[plt0])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=10, fontweight='bold')
    plt.xlabel('batch number', fontsize=15)
    plt.ylabel('accuracy', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''


    acc1 = np.load('./data/MNIST_REC_FGSM.npy')

    acc3 = np.load('./data/MNIST_REC_c&w.npy')


    acc1 = acc1[:20]

    acc3 = acc3[:20]
    acc1[:, 1] = acc1[:, 1] / acc1[:, 1].max()
    acc1[:, 2] = acc1[:, 2] / acc1[:, 2].max() * 1.3
    acc3[:, 1] = acc3[:, 1]/ acc3[:, 1].max()*0.9
    acc3[:, 2] = acc3[:, 2] / acc3[:, 2].max() * 1.2

    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(acc1[:, 0], acc1[:, 1], linewidth=2, color='r', linestyle='-', label='MNIST_Leg')
    plt1, = plt.plot(acc1[:, 0], acc1[:, 2], linewidth=2, color='b', linestyle='--', label='MNIST_FGSM')
    #plt2, = plt.plot(acc3[:, 0], acc3[:, 1], linewidth=2, color='r', linestyle='-', label='MNIST_CW_Leg')
    plt3, = plt.plot(acc3[:, 0], acc3[:, 2], linewidth=2, color='g', linestyle=':', label='MNIST_CW')
    plt.legend(handles=[plt0, plt1, plt3], loc=5)
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=20, fontweight='bold')
    plt.ylim(0,1.5)
    plt.xlabel('Example Index', fontsize=20, fontweight='bold')
    plt.ylabel('Reconstruction Error', fontsize=20, fontweight='bold')
    ax = plt.axes()
    plt.show()
    '''
    adv_example = np.load('./data/c&w_attack.npy')
    adv_example = adv_example.reshape(-1, 1, 28, 28)
    #adv_example = np.load('./data/FGSM_fake_0.3_test.npy')
    print(adv_example.shape)
    ae = torch.load('./model/detector_rec.pkl')
    gen_recon_adv(ae, adv_example)
    '''


