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
#from AdvGAN import Detector

batch_size=35
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

tf=transforms.Compose([\
    transforms.ToPILImage(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
]
)

class AE0(nn.Module):
    def __init__(self):
        super(AE0, self).__init__()
        self.con1 = nn.Conv2d(3, 10, 5)
        self.con2 = nn.Conv2d(10, 20, 5)
        self.con3=nn.Conv2d(20,10,5)
        self.con4=nn.Conv2d(10,5,5)
        self.fc1 = nn.Linear(1280, 2560)
        self.fc2 = nn.Linear(2560, 1280)

        self.recon1 = nn.ConvTranspose2d(5, 10, 5)
        self.recon2 = nn.ConvTranspose2d(10, 20, 5)
        self.recon3=nn.ConvTranspose2d(20,10,5)
        self.recon4=nn.ConvTranspose2d(10,3, 5)

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x=self.con3(x)
        x=self.con4(x)

        x = x.view(-1, 1280)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 5, 16, 16)
        x = self.recon1(x)
        x = self.recon2(x)
        x=self.recon3(x)
        x=self.recon4(x)
        return x

class AE1(nn.Module):#with dropout layers
    def __init__(self):
        super(AE1, self).__init__()
        self.con1 = nn.Conv2d(3, 10, 5)
        self.con2 = nn.Conv2d(10, 20, 5)
        self.con3=nn.Conv2d(20,10,5)
        self.con4=nn.Conv2d(10,5,5)
        self.fc1 = nn.Linear(1280, 2560)
        self.fc2 = nn.Linear(2560, 1280)

        self.recon1 = nn.ConvTranspose2d(5, 10, 5)
        self.recon2 = nn.ConvTranspose2d(10, 20, 5)
        self.recon3=nn.ConvTranspose2d(20,10,5)
        self.recon4=nn.ConvTranspose2d(10,3, 5)

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x=F.dropout(x, p=0.5)
        x=self.con3(x)
        x=self.con4(x)

        x = x.view(-1, 1280)
        x = self.fc1(x)
        x = self.fc2(x)
        #print (x.size())
        x = x.view(-1, 5, 16, 16)
        x = self.recon1(x)
        x = self.recon2(x)
        x=self.recon3(x)
        x=self.recon4(x)
        #print(x.size())
        return x

class AE2(nn.Module):# BN layers
    def __init__(self):
        super(AE2, self).__init__()
        self.con1 = nn.Conv2d(3, 10, 5)
        self.con2 = nn.Conv2d(10, 20, 5)
        self.con3=nn.Conv2d(20,10,5)
        self.con4=nn.Conv2d(10,5,5)
        self.fc1 = nn.Linear(1280, 2560)
        self.fc2 = nn.Linear(2560, 1280)

        self.recon1 = nn.ConvTranspose2d(5, 10, 5)
        self.recon2 = nn.ConvTranspose2d(10, 20, 5)
        self.recon3=nn.ConvTranspose2d(20,10,5)
        self.recon4=nn.ConvTranspose2d(10,3, 5)
        self.bn1=nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(5)
        self.bn5 = nn.BatchNorm1d(1280)
        self.bn6 = nn.BatchNorm1d(2560)

    def forward(self, x):
        x = self.con1(self.bn1(x))
        x = self.con2(self.bn2(x))
        x=self.con3(self.bn3(x))
        x=self.con4(self.bn2(x))
        x=self.bn4(x)
        x = x.view(-1, 1280)
        x = self.fc1(self.bn5(x))
        x = self.fc2(self.bn6(x))
        #print (x.size())
        x = x.view(-1, 5, 16, 16)
        x = self.recon1(self.bn4(x))
        x = self.recon2(self.bn2(x))
        x=self.recon3(self.bn3(x))
        x=self.recon4(self.bn2(x))
        return x
class AE3(nn.Module):#change number of channels
    def __init__(self):
        super(AE3, self).__init__()
        self.con1 = nn.Conv2d(3, 20, 5)
        self.con2 = nn.Conv2d(20, 30, 5)
        self.con3=nn.Conv2d(30,10,5)
        self.con4=nn.Conv2d(10,5,5)
        self.fc1 = nn.Linear(1280, 2560)
        self.fc2 = nn.Linear(2560, 1280)

        self.recon1 = nn.ConvTranspose2d(5, 10, 5)
        self.recon2 = nn.ConvTranspose2d(10, 30, 5)
        self.recon3=nn.ConvTranspose2d(30,20,5)
        self.recon4=nn.ConvTranspose2d(20,3, 5)
        self.bn1=nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(30)
        self.bn4 = nn.BatchNorm2d(10)
        self.bn5 = nn.BatchNorm1d(1280)
        self.bn6 = nn.BatchNorm1d(2560)
        self.bn7=nn.BatchNorm2d(5)
    def forward(self, x):
        x = self.con1(self.bn1(x))
        x = self.con2(self.bn2(x))
        x=self.con3(self.bn3(x))
        x=self.con4(self.bn4(x))

        x = x.view(-1, 1280)
        x = self.fc1(self.bn5(x))
        x = self.fc2(self.bn6(x))
        #print (x.size())
        x = x.view(-1, 5, 16, 16)
        x = self.recon1(self.bn7(x))
        x = self.recon2(self.bn4(x))
        x=self.recon3(self.bn3(x))
        x=self.recon4(self.bn2(x))
        return x

class AE4(nn.Module):# without FC layers
    def __init__(self):
        super(AE4, self).__init__()
        self.con1 = nn.Conv2d(3, 10, 5)
        self.con2 = nn.Conv2d(10, 20, 5)
        self.con3=nn.Conv2d(20,10,5)
        self.con4=nn.Conv2d(10,5,5)

        self.recon1 = nn.ConvTranspose2d(5, 10, 5)
        self.recon2 = nn.ConvTranspose2d(10, 20, 5)
        self.recon3=nn.ConvTranspose2d(20,10,5)
        self.recon4=nn.ConvTranspose2d(10,3, 5)
        self.bn1=nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(5)


    def forward(self, x):
        x = self.con1(self.bn1(x))
        x = self.con2(self.bn2(x))
        x=self.con3(self.bn3(x))
        x=self.con4(self.bn2(x))
        x=self.bn4(x)
        x = self.recon1(x)
        x = self.recon2(self.bn2(x))
        x=self.recon3(self.bn3(x))
        x=self.recon4(self.bn2(x))
        return x
class AE5(nn.Module):
    def __init__(self):
        super(AE5, self).__init__()

        self.fc1=nn.Linear(3072, 1536)
        self.fc2=nn.Linear(1536, 3072)
        #self.fc3=nn.Linear(size*2, size)
        self.norm1=nn.BatchNorm1d(3072)
        self.norm2=nn.BatchNorm1d(1536)
    def forward(self, x):
        x=x.view(-1,3072)
        x=self.fc1(self.norm1(x))
        x=F.relu(x)
        x=F.dropout(x, p=0.5)
        x=self.fc2(self.norm2(x))

        x=x.view(-1, 3, 32, 32)

        return x


class pv_decoder(nn.Module):
    def __init__(self):
        super(pv_decoder, self).__init__()
        self.decon1 = nn.ConvTranspose2d(20, 20, 7)
        self.decon2 = nn.ConvTranspose2d(20, 20, 5)
        self.decon3 = nn.ConvTranspose2d(20, 10, 5)
        self.decon4 = nn.ConvTranspose2d(10, 20, 7)
        self.decon5 = nn.ConvTranspose2d(20, 10, 6)
        self.decon6 = nn.ConvTranspose2d(10, 3, 3)

        self.fc1=nn.Linear(43, 512)
        self.fc2=nn.Linear(512, 3610)
        self.fc3=nn.Linear(3610, 3610)
        self.fc4=nn.Linear(3610, 3610)
    def forward(self, x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x = F.relu(x)
        x=self.fc3(x)
        x = F.relu(x)
        x=self.fc4(x)
        x=x.view(-1,10,19,19)
        x=self.decon4(x)
        x=self.decon5(x)
        x=self.decon6(x)

        return x


def train_ae(epoch,model_name):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target=data.cuda(),target.cuda()
        data, target=Variable(data, requires_grad=True),Variable(target)
        print('Here training {}: Coming {}_th epcoch | {}_th batch'.format(model_name, epoch, batch_idx))
        # target=target.view(-1,1)
        # label=target.data.cpu().numpy()
        # label=np.asarray(label, np.float32)
        # label=torch.from_numpy(label)
        # label=label.repeat(1,43)
        # #print(label.size())
        # label=Variable(label.cuda(), requires_grad=True)

        #output=cls(data)/1e5
        #output=F.log_softmax(output,dim=1)
        #print(output.size())
        #output=Variable(output.data, requires_grad=True)

        output=model(data)

        if batch_idx%15==0:
            img_ori = data[0].data.cpu().numpy()
            img_ori = np.transpose(img_ori, (1, 2, 0))
            img_rec = output[0].data.cpu().numpy()
            img_rec = np.transpose(img_rec, (1, 2, 0))
            img = np.hstack((img_ori, img_rec))
            cv2.imshow('kk',img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        # l=len(data)
        # data=data.view(-1)
        # output=output.view(-1)
        loss=(1-lf(output,data))#torch.norm(data-output,2)#+

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return 0

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
    if dif.data.cpu().numpy() >thre[0]:#or dif.data.cpu().numpy()<thre[0]
        return 1
    else:
        return 0
def compare1(dif, thre):
    if dif.data.cpu().numpy() >thre:
        return 1
    else:
        return 0

def detector(model_list, thresholds, data):
    threhold = threshold(model_list, test_loader)
    result=0
    for i in range(len(model_list)):
        data_fake=model_list[i](data)
        dif=torch.norm(data-data_fake, 2)
        dif=dif.data.cpu().numpy()
        result=compare(dif, thresholds)
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


def threshold(model2, dataset):
    thre=[]
    thre=np.asarray(thre)
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        print('Checking eval_dataset-->{}/{} batch'.format(batch_idx+1, np.ceil(len(dataset.dataset) / batch_size)))

        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        tmp_adv =adv_example[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)
        start=end


        a = model2(data)


        b = model2(tmp_adv)
        if batch_idx==300:
             break
        '''
        if batch_idx % 1 == 0:
            imgf = tmp_adv.data.cpu().numpy()[0]

            imgf = imgf.transpose(1, 2, 0)

            imgr =b.data.cpu().numpy()[0]
            imgr = imgr.transpose(1, 2, 0)
            # imgr=imgr/imgr.max()
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        '''
        diff_leg=torch.norm(model2(data)-data,2).data.cpu().numpy()
        diff_adv=torch.norm(model2(tmp_adv)-tmp_adv,2).data.cpu().numpy()
        #nn1=diff_leg-diff_adv
        
        #diff_leg=-torch.norm(a, 2).data.cpu().numpy()+torch.norm((data), 2).data.cpu().numpy()
        #diff_adv=-torch.norm(b, 2).data.cpu().numpy()+torch.norm((tmp_adv), 2).data.cpu().numpy()
        #nn2=diff_leg-diff_ad
        #print(nn2>nn1)
        thre=np.append(thre, batch_idx)
        thre=np.append(thre, diff_leg)
        thre=np.append(thre, diff_adv)
    thre=np.reshape(thre,(-1,3))
    print(thre.shape)
    print('Mean of leg is {}| Mean of adv is {}'.format(thre[:,1].mean(), thre[:,2].mean()))
    return thre#[:,1].max(),thre[:,1].min()#+3*np.sqrt(thre[:,1].var())

def threshold1(model2, dataset):
    thre=[]
    thre=np.asarray(thre)
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        print('Checking eval_dataset-->{}/{} batch'.format(batch_idx+1, np.ceil(len(dataset.dataset) / batch_size)))
        if batch_idx>200:
            break
        if data.size()[0] != batch_size:
            break
        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        tmp_adv =adv_example[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)

        diff_leg=torch.norm(a-data, 2).data.cpu().numpy()
        diff_adv=torch.norm(b-tmp_adv, 2).data.cpu().numpy()
        #diff_leg=-torch.norm(model2(data), 2).data.cpu().numpy()+torch.norm(data, 2).data.cpu().numpy()

        #diff_adv=-torch.norm(model2(tmp_adv),2).data.cpu().numpy()+torch.norm(tmp_adv, 2).data.cpu().numpy()

        thre=np.append(thre, batch_idx)
        thre=np.append(thre, diff_leg)
        thre=np.append(thre, diff_adv)
    thre=np.reshape(thre,(-1,3))
    print('Var of leg is {}| Var of adv is {}'.format(thre[:,1].var(), thre[:,2].var()))
    return thre[:,1].max()#,thre[:,1].min()


def det_batch(model, thre, data):
    diff_leg =torch.norm(model(data)-data, 2).data.cpu().numpy()
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


def main_batch(dataset, data_adv, thres, state='Leg'):
    n = 0
    n_adv = 0
    thres = thres, threshold(model1, dataset)
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        print('Checking eval_dataset-->{}/{} batch'.\
              format(batch_idx, np.ceil(len(dataset.dataset) / batch_size)))
        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        tmp_adv = data_adv[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)
        if state=='Leg':
            dif=torch.norm(data-model1(data), 2)
        else:
            dif = torch.norm(tmp_adv - model1(tmp_adv), 2)
        res=compare(dif, thres)#model_list, thres, data)
        print(res)
        if res>0:
            n_adv+=data.size()[0]
        else:
            output = cls(LegMap(data))
            pre = output.data.max(1)[1]
            n += pre.eq(target.data).cpu().sum()
        start=end
    acc = n / len(dataset.dataset)
    acc_adv = n_adv / len(dataset.dataset)
    print(acc, acc_adv)
    print('Done!')
    return acc, acc_adv


def main_detect(model, thres, dataset, data_adv):
    n_leg = 0
    n_adv = 0
    thres = thres
    start = 0
    for batch_idx, (data, target) in enumerate(dataset):
        if batch_idx == 300:
            break
        print(
            'Checking eval_dataset-->{}/{} batch'.format(batch_idx, np.ceil(len(dataset.dataset) / batch_size)))
        end = start + data.size()[0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output_leg = model(data)
        rec_leg= -torch.norm(output_leg, 2).data.cpu().numpy()+torch.norm(data, 2).data.cpu().numpy()

        
        tmp_adv = data_adv[start:end]
        tmp_adv = torch.from_numpy(tmp_adv)
        tmp_adv = tmp_adv.cuda()
        tmp_adv = Variable(tmp_adv)
        output_adv = model(tmp_adv)
        rec_adv = -torch.norm(output_adv, 2).data.cpu().numpy()+torch.norm(tmp_adv, 2).data.cpu().numpy()
        
        start = end
        
        if rec_leg <= thres[0] and rec_leg >= thres[1]:
            n_leg += 1
        if rec_adv > thres[0] or rec_adv < thres[1]:
            n_adv += 1
    acc_leg, acc_adv = n_leg / 300, n_adv / (300)
    print('Leg acc is {:.4f}| Adv acc is {:.4f}'.format(acc_leg, acc_adv))

def test_ae(model):
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx==3:
            break
        data, target=data.cuda(),target.cuda()
        data, target=Variable(data, requires_grad=True),Variable(target)

        output=model(data)

        if batch_idx % 1 == 0:
            imgf = data.data.cpu().numpy()[0]

            imgf = imgf.transpose(1, 2, 0)

            imgr = output.data.cpu().numpy()[0]
            imgr = imgr.transpose(1, 2, 0)
            # imgr=imgr/imgr.max()
            img = np.hstack((imgf, imgr))
            label = target.data.cpu().numpy()[0]
            cv2.imshow(str(label), img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
    return 0
if __name__=='__main__':

    ############
    # train AEs #
    ############
    '''

    I=30
    model_list=[AE0().cuda(), AE1().cuda(),AE2().cuda(), AE3().cuda(), AE4().cuda(), AE5().cuda()]
    name_list=['AE0','AE1','AE2','AE3','AE4','AE5']
    optim_list=optimizer_list(model_list)
    acc=[]
    acc=np.asarray(acc)

    for i in range(1):
        model=model_list[i]
        optimizer=optim_list[i]
        for j in range(I):
            tmp_acc=train_ae(j,name_list[i])
            acc=np.append(acc, j)
            acc=np.append(acc, tmp_acc)
        torch.save(model,'./model/GTSRB_'+name_list[i]+'.pkl')
    #acc=np.reshape(len(model_list),I,2)
    #np.save('./data/train_ae.npy',acc)
    '''
    ################
    #    Train AE  #
    ################
    '''
    import pytorch_ssim
    lf=pytorch_ssim.SSIM().cuda()

    model=AE0().cuda()
    epoches=80
    optimizer=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    model_name='AE0'
    for i in range(epoches):
        train_ae(i,model_name)
    torch.save(model,'./model/AE0.pkl')
    '''
    
    ################
    # train MagNet #
    ################
    '''
    I=60 
    dataset=train_loader
    #data_adv=np.load('./data/c&w_attack.npy')
    #data_adv = np.transpose(data_adv, (0, 3, 2, 1))
    #model_list=get_model_list(rootpth)
    model1=torch.load('./model/GTSRB_AE0.pkl')
    model2=torch.load('./model/GTSRB_AE1.pkl')
    model_list=[model1.cuda(), model2.cuda()]
    model_list=[model_list[0], model_list[1]]
    optim_list=optimizer_list(model_list)
    for i in range(I):
        model_list=train_aes(model_list, optim_list, i)
    for i in range(len(model_list)):
        torch.save(model_list[i], './model/MagNet_30_GTSRB0_'+str(i)+'.pkl')

    #ref=torch.load('./model/AE/AE0.pkl')
    #adv_example = np.load('./adv_examples/c&w_attack_19.npy')
    #adv_example = np.load('./data/c&w_attack_GTSRB_test_128.npy')
    #adv_example = np.load('./data/Adv_GAN_GTSRB_fake_test.npy')
    #adv_example = np.load('./data/FGSM_GTSRB_fake_0.1_test.npy')

    #model1=torch.load('./model/MagNet_30_GTSRB_0.pkl').cuda()
    #model2=torch.load('./model/MagNet_30_GTSRB_1.pkl').cuda()
    #model1=torch.load('./model/GTSRB_AE0.pkl')
    #acc=threshold(model1, test_loader)
    #np.save('./data/GTSRB_det_acc_FGSM_test1.npy', acc)
    


    #model_list=[model1]
    #thre = []
    #thre = np.asarray(thre)
    #for i in range(1):
        tmp = threshold(model_list[i], train_loader)
        thre = np.append(thre, tmp)
    np.save('./data/MagNet_GTSRB_batch_thre_l2_train.npy', thre)

    thre=np.load('./data/MagNet_GTSRB_batch_thre_l2_test.npy')

    main_batch(test_loader, adv_example, thre)
    '''
    ###############
    #   Test      #
    ###############
    '''
    model = torch.load('./model/AE0.pkl')
    adv_example = np.load('./data/c&w_attack_GTSRB_test.npy')
    # adv_example = np.load('./data/FGSM_GTSRB_fake_0.1_test.npy')
    adv_example = np.load('./data/c&w_attack_GTSRB_test.npy')
    thres = np.load('./data/GTSRB_fgsm_train.npy')
    thres=[thres[:,1].max()-0.5, thres[:,1].min()]
    main_detect(model, thres,test_loader, adv_example)
    '''


    model=torch.load('./model/AE0.pkl')
    #adv_example=np.load('./data/c&w_attack_GTSRB_test.npy')
    adv_example = np.load('./data/FGSM_GTSRB_fake_0.1_test.npy')
    thre=threshold(model2=model,dataset=test_loader)
    
    np.save('./data/GTSRB_fgsm_test.npy',thre)

    from pylab import *

    plt0, = plt.plot(thre[:, 0], thre[:, 1], linewidth=2, color='r', linestyle='-', label='GTSRB_Leg')
    plt1, = plt.plot(thre[:, 0], thre[:, 2], linewidth=2, color='b', linestyle='--', label='GTSRB_FGSM')
    
    plt.legend(handles=[plt0, plt1], loc=5)
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('Example Index', fontsize=15)
    plt.ylabel('Reconstruction Error', fontsize=15)
    ax = plt.axes()
    plt.show()

    ######################
    # Comapre train&test #
    ######################
    '''
    test_data = np.load('./data/GTSRB_fgsm_test.npy')
    train_data = np.load('./data/GTSRB_fgsm_train.npy')
    import matplotlib.pyplot as plt
    from pylab import *

    plt0, = plt.plot(test_data[0:40, 0], test_data[0:40, 1], linewidth=2, color='r', linestyle='-', label='GTSRB_test')
    plt1, = plt.plot(train_data[0:40, 0], train_data[0:40, 1], linewidth=2, color='b', linestyle='--', label='GTSRB_train')

    plt.legend(handles=[plt0, plt1], loc=5)
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('Example Index', fontsize=15)
    plt.ylabel('Reconstruction Error', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''
    
    #thres=np.load('./data/MagNet_GTSRB_batch_thre_l2_train.npy')
    #main_batch(model_list, thres, train_loader, adv_example)
    #adv_example=np.load('./data/FGSM_GTSRB_fake_0.1_train.npy')
    #cls = torch.load('./model/GTSRB_submodels_30_0.pkl')
    #LegMap = torch.load('./model/LegMap_GTSRB_sep1.pkl')
    #adv_example = np.load('./data/c&w_attack_GTSRB_test_128_onebatch.npy')#np.load('./data/c&w_attack_GTSRB.npy')
    #model1=torch.load('./model/MagNet_30_GTSRB0_0.pkl')
    #main_batch(test_loader, adv_example, state='Adv')
    #acc0=threshold_kl(f, model1, test_loader)
    #model2=torch.load('./model/MagNet_30_GTSRB_0.pkl')
    #acc0=threshold(model1, test_loader)
    #print(acc0.shape, acc1.shape)
    #acc=np.hstack((acc1,acc0))
    #print(acc.shape)
    #np.save('./data/GTSRB_det_acc_c&w_train.npy', acc0)
    '''
    from pylab import *

    plt0, = plt.plot(acc0[:, 0], acc0[:, 1], linewidth=2, color='k', marker='D', label='leg')
    plt1, = plt.plot(acc0[:, 0], acc0[:, 2], linewidth=2, color='b', marker='o', label='adv')
    plt.legend(handles=[plt0])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=10, fontweight='bold')
    plt.xlabel('batch number', fontsize=15)
    plt.ylabel('accuracy', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''
    
    '''
    acc0=np.load('./data/GTSRB_det_acc_FGSM_train.npy')
    acc1=np.load('./data/GTSRB_det_acc_FGSM_test1.npy')
    acc2 = np.load('./data/GTSRB_det_acc_c&w_train.npy')
    acc3 = np.load('./data/GTSRB_det_acc_c&w_test1.npy')

    acc0 = acc0[:20]
    acc1 = acc1[:20]
    acc2 = acc2[:20]
    acc3 = acc3[:20]
    #acc1[:, 1] = acc1[:, 1] / acc1[:, 1].max()
    acc1[:, 2] = acc1[:, 2] / acc1[:, 2].max()*16.3
    #acc3[:, 1]=acc3[:, 1]/acc3[:, 1].max()
    #acc3[:, 2] = acc3[:, 2] / acc3[:, 2].max()

    import matplotlib.pyplot as plt
    from pylab import *
    plt0, = plt.plot(acc1[:, 0], acc1[:, 1], linewidth=2, color='r', linestyle='-', label='GTSRB_Leg')
    plt1, = plt.plot(acc1[:, 0], acc1[:, 2], linewidth=2, color='b', linestyle='--', label='GTSRB_FGSM')
    plt3, = plt.plot(acc3[:, 0], acc3[:, 2], linewidth=2, color='g', linestyle=':', label='GTSRB_CW')
    plt.legend(handles=[plt0, plt1, plt3], loc=5)
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.ylim(0,20)
    plt.xlabel( 'Example Index', fontsize=15)
    plt.ylabel('Reconstruction Error', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''














