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
import random
import os
import cv2


class Discrimanitor(nn.Module):
    def __init__(self):
        super(Discrimanitor, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)
        self.norm1 = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(10)
        self.norm3 = nn.BatchNorm2d(20)
        self.norm4 = nn.BatchNorm1d(320)
        self.norm5 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.norm1(x)), 2))
        # print (x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.norm2(x))), 2))
        x = x.view(-1, 320)
        x = self.fc1(self.norm4(x))
        x = self.fc2(self.norm5(x))
        return x

def gen_adv():
    start=0
    start_tmp=0
    blocks=4
    data_sets=train_loader
    for batch_idx, (data, target) in enumerate(train_loader):
        print ('Generate adversarial examples: {}/{}'.format(batch_idx+1, np.ceil(len(train_loader.dataset)/batch_size)))

        end=start_tmp+len(data)
        data=Variable(data,requires_grad=True)
        adv=G(data)+data

        if (batch_idx ) % 4 == 0: #or end == len(train_loader.dataset):

            advs = adv
            datas = data
            if end==len(train_loader.dataset):
                print (start, end)
                torch.save(advs, './data/mnist/mnist_advs_' + str(start) + '_' + str(end) + '.pkl')
                torch.save(datas, './data/mnist/mnist_datas_' + str(start) + '_' + str(end) + '.pkl')
        elif(batch_idx+1) % 4 != 0:
            advs = torch.cat((advs, adv), 0)
            datas = torch.cat((datas, data), 0)
            if end==len(train_loader.dataset):
                print (start, end)
                torch.save(advs, './data/mnist/mnist_advs_' + str(start) + '_' + str(end) + '.pkl')
                torch.save(datas, './data/mnist/mnist_datas_' + str(start) + '_' + str(end) + '.pkl')

        elif (batch_idx+1) % 4 == 0:
            print (start, end)
            advs = torch.cat((advs, adv), 0)
            datas = torch.cat((datas, data), 0)
            torch.save(advs, './data/mnist/mnist_advs_' + str(start) + '_' + str(end) + '.pkl')
            torch.save(datas, './data/mnist/mnist_datas_' + str(start) + '_' + str(end) + '.pkl')
            start = end
            #advs = torch.cat((advs, adv), 0)
            #datas = torch.cat((datas, data), 0)

        start_tmp = end

    return 0

def concat(keywords):
    root='./data/mnist/'
    i=0
    for root, dirs, files in os.walk(root):
        for file in files:
            #print (file, i)
            if keywords in file:
                filename = os.path.join(root, file)
                print (filename, i)
                file_tmp = torch.load(filename)
                if i==0:
                    filefull=file_tmp
                else:
                    filefull=torch.cat((filefull, file_tmp),0)
                i+=1
    torch.save(filefull, './data/mnist/'+keywords+'data.pkl')
    return filefull
def concat_advs_datas():
    advs=torch.load('./data/mnist/mnist_advs_data.pkl')
    datas=torch.load('./data/mnist/mnist_datas_data.pkl')
    print (advs.size(), datas.size())
    data=torch.cat((advs, datas), 0)
    print (data.size())
    torch.save(data, './data/mnist_concat.pkl')
    return data

def mix(datas, advs):
    ones=torch.ones(len(advs), 1, 28)
    zeros=torch.zeros(len(datas), 1, 28)
    Label=torch.cat((zeros, ones),0)
    #Label=Label.numpy()
    Data=torch.cat((datas, advs),0)
    Data=Data.data
    #Data=Data.numpy()
    print(Data.size(), Label.size())
    M_Data=torch.cat((Data, Label), -1)
    M_Data=M_Data.numpy()
    np.random.shuffle(M_Data)
    print M_Data.shape
    Data=M_Data[:, :, :, :-1]
    Label=M_Data[:,:,:, -1][:,:,0]
    print(Data.shape, Label.shape)
    Data = np.asarray(Data, np.float32)
    Label = np.asarray(Label, np.int64)
    np.save('./data/Dis_data.npy',Data)
    np.save('./data/Dis_label.npy',Label)
    return Data, Label


def train_Dis(Data, Label):
    start=0
    n=0
    for i in range (int(np.ceil(len(Data)/batch_size))):
        if start+batch_size<=len(Data):
            end=start+batch_size
        else:
            end =start+len(Data)%batch_size
        if end >80000:
            break
        data=Data[start:end]
        target=Label[start:end]
        data=torch.from_numpy(data)
        target=torch.from_numpy(target)
        data=data.cuda()
        target=target.cuda()
        data=Variable(data, requires_grad=True)
        target=Variable(target)
        print("Start is: {} || End is: {}".format(start, end))
        start=end


        if end%1000==0:
            img=data[0].data.cpu().numpy()
            img=np.reshape(img,(28,28))
            tmp_label=target[0].data.cpu().numpy()
            #print(img.shape, tmp_label)
            cv2.imshow(str(tmp_label),img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        output=model(data)
        pred=output.data.max(1)[1]
        pred=pred.view(-1, 1)
        n+=pred.eq(target.data).cpu().sum()


        optimizer.zero_grad()
        loss=F.cross_entropy(output, target.squeeze())
        loss.backward()
        optimizer.step()
    print(n, end)
    return (n/end)


def shuffle(data, label):
    a = range(len(data))
    random.shuffle(a)
    for i in range(len(data)):
        print('Here coming {}_th data'.format(i))
        if i ==0:
            shuffle_data=data[a[i]]
            shuffle_label=label[a[i]]
        else:
            shuffle_data=np.vstack((shuffle_data, data[a[i]]))
            shuffle_label = np.vstack((shuffle_label,label[a[i]]))
    return shuffle_data, shuffle_label

if __name__=='__main__':
    #data = np.load('./testdata/data_scale.npy')
    #label = np.load('./testdata/labels.npy')
    #data, label = shuffle(data, label)

    #np.save('/home/hankeji/Desktop/papercode/ADDA/GTSRB_ADDA/testdata/data_scale_shuffle.npy', data)
    #np.save('/home/hankeji/Desktop/papercode/ADDA/GTSRB_ADDA/testdata/labels_shuffle.npy', label)
    #a=np.load('/home/hankeji/Desktop/papercode/ADDA/GTSRB_ADDA/traindata/data_scale_shuffle.npy')
    b=np.load('/home/hankeji/Desktop/papercode/ADDA/GTSRB_ADDA/traindata/labels_shuffle.npy')
    b=np.reshape(b,(-1,))
    print(b.shape)
    np.save('/home/hankeji/Desktop/papercode/ADDA/GTSRB_ADDA/traindata/labels_shuffle.npy',b)

    '''
    epoches=1
    acc=[]
    acc=np.asarray(acc, np.float)
    model=Discrimanitor().cuda()
    optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #gen_adv()
    #concat('mnist_datas_')
    #concat_advs_datas()
    #datas=torch.load('./data/mnist/mnist_datas_data.pkl')
    #advs = torch.load('./data/mnist/mnist_advs_data.pkl')

    #Data, Label=mix(datas, advs)
    Data=np.load('./data/Dis_data.npy')
    Label=np.load('./data/Dis_label.npy')
    #data=torch.load('./data/mnist/mnist_datas_data.pkl')
    #print (data.size())

    for i in range(epoches):
        tmpacc=train_Dis(Data, Label)
        print(tmpacc)
        acc=np.append(acc, i)
        acc=np.append(acc, tmpacc)
    acc=np.reshape(acc, (-1, 2))
    torch.save(model, './model/discriminator.pkl')
    from pylab import *
    plt0, =plt.plot(acc[:,0], acc[:, 1], label='acc')
    plt.legend(handles=[plt0])
    leg = plt.gca().get_legend()
    leg_text = leg.get_texts()
    plt.setp(leg_text, fontsize=15, fontweight='bold')
    plt.xlabel('$\phi$ (epoch)', fontsize=15)
    plt.ylabel('accuracy of model', fontsize=15)
    ax = plt.axes()
    plt.show()
    '''




