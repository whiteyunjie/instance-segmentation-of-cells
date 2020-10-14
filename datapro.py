import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
#from torchsummary import summary
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

#图片路径
train_list = sorted([os.path.join('dataset1/train/',img) for img in os.listdir('dataset1/train/')])
trainGT_list = sorted([os.path.join('dataset1/train_GT/SEG/',img) for img in os.listdir('dataset1/train_GT/SEG/')])
imgsize = 628
#???
data_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip()
        #transforms.Pad(92,padding_mode='symmetric'),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
datalabel_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
#numpy形式
def loadpro_img(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name,-1)#读取时不做改变，默认参数1加载彩色图片
        img = img.astype(np.uint8)
        #img = Image.fromarray(img)
        images.append(img)
    return np.array(images)

#image形式
def load_img(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name,-1)#读取时不做改变，默认参数1加载彩色图片
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        images.append(img)
    return images

#gt图片转成onehot编码(多类别变成2类)
def gt2onehot(trainGT_data):
    trainGTbin = np.zeros(trainGT_data.shape)
    for i in range(len(trainGT_data)):
        gt = trainGT_data[i].reshape((-1,))
        index = np.argwhere(gt > 0)
        #trainGTbin[i][index] = 1 
        gtbin = trainGTbin[i].reshape((-1,))
        gtbin[index] = 1
        trainGTbin[i] = gtbin.reshape((imgsize,imgsize))
    mask = np.zeros((len(trainGTbin),2,imgsize,imgsize))
    for i in range(len(trainGTbin)):
        enc = OneHotEncoder(categories='auto')
        a=enc.fit_transform(trainGTbin[i].reshape((-1,1)))
        label=a.toarray()
        label1=label[:,0].reshape((imgsize,imgsize))
        label2=label[:,1].reshape((imgsize,imgsize))
        mask[i] = np.array([label1,label2])
    return mask
#图片pad
def imgpad(train_data):
    data = np.zeros((len(train_data),1,imgsize+92*2,imgsize+92*2))
    for i in range(len(train_data)):
        data[i] = np.pad(train_data[i],((92,92)),'symmetric')
    return data

class myDataset(Dataset):
    def __init__(self,imgs,labels,gt,transform_x=None,transform_y=None):
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.imgs = imgs
        self.labels = labels
        self.gt = gt
    def __getitem__(self,index):
        img = self.imgs[index]
        img = img/255
        img = (img-0.5)/0.5
        #img = self.transform_x(img)
        label = self.labels[index]
        #label = self.transform_y(label)
        return torch.from_numpy(img),torch.from_numpy(label),torch.from_numpy(self.gt[index])
    def __len__(self):
        return len(self.imgs)

def generatedataset():
    train_data = loadpro_img(train_list)
    trainGT_data = loadpro_img(trainGT_list)
    mask = gt2onehot(trainGT_data)
    data = imgpad(train_data)
    train_dataset = myDataset(data,mask,trainGT_data,data_transforms,datalabel_transforms)
    return train_dataset

def loaddata(issplit=False):
    train_data = loadpro_img(train_list)
    trainGT_data = loadpro_img(trainGT_list)
    if issplit:
        train_img,test_img,train_gt,test_gt=train_test_split(train_data,trainGT_data,test_size=20,random_state = 1)
        mask = gt2onehot(train_gt)
        data = imgpad(train_img)
        testdata = imgpad(test_img)
        train_dataset = myDataset(data,mask,train_gt,data_transforms,datalabel_transforms)
        dataloders = DataLoader(train_dataset,batch_size=2,shuffle=True)
        testdata = testdata/255
        testdata = (testdata-0.5)/0.5
        return dataloders,testdata,test_gt
    else:
        mask = gt2onehot(trainGT_data)
        data = imgpad(train_data)
        train_dataset = myDataset(data,mask,trainGT_data,data_transforms,datalabel_transforms)
        dataloders = DataLoader(train_dataset,batch_size=2,shuffle=True)
        return dataloders