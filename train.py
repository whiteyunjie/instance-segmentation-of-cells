import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from model import myUnet
from datapro import loaddata
from evaluation import Jaccard_eval

issplit = False

dataloders = loaddata(issplit) 

model = myUnet().cuda()
epoch_nums = 10
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.99)
loss = nn.CrossEntropyLoss().cuda()
batch_size = 2

losses = []
jses = []
for echo in range(epoch_nums):
    train_loss = 0
    train_js = 0
    n = 0
    for img,label,gt in tqdm(dataloders):
        img = Variable(img.float()).cuda()
        label = Variable(label).cuda()
        out = model(img)
        lossvalue = loss(out,label.float())
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        
        train_loss += float(lossvalue)
        acc = 0
        n += batch_size
        if n >len(dataloders.dataset):
            pred=out[0].argmax(0)
            pred = pred.numpy()
            pred = pred.astype(np.uint8)
            true=gt[0]
            _,score=Jaccard_eval(pred,true)
            train_js += score
        else:
            for j in range(batch_size):
                pred=out[j].argmax(0)
                pred = pred.numpy()
                pred = pred.astype(np.uint8)
                true=gt[j]
                _,score=Jaccard_eval(pred,true)
                train_js += score
        
    jses.append(train_js/len(dataloders.dataset))   
    losses.append(train_loss/len(dataloders.dataset))
    print('echo:'+ ' '+str(echo))
    print('loss:'+ ' '+str(train_loss/len(dataloders.dataset)))
    print('jaccard:'+ ' '+str(train_js/len(dataloders.dataset)))

#torch.save(model.state_dict(),'unet2param.pkl')
