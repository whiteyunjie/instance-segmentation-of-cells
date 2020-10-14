import imageio
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
from datapro import generatedataset
from evaluation import Jaccard_eval
from repress import watershed_process,kluster_proess

model_path = 'unet1params4.pkl'
repressmethod = 'watershed'  #选择后处理方法'watershed','cluster'
imgsize = 628

train_dataset = generatedataset() 
#加载模型
model = myUnet().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()
pred_all = []
score_all = []
isval = True #true表明用来预测，否则就是验证正确率
i = 0
for img,label,gt in train_dataset:
    img = img.reshape((1,1,imgsize+92*2,imgsize+92*2))
    img = Variable(img.float()).cuda()
    label = Variable(label).cuda()
    orimg = img
    with torch.no_grad():
        testout = model(img)
        testout = torch.sigmoid(testout)
    #testloss = loss(testout,label)
    pred = testout[0].argmax(0).cpu()
    pred = pred.numpy()
    pred = pred.astype(np.uint8)
    if repressmethod == 'connectedComponents':
        maxval,pred_img = cv2.connectedComponents(pred, 4, cv2.CV_32S)
    elif repressmethod == 'watershed':
        pred_img = watershed_process(pred)
    else:
        pred_img = kluster_proess(pred)
    pred_all.append(pred_img)
    if isval:
        true = gt.numpy()
        _,score = Jaccard_eval(pred_img,true)
        score_all.append(score)
        print('image:{}/175, score: {:.4f}'.format(i,score))
    i = i+1
print('final score: %.5f'%np.mean(score_all))   