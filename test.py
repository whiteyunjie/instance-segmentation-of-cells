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
from datapro import loaddata
from evaluation import Jaccard_eval
from repress import watershed_process,kluster_proess
from datapro import loadpro_img

result_path = 'dataset1/test_RES' #存储路径
model_path = 'unet1params3.pkl'
test_path = 'dataset1/test/'
repressmethod = 'connectedComponents'  #选择后处理方法'watershed','cluster'
imgsize = 628

#加载模型
model = myUnet().cuda()
model.load_state_dict(torch.load(model_path))

test_list = sorted([os.path.join(test_path,img) for img in os.listdir(test_path)])
test_data = loadpro_img(test_list)  
test_datapro = np.zeros((len(test_data),1,imgsize+92*2,imgsize+92*2))
for i in range(len(test_data)):
    test_datapro[i] = np.pad(test_data[i],((92,92)),'symmetric')
    test_datapro[i] = test_datapro[i]/255.0
    test_datapro[i] = (test_datapro[i]-0.5)/0.5
test_datapro = torch.from_numpy(test_datapro)
model.eval()
pred_all=[]
i = 0
for index,img in enumerate(test_datapro):
    img = img.reshape((1,1,imgsize+92*2,imgsize+92*2))
    img = Variable(img.float()).cuda()
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
    imageio.imwrite(os.path.join(result_path, 'mask{:0>3d}.tif'.format(index)),pred_img.astype(np.uint16))
    pred_all.append(pred_img)
    i = i+1
    print('image:{}/33'.format(i))
print('finish')  
    