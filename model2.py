import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class convnet(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(convnet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(outchannels,outchannels,3),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class myUnet(nn.Module):
    def __init__(self):
        super(myUnet,self).__init__()
        #down_sample
        self.convnet1 = convnet(1,64)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.convnet2 = convnet(64,128)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.convnet3 = convnet(128,256)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.convnet4 = convnet(256,512)
        self.maxpool4 = nn.MaxPool2d(2)
        
        self.convnet5 = convnet(512,1024)
        
        #up_sample
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.convnet6 = convnet(1024,512)
        
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.convnet7 = convnet(512,256)
        
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.convnet8 = convnet(256,128)
        
        self.up4 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.convnet9 = convnet(128,64)
        
        self.conv10 = nn.Conv2d(64,2,1)
    def forward(self,x):
        x1 = self.convnet1(x)
        x2 = self.convnet2(self.maxpool1(x1))
        x3 = self.convnet3(self.maxpool2(x2))
        x4 = self.convnet4(self.maxpool3(x3))
        
        x5 = self.convnet5(self.maxpool4(x4))
        
        xup1 = self.up1(x5)
        pad = (x4.size(2)-xup1.size(2))//2
        xcrop1 = x4[:,:,pad:pad+xup1.size(2),pad:pad+xup1.size(2)]
        xcat1 = torch.cat([xcrop1,xup1],1)
        x6 = self.convnet6(xcat1)
        
        xup2 = self.up2(x6)
        pad = (x3.size(2)-xup2.size(2))//2
        xcrop2 = x3[:,:,pad:pad+xup2.size(2),pad:pad+xup2.size(2)]
        xcat2 = torch.cat([xcrop2,xup2],1)
        x7 = self.convnet7(xcat2)
        
        xup3 = self.up3(x7)
        pad = (x2.size(2)-xup3.size(2))//2
        xcrop3 = x2[:,:,pad:pad+xup3.size(2),pad:pad+xup3.size(2)]
        xcat3 = torch.cat([xcrop3,xup3],1)
        x8 = self.convnet8(xcat3)
        
        xup4 = self.up4(x8)
        pad = (x1.size(2)-xup4.size(2))//2
        xcrop4 = x1[:,:,pad:pad+xup4.size(2),pad:pad+xup4.size(2)]
        xcat4 = torch.cat([xcrop4,xup4],1)
        x9 = self.convnet9(xcat4)
        
        xf = self.conv10(x9)
        return xf