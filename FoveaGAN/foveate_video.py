#import necessary libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import random
import torchvision.utils as vutils
import cv2
from torchvision import io
import torchvision
import torchvision.utils as vutils
from PIL import Image
import time
import glob


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        Encoder Block for the Generator of type nn.Module
        '''
        
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(2)
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, 3, padding = 1, groups=in_channels, bias = False),
            nn.Conv2d(in_channels,out_channels,1,1,0,bias = False)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels, 3, padding = 1, groups=out_channels, bias = False),
            nn.Conv2d(out_channels,out_channels,1,1,0,bias = False)
        )


    def forward(self, x):
        '''
            Forward Propogation, Skip connections are introduced between encoder and decoder.
        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''
    
        conv1 = self.conv1(x)
        conv1 = F.elu(conv1, inplace = True)
        
        conv2 = self.conv2(conv1)
        conv2 = F.elu(conv2, inplace = True)
        
        x = torch.cat([x, conv2], dim = 1)
        x = self.avg_pool(x)
        
        return x

class TemporalBlock(nn.Module):
    '''
    Decoder Block for the Generator of type nn.Module
    '''
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        self.hidden = torch.zeros((999))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+out_channels,in_channels+out_channels, 3, padding = 1, groups=in_channels+out_channels, bias = False),
            nn.Conv2d(in_channels+out_channels,out_channels,1,1,0,bias = False)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels, 3, padding = 1, groups=out_channels, bias = False),
            nn.Conv2d(out_channels,out_channels,1,1,0,bias = False)
        )
        
    def forward(self, x):
        '''
            Forward Propogation, Skip connections are introduced between encoder and decoder.
        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''
        
        if self.hidden.shape[0] == 999:        #quick fix
            self.hidden = torch.zeros((x.shape[0],self.out_channels,x.shape[2],x.shape[3])).cuda()
        
        conv1 = torch.cat([self.hidden, x], dim = 1)
        
        conv1 = self.conv1(conv1)
        conv1 = self.batch_norm(conv1)
        conv1 = F.elu(conv1)
        
        self.hidden = conv1.clone().detach()
        
        conv2 = self.conv2(conv1)
        conv2 = F.elu(conv2)
        
        x = torch.cat([x,conv2], dim  = 1)
        x = self.upsample(x)
        
        return x

class Generator(nn.Module):
    '''
    Generator class of type nn.Module
    '''

    def __init__(self):
        
        super().__init__()
        
        self.enc_1 = ResidualBlock(3, 32)             
        self.enc_2 = ResidualBlock(32 + 3, 64)           
        self.enc_3 = ResidualBlock(64 + 35, 128)          

        self.bottleneck = ResidualBlock(128 + 99, 128)    
        
        self.dec_3 = TemporalBlock(128 + 99 + 128 + 227, 128)
        self.dec_2 = TemporalBlock(128 + 99 + 128 + 227 + 128 + 64 + 35, 64)
        self.dec_1 = TemporalBlock(128 + 99 + 128 + 227 + 128 + 64 + 35 + 64 + 35 , 32)
        
        self.conv_dense = nn.Conv2d(940, 3, 1, padding = 0, bias = False)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        '''
            Forward Propogation, Skip connections are introduced between encoder and decoder.
        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''
        
        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)

        bottleneck = self.bottleneck(enc_3)
        bottleneck = self.upsample(bottleneck)

        dec_3 = torch.cat([bottleneck, enc_3], dim = 1)
        dec_3 = self.dec_3(dec_3)
        
        
        dec_2 = torch.cat([dec_3, enc_2], dim = 1)
        dec_2 = self.dec_2(dec_2)
        
        dec_1 = torch.cat([dec_2, enc_1], dim = 1)
        dec_1 = self.dec_1(dec_1)
        
        x = self.conv_dense(dec_1)
        
        return x

gen = Generator()
gen.cuda()

#Initialize the Generator and load the pretrained weights
checkpoints = torch.load('checkpoint_small_t.pth', map_location= torch.device('cuda'))
gen.load_state_dict(checkpoints['gen_state_dict'])


sampled_list = sorted(glob.glob('sampled_images/*'))
ground_truth_list = sorted(glob.glob('images/*'))


sam_lis = []
gt_lis = []
for i in range(len(sampled_list)):
  sam_lis.append('sampled_images/frame'+str(i)+'.jpg.jpg')
  gt_lis.append('images/frame'+str(i)+'.jpg')


from split_train_val import split_train_val
train_loader, test_loader = split_train_val(0.95, 544, sam_lis[:1500], gt_lis[:1500], 1,0)


gen.eval()

count = 0
for j, images in enumerate(train_loader):
    start_time = time.time() # start time of the loop
    images = images[0]
    fake = gen(images)

    fake = fake.detach().cpu()


    for idx in range(fake.shape[0]):


        img = fake[idx]
        min = float(img.min())
        max = float(img.max())
        img.clamp_(min = min, max = max )
        img.add_(-min).div_(max - min + 1e-5)

        img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        #img = fake[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        cv2.imshow('hello',img)
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

        #cv2.imwrite('output/frame{}.jpg'.format(j), img)
        key = cv2.waitKey(1) & 0xFF


        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
