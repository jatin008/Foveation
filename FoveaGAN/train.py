#import necessary libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import random
import torchvision.utils as vutils
from torch import optim
from feature_extractor import FeatureExtractor
from torch.autograd import Variable
import torchvision
import torch.autograd as autograd
from torch.autograd import Variable
import os
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import rand
import glob


class ResidualBlock(nn.Module):
    '''
        Encoder Block for the Generator of type nn.Module     
    '''
    def __init__(self, in_channels, out_channels):
        
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



class DiscResidualBlock(nn.Module):

    '''
    Residual block for the Discriminator
    '''
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding =1, bias = False) 
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding =1, bias = False)
        
        self.avg_pool = nn.AvgPool3d(2)

        self.batch = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        '''
             Forward Propogation.
        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''
    
        conv1 = self.conv1(x)
        conv1 = F.elu(conv1, inplace = True)
        conv1 = self.batch(conv1)
        
        conv2 = self.conv2(conv1)
        conv2 = F.elu(conv2, inplace = True)
        conv2 = self.batch(conv2)

        x = torch.cat([x, conv2], dim = 1)
        x = self.avg_pool(x)
        
        return x

class Discriminator(nn.Module):
    
    def __init__(self, in_channels = 6, out_channels = 1, sigmoid = True):
        
        super().__init__()
        
        self.sigmoid = sigmoid
        
        #70x70 patchGAN disctiminator
        self.dec1 = DiscResidualBlock(in_channels, 32)
        self.dec2 = DiscResidualBlock(32 + in_channels, 64)
        self.dec3 = DiscResidualBlock(64 + 32 + in_channels, 128)
        self.dec4 = DiscResidualBlock(128 + 96 + in_channels, 128)

        self.conv_dense = nn.Conv3d(128 + 128 + 96 + in_channels, 1, 1, padding = 0, bias = False)

        
    def forward(self,x):
        '''
             Forward Propogation.
        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        
        '''
        
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        dec6 = self.conv_dense(dec4)


        if self.sigmoid:
            dec6 = nn.Sigmoid()(dec6)
            
        return dec6

'''
2nd Discriminator that would learn the fft domain features of video sequence
'''
class FFTDisc(Discriminator):
  def __init__(self, sigmoid = True):

    super().__init__(in_channels = 6, sigmoid = False)
    self.flat = nn.Flatten()
    self.sigmoid = sigmoid

  def forward(self, x):


    x = torch.rfft(x,signal_ndim = 3, onesided = False)

    real = x[:,:,:,:,:,0]
    imag = x[:,:,:,:,:,1]


    x = torch.cat([real, imag], dim = 1)

    x = super().forward(x)

    x = self.flat(x)

    if self.sigmoid:
      x = nn.Sigmoid()(x)

    return x




gt_list = os.listdir('images')
sampled_list = sorted(glob.glob('sampled_images/*'))
ground_truth_list = sorted(glob.glob('images/*'))


sam_lis = []
gt_lis = []
for i in range(len(ground_truth_list)):
  sam_lis.append('sampled_images/frame'+str(i)+'.jpg')
  gt_lis.append('images/frame'+str(i)+'.jpg')


from split_train_val import split_train_val
train_loader, test_loader = split_train_val(0.8, 192, sam_lis[:1500], gt_lis[:1500], 16,4)

gen = Generator()
gen.cuda()

disc = Discriminator(sigmoid = True)
disc.cuda()


fftdisc = FFTDisc(sigmoid = True)
fftdisc.cuda()

gan_criterion = nn.BCELoss().cuda()
perceptual_criterion = nn.MSELoss().cuda()
content_criterion = nn.L1Loss().cuda()

#initialize the optimizers for Generator and Discriminator
optimizerD = optim.Adam(disc.parameters(), lr = 0.0003, betas = (0.5,0.999))
optimizerG = optim.Adam(gen.parameters(), lr = 0.0001, betas = (0.5,0.999))
optimizerD_fft = optim.Adam(fftdisc.parameters(), lr = 0.0003, betas = (0.5,0.999))

#set the feature extractor to evaluation mode as it will be used only to calculate perceptual loss
feature_extractor = FeatureExtractor()
feature_extractor.eval()
feature_extractor.cuda()


epochs = 100
for e in range(0,epochs):
  for i, data in enumerate(train_loader):
    
    sampled_images, ground_truth = data

    #to prevent accumulation of gradients
    optimizerD.zero_grad()
    
    #training discriminator with real images
    real_disc_input = torch.cat([ground_truth, ground_truth], dim = 1)
    real_disc_input = real_disc_input.reshape(1,real_disc_input.shape[1],real_disc_input.shape[0],real_disc_input.shape[2],real_disc_input.shape[3])
    output = disc.forward(real_disc_input)
    
    #print(output.shape)
    target = torch.ones_like(output)*0.9
    #print(target.shape)

    errorD_real = gan_criterion(output,target)
    print('errorD_real = ', errorD_real)


    #training discriminator with fake images
    fake_images = gen.forward(sampled_images).detach()
    fake_disc_input = torch.cat([fake_images, ground_truth], dim = 1)
    fake_disc_input = fake_disc_input.reshape(1,fake_disc_input.shape[1],fake_disc_input.shape[0], fake_disc_input.shape[2],fake_disc_input.shape[3])
    output = disc.forward(fake_disc_input)
    target = torch.zeros_like(output)
    errorD_fake = gan_criterion(output,target)
    print('ErrorD_Fake = ', errorD_fake)


    # Total spatial discriminator error
    errorD = errorD_real + errorD_fake
    errorD.backward()
    optimizerD.step()


    #fft discriminator
    optimizerD_fft.zero_grad()

    #real
    fft_real_disc_input = ground_truth.reshape(1,ground_truth.shape[1],ground_truth.shape[0],ground_truth.shape[2],ground_truth.shape[3])
    output = fftdisc.forward(fft_real_disc_input)
    target = torch.ones_like(output)

    fft_errorD_real = gan_criterion(output, target)

    #fake
    fft_fake_disc_input = fake_images.reshape(1,fake_images.shape[1],fake_images.shape[0], fake_images.shape[2],fake_images.shape[3])
    output = fftdisc.forward(fft_fake_disc_input)
    target = torch.zeros_like(output)
    fft_errorD_fake = gan_criterion(output, target)

    #Total fft Discriminator error
    fft_errorD = fft_errorD_real + fft_errorD_fake
    fft_errorD.backward()
    optimizerD_fft.step()

    print('Total Discriminator Loss =',fft_errorD)


    
    #for Generator
    #gen.zero_grad()
    optimizerG.zero_grad()

    
    #gan loss from spatial discriminator
    fake_images = gen.forward(sampled_images)
    fake_disc_input = torch.cat([fake_images, ground_truth], dim = 1)
    fake_disc_input = fake_disc_input.reshape(1,fake_disc_input.shape[1],fake_disc_input.shape[0], fake_disc_input.shape[2],fake_disc_input.shape[3])    
    
    output = disc.forward(fake_disc_input).detach()
    target = torch.ones_like(output)
    gan_loss_spatial = gan_criterion(output,target)

    print('gan_loss_spatial = ',gan_loss_spatial)


    #gan loss from fft discriminator
    output = fftdisc.forward(fft_fake_disc_input).detach()
    target = torch.ones_like(output)
    gan_loss_fft = gan_criterion(output,target)


    gan_loss = gan_loss_spatial + gan_loss_fft

    print('Gan Loss Gen = ', gan_loss)

    
    #perceptual loss
    gen_feature = feature_extractor(fake_images)
    real_feature = feature_extractor(ground_truth).detach()
    perceptual_loss =  perceptual_criterion(gen_feature,real_feature)
    print('perceptual_loss = ', perceptual_loss)

    
    #content loss
    content_loss = content_criterion(fake_images, ground_truth)
    print('content_loss = ', content_loss)


    #optical loss
    
    
    errorG = gan_loss + 100*perceptual_loss + content_loss
    #errorG =  content_loss + perceptual_loss 

    print('Generator Loss = ', errorG)
    errorG.backward()
    optimizerG.step()
    
    

    #print('Epoch:-{} [{}/{}] Generator Loss :- {:6f}, Discriminator Loss :- {:6f}'.format(e,i,len(train_loader), errorG.item(), errorD.item()))

    print('Epoch:-{} [{}/{}] Generator Loss :- {:6f}'.format(e,i,len(train_loader), errorG.item()))

    print('############################################')
    #save the weights and images every 200th iteration of an epoch
    if i%10 == 0:
        for j, test_data in enumerate(test_loader): #for testing purposer I'm using train_loader instead of test_loader
            gen.eval()

            test_sampled_images, test_ground_truth = test_data

            vutils.save_image(test_ground_truth, 'train_output/'+'real_samples.png', normalize = True)

            fake = gen.forward(test_sampled_images)
            vutils.save_image(fake.data, 'train_output/'+'fake_samples_epoch{}_{}_{}.png'.format(e,i,j), normalize = True)
            
            
            gen.train()


            #save batches
            torch.save({'epochs': e,
                        'gen_state_dict' :gen.state_dict(),
                        'disc_state_dict' :disc.state_dict(),
                        'fft_disc_state_dict' :fftdisc.state_dict(),

                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'optimizerD_fft_dict':optimizerD_fft.state_dict(),
                        'errorG':errorG,
                        'errorD':errorD
                        }, 'checkpoint_small.pth')
          
            
          
            break