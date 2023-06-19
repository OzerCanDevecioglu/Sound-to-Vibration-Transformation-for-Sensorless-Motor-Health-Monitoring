import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
from model_details import Generator
import seaborn as sn
from scipy.stats import norm
import scipy.signal as sig
import copy
import scipy.io as sio
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchaudio
from scipy.fft import fft, fftfreq, fftshift
from torch_stoi import NegSTOILoss
import math
from sklearn.metrics import classification_report,confusion_matrix


spectrogram = torchaudio.transforms.Spectrogram(n_fft = 256 ,win_length=256 ,hop_length=128)

data_dir = 'input/gan-getting-started'
batch_size = 8

# DataModule  -----------------------------------------------------------------    
dm = ECGDataModule(data_dir, batch_size, phase='test')
dm.prepare_data()
dataloader = dm.train_dataloader()

G = Generator().cuda()
  
num_epoch = 3000
lr=0.001
betas=(0.5, 0.999)

G_params = list(G.parameters())

optimizer_g = torch.optim.Adam(G_params, lr=lr, betas=betas)

criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
total_loss_g = []
result = {}
    
E=0.0001    
  
for e in range(1,num_epoch):
    print("Epoch: "+str(e))
    G.train()
    total_loss_g = []
    for input_img, real_img,labels in (dataloader): 
      if(0):
          # check beats
          plt.subplot(211)
          plt.plot(input_img[1,0,:].cpu().detach())
          plt.title("Noisy Audio Signal/Clear Audio Signal")
          plt.subplot(212)
          plt.plot(real_img[1,0,:].cpu().detach())
        

      input_img=input_img.cuda()
      real_img=real_img.cuda()
      colabels=labels.cuda()
   
      # Generator 
      fake_img = G(input_img)[0].cuda()
      fake_img_ = fake_img.detach() # commonly using 
      loss_g_mae = criterion_mae(fake_img, real_img) # MSELoss  
      rspre = spectrogram(torch.tensor(fake_img.cpu()))+E
      ispre = spectrogram(torch.tensor(real_img.cpu()))+E
      loss_g_dim = criterion_mae(rspre.log10(), ispre.log10()) # MSELoss
      class_loss=criterion_mse(G(input_img)[1].unsqueeze(1),colabels)
      loss_g = loss_g_mae + 10*class_loss +10*loss_g_dim

      loss_g.backward()
      optimizer_g.step()
      optimizer_g.zero_grad()
      loss_g=np.mean(total_loss_g)
      total_loss_g.append(loss_g)

    if e%10 == 0:
  
      # Sanity Check
        data_dir = "valmats"
        dm2 = TECGDataModule(data_dir, batch_size=100, phase='test')
        dm2.prepare_data()
        dataloader2 = dm2.train_dataloader()
        net = G
        net.eval()
        predicted = []
        predicted=pd.DataFrame(data=predicted)
        actual = []
        actual=pd.DataFrame(data=actual)
        ractual = []
        ractual=pd.DataFrame(data=ractual)
        llabell = []
        llabell=pd.DataFrame(data=llabell)
        cpred = []
        cpred=pd.DataFrame(data=cpred)
    
        cc=[]
        with torch.no_grad():
          for base, style,clabels in (dataloader2):     
              output = net(base.cuda())[0].squeeze().cpu()
              class_loss=criterion_mse(net(base.cuda())[1].unsqueeze(1).cpu(),clabels)
              cc.append(class_loss.cpu().detach().numpy())
              
              ganoutput=output.detach().numpy()
              ganoutput=pd.DataFrame(data=ganoutput)
              predicted=pd.concat([predicted,ganoutput])
             
              ganacc=base.detach().numpy().squeeze()
              reall=style.detach().numpy().squeeze()
              clabels=clabels.detach().numpy().squeeze()
              clabels=pd.DataFrame(data=clabels)
              clpreedd=pd.DataFrame(data=net(base.cuda())[1].squeeze(-1).cpu().detach().numpy())


                
              reall=pd.DataFrame(data=reall)
              ganacc=pd.DataFrame(data=ganacc)
              actual=pd.concat([actual,ganacc])
              ractual=pd.concat([ractual,reall])
              llabell=pd.concat([llabell,clabels])
              cpred=pd.concat([cpred,clpreedd])


        
        b=cpred.values.argmax(1)
        a=llabell.values.argmax(1)
        
        print(classification_report(a,b,digits=4))
        print(confusion_matrix(a,b))   
        
        lossa=np.mean(cc)
        gan_outputs=predicted.values.reshape(len(predicted)*4096,1)
        real_outputs=actual.values.reshape(len(actual)*4096,1)
        ractual=ractual.values.reshape(len(ractual)*4096,1)
      
        gan_outputs=gan_outputs[:len(gan_outputs)]
        real_outputs=real_outputs[:len(real_outputs)]
        ractual=ractual[:len(ractual)]
      
        gan_outputs1=gan_outputs.reshape(int(len(gan_outputs)/4096),4096)
        real_outputs1=real_outputs.reshape(int(len(real_outputs)/4096),4096)
        ractual1=ractual.reshape(int(len(ractual)/4096),4096)

        psnr_list=[]
        for ii in range(len(gan_outputs1)):
       
            psg=np.abs(fftshift(fft(gan_outputs1[ii,:]))) /  np.max(np.abs(fftshift(fft(gan_outputs1[ii,:]))))
            psr=np.abs(fftshift(fft(ractual1[ii,:]))) /  np.max(np.abs(fftshift(fft(ractual1[ii,:]))))
            mse = np.mean((psg - psr) ** 2)
            res=10 * math.log10(1. / mse)
            psnr_list.append(res)
        with open('datamax.txt', 'a') as f:
              f.write("\n"+"PSNR: "+str(np.mean(psnr_list))+"  Class Loss: "+str(lossa))
             
        total_loss_g.append(loss_g.item())
        from random import randrange
        
        plt.figure()
        plt.subplot(221)
        plt.plot(real_outputs1[0,:])
        plt.grid()
        plt.title("Input")
        
        plt.subplot(222)
        plt.plot(gan_outputs1[0,:]) 
        plt.grid()
        plt.title("Output")
        
        
        
        plt.subplot(223)
        plt.plot(real_outputs1[800,:])
        plt.grid()
        plt.title("Input")
        
        plt.subplot(224)
        plt.plot(gan_outputs1[800,:]) 
        plt.grid()
        plt.title("Output")
        
        
        
        
        
        
        plt.savefig("figs/"+str(e)+"loss_"+str(lossa)+".png")
        plt.close()
        torch.save(G.state_dict(), 'weights/model_weights_'+str(e)+'_.pth')

        
        
        
   