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
from GAN_Arch_details import CycleGAN_Unet_Generator




epnum=["1500"]

for a in epnum:
    print(a)
    G_basestyle = CycleGAN_Unet_Generator()
    checkpoint =torch.load("weights/model_weights_"+str(a)+"_.pth")    
    G_basestyle.load_state_dict(checkpoint)   
    G_basestyle.eval()   
    data_dir = "tesmats/"
    batch_size = 4000  
    dm = TECGDataModule(data_dir, batch_size, phase='test')
    dm.prepare_data()
    dataloader = dm.train_dataloader()
    net = G_basestyle
    net.eval()
    predicted = []
    predicted=pd.DataFrame(data=predicted)
    actual = []
    actual=pd.DataFrame(data=actual)
    with torch.no_grad():
        for base, style,truelabel in (dataloader):

            output = net(base)[0].squeeze()
            outputlabel = net(base)[1].squeeze()
            prelab=outputlabel.argmax(-1).detach().numpy()
            aclab=truelabel.argmax(-1).detach().numpy()
            style=style.detach().numpy()
            style=style.squeeze()           
            truelabel=truelabel.detach().numpy()
            truelabel=truelabel.squeeze()
            ganoutput=output.detach().numpy()
            ganoutput=pd.DataFrame(data=ganoutput)
            predicted=pd.concat([predicted,ganoutput])
            ganacc=base.detach().numpy().squeeze()
            ganacc=pd.DataFrame(data=ganacc)
            actual=pd.concat([actual,ganacc])
     
    ch1array=pd.concat([actual,predicted])
    labelsarray=np.ones((2000,2))
    labelsarray[0:1000,1]=-1*labelsarray[0:1000,1]
    labelsarray[1000:2000,0]=-1*labelsarray[1000:2000,0]
     
    
    ch1array=ch1array.values.reshape(len(ch1array)*4096,1)
    labelarray=labelsarray.reshape(len(labelsarray)*2,1)
    import scipy.io as sio
    sio.savemat('predicted'+str(a)+'.mat', {'predicted':ganoutput.values})    
    sio.savemat('ganacc'+str(a)+'.mat', {'ganacc':ganacc.values}) 
    sio.savemat('actual'+str(a)+'.mat', {'actual':style}) 
    sio.savemat('label'+str(a)+'.mat', {'label':truelabel}) 
    
    from sklearn.metrics import classification_report,confusion_matrix
     
    print(classification_report(aclab,prelab,digits=4))
    print(confusion_matrix(aclab,prelab))   