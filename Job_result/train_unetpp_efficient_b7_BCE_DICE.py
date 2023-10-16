import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss, FocalLoss, LovaszLoss

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from run_length_encoding import *
from load import *
from loss import *
from dacon_dice import *

from collections import OrderedDict


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  # Set the GPUs 0 and 1 to use


#get gpu DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 구성 별로 지정 필요.
SEED = 7


ARCHITECTURE = 'DeepLabV3Plus' # UnetPlusPlus, DeepLabV3, DeepLabV3Plus
ENCODER = 'timm-efficientnet-b1' # resnet152
ENCODER_WEIGHT= 'noisy-student' # 1: imagenet
N_CLASSES = 1
ACTIVATION = None
OPTIMIZER = 'AdamW'
SAVED_MODEL_PATH = '/root/jupyter/Dacon/deeplabv3p/model_save_{}_{}_7/'.format(ARCHITECTURE,ENCODER)

# Train Parameters
TRAIN_DATA_CSV = './train_52857.csv'
BATCH_SIZE = 160 # 2GPUs Maximum
VALID_SET_RATIO = .1
START_EPOCH = 1 # 고정 
NUM_EPOCH = 30
LOSS_PATH = "./loss_history/" # 고정
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 5.0e-02

INF = float('inf') # 고정
tol = 1e-5 

if not os.path.exists(SAVED_MODEL_PATH):
    print('create directory {}'.format(SAVED_MODEL_PATH))
    os.mkdir(SAVED_MODEL_PATH)


is_weight = get_weight(SAVED_MODEL_PATH)
if is_weight == False:
    print('there is no saved model')
    model = get_model(ARCHITECTURE)
    model = model(classes=N_CLASSES,
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHT,
                activation=ACTIVATION)
    
    model = nn.DataParallel(model) 
    model.to(DEVICE)
else:
    last_epoch, last_ckpt = is_weight
    print('last epoch is {}'.format(last_epoch))
    print('model-{} loaded..'.format(last_epoch))
    model = get_model(ARCHITECTURE)
    model = model(classes=N_CLASSES,
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHT,
                activation=ACTIVATION)
    
    model = nn.DataParallel(model) 
    model.load_state_dict(last_ckpt, strict=False)
    model.to(DEVICE)
    START_EPOCH = last_epoch+1


optimizer = get_optimizer(OPTIMIZER)
optimizer = optimizer(model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Transform
transform_train = get_transform_for_train(version=4) # version4 fix
transform_valid = get_transform_for_test()

train_dataloader, validation_dataloader = train_valid_seed(csv_file=TRAIN_DATA_CSV,
                                                      transform_train=transform_train,
                                                      transform_valid=transform_valid,
                                                      batch_size=BATCH_SIZE,
                                                      test_size=VALID_SET_RATIO,
                                                     random_seed=SEED,
                                                     shuffle=True)

bceLoss = torch.nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(mode='binary')

pasted_epoch_score = [INF] 
pasted_dice_loss = []
pasted_bce_loss = []
pasted_loss = []
pasted_val_loss = []
pasted_train_score = []


def model_train(model, train_dataloader, valid_dataloader, EPOCHES, losses, loss_weights):
    for epoch in range(1, 1+EPOCHES):
    
        model.train()
        
        
        
        #def 함수로 변환
            
        
    
    for imgs, msks in tqdm(train_dataloader):
        imgs = imgs.to(device=DEVICE, dtype = torch.float)
        msks = msks.to(device=DEVICE, dtype = torch.float)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        
        loss_values = []
        for i, loss_func in enumerate(losses):
            loss_values.append(loss_weights[i] * loss_func(outputs,msks.unsqueeze(1)))
            
        dc_sc = calculate_dice_scores_from_rle(outputs,msks)
            
        
        
        loss.backward()
        optimizer.step()
        
        epoch_bce_loss += bceloss.item()
        epoch_dice_loss += diceloss.item()
        epoch_train_loss += loss.item()
        epoch_train_score += dc_sc
        
    pasted_bce_loss.append(epoch_bce_loss/len(train_dataloader))
    pasted_dice_loss.append(epoch_dice_loss/len(train_dataloader))
    pasted_loss.append(epoch_train_loss/len(train_dataloader))
    pasted_train_score.append(epoch_train_score/len(train_dataloader))
    
    epoch_score = 0
    val_asl_loss = 0
    
    with torch.no_grad():
        model.eval()
        result = []
        for imgs,msks in tqdm(validation_dataloader):
            imgs = imgs.to(device=DEVICE, dtype = torch.float)
            msks = msks.to(device=DEVICE, dtype = torch.float)
            outputs = model(imgs)
            
            dc_sc = calculate_dice_scores_from_rle(outputs,msks)
            epoch_score += dc_sc
            
            
            val_asl = validation_asl(outputs, msks.unsqueeze(1))
            val_asl_loss += val_asl.item()
    
    
    print(f'Epoch {epoch}')
    print(f'BCE Loss: {epoch_bce_loss/len(train_dataloader)}')
    print(f'DICE Loss: {epoch_dice_loss/len(train_dataloader)}')
    print(f'Total Loss: {epoch_train_loss/len(train_dataloader)}')
    print(f'Total Score: {epoch_train_score/len(train_dataloader)}')
    print(f'Validation Asymmetric Loss: {val_asl_loss/len(validation_dataloader)}')
    print(f'Validation Dice Score: {epoch_score/len(validation_dataloader)}')
    
    pasted_epoch_score.append(epoch_score/len(validation_dataloader))
    pasted_val_loss.append(val_asl_loss/len(validation_dataloader))
    
    # save a weight every epoch
    path = SAVED_MODEL_PATH + '{}_{}-{num:0004d}.pth'.format(ARCHITECTURE,ENCODER,num=epoch)
    torch.save(model.state_dict(), path)
    
    if np.abs(pasted_epoch_score[-2] - pasted_epoch_score[-1])< tol:
        print('Early Stop')
        break;
    
# save epoch losses as .csv
loss_n_score = [pasted_bce_loss, pasted_dice_loss, pasted_val_loss, pasted_epoch_score[1:]]
loss_df = pd.DataFrame(loss_n_score)
loss_df = loss_df.transpose()
# !!!!파일명 변경 필요!!!!
loss_df.to_csv(path_or_buf=LOSS_PATH + '{}-{}_7_1'.format(ARCHITECTURE, ENCODER)+'.csv', 
               index=False,
               header=['train_BCELoss','train_DiceLoss', 'pasted_val_Loss','val_DiceScore'])
