'''
*** 변경사항 ***
SAVED_MODEL_PATH -> model_save_{ARCHITECTURE}_{ENCODER}/ 로 자동 지정 및 자동 생성.
SEED -> architecture, encoder등의 변경 사항이 있을 경우 변경 필요.
        모든 파라미터가 동일한 경우에 대해 이어서 학습 할 경우 동일한 SEED사용.
get_transform_for_train(), get_transform_for_test() -> load.py
get_dataset(), get_dataloader(),random_split_train_valid() -> load.train_valid_seed()로 변경.

*** 주의 사항 ***
모델이 같아도, 다른 paramters 변경 할 경우 이전 모델 학습 이어서 불가.
SAVED_MODEL_PATH 따로 지정 해주어야 함.
아래 쪽에 csv파일 저장되는 부분이랑 model weight저장되는 부분 한번 더 확인하고 실행.

*** 사용 방법 ***
해당 파일을 copy and paste.
파일 명 변경 -> train_ARCHITECTURE_ENCODER.py
model params들을 변경해준다.
SEED 변경해준다.
 

'''

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



#gpu setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  # Set the GPUs 0 and 1 to use


#get gpu DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 구성 별로 지정 필요.
SEED = 18 

# Model Parameters
ARCHITECTURE = 'UnetPlusPlus' # UnetPlusPlus, DeepLabV3, DeepLabV3Plus
ENCODER = 'timm-efficientnet-b7' # resnet152
ENCODER_WEIGHT= 'noisy-student' # 1: imagenet
N_CLASSES = 1
ACTIVATION = None
OPTIMIZER = 'AdamW'
SAVED_MODEL_PATH = '/root/jupyter/Dacon/deeplabv3p/model_save_{}_{}_3/'.format(ARCHITECTURE,ENCODER)

# Train Parameters
TRAIN_DATA_CSV = './train_28002.csv'
BATCH_SIZE = 48 # 2GPUs Maximum
VALID_SET_RATIO = .1
START_EPOCH = 1 # 고정 
NUM_EPOCH = 30
LOSS_PATH = "./loss_history/" # 고정
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 5.0e-02

# ASL loss
GAMMA_NEG=3
GAMMA_POS=1

# Others
INF = float('inf') # 고정
tol = 1e-6 


# 모델 저장 경로(SAVED_MODEL_PATH) 확인 및 디렉토리 생성
if not os.path.exists(SAVED_MODEL_PATH):
    print('create directory {}'.format(SAVED_MODEL_PATH))
    os.mkdir(SAVED_MODEL_PATH)
    


#load weight
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

    
                                                 
# OPTIMIZER 
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


# LOSS
#bceLoss = torch.nn.BCEWithLogitsLoss()
#dice_loss = DiceLoss(mode='binary')
#iouloss = IoULoss()
asl_loss = AsymmetricLoss(gamma_neg=GAMMA_NEG, gamma_pos=GAMMA_POS, clip=0.05, disable_torch_grad_focal_loss=True)
#lovasz
tversky_loss = TverskyLoss(mode='binary',alpha=0.7, beta=0.3)



# LISTs of Train Losses and Valid Score
pasted_epoch_score = [INF] 
pasted_val_loss = []
pasted_tversky_loss = []
pasted_asl_loss = []
pasted_loss = []


for epoch in range(START_EPOCH, START_EPOCH+NUM_EPOCH):
    
    model.train()

    epoch_asl_loss = 0
    epoch_tversky_loss = 0
    epoch_train_loss = 0
    
    for imgs, msks in tqdm(train_dataloader):
        imgs = imgs.to(device=DEVICE, dtype = torch.float)
        msks = msks.to(device=DEVICE, dtype = torch.float)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        
        #diceloss = dice_loss(outputs, msks.unsqueeze(1))
        tverskyloss = tversky_loss(outputs, msks.unsqueeze(1))
        aslloss = asl_loss(outputs, msks.unsqueeze(1))
        
        loss = aslloss + tverskyloss
        
        loss.mean().backward()
        optimizer.step()
        
        epoch_asl_loss += aslloss.item()
        epoch_tversky_loss += tverskyloss.item()
        epoch_train_loss += loss.item()
        
    pasted_asl_loss.append(epoch_asl_loss/len(train_dataloader))
    pasted_tversky_loss.append(epoch_tversky_loss/len(train_dataloader))
    pasted_loss.append(epoch_train_loss/len(train_dataloader))
    
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
    print(f'Asymmetric Loss: {epoch_asl_loss/len(train_dataloader)}')
    print(f'Tversky Loss: {epoch_tversky_loss/len(train_dataloader)}')
    print(f'Total Loss: {epoch_train_loss/len(train_dataloader)}')
    
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
loss_n_score = [pasted_asl_loss, pasted_Tversky_loss, pasted_loss, pasted_val_loss, pasted_epoch_score[1:]]
loss_df = pd.DataFrame(loss_n_score)
loss_df = loss_df.transpose()
# !!!!파일명 변경 필요!!!!
loss_df.to_csv(path_or_buf=LOSS_PATH + '{}-{}_3'.format(ARCHITECTURE, ENCODER)+'.csv', 
               index=False,
               header=['train ASLoss','train TverskyLoss','train TotalLoss','val ASLoss','val_DiceScore'])

