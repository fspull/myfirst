import os
import re
import glob
import cv2
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
#import torchsummary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


import segmentation_models_pytorch as smp

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from load_data import *
from run_length_encoding import *
#from load_model import *
from load import *
from loss import *

from collections import OrderedDict
#from sklearn.model_selection import KFold

#gpu setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPUs 0 and 1 to use
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  # Set the GPUs 0 and 1 to use ; multiple GPUs


#get gpu_0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Parameters
ARCHITECTURE = 'DeepLabV3Plus'
ENCODER = 'timm-efficientnet-b1' #timm-regnety_016
ENCODER_WEIGHT= 'noisy-student' #imagenet
N_CLASSES = 1
ACTIVATION = None
OPTIMIZER = 'AdamW'
# 모델 및 백본 변경시 수정 필요.
SAVED_MODEL_PATH = '/root/jupyter/Dacon/deeplabv3p/model_save/'

# Train Parameters
BATCH_SIZE = 128
TRAIN_SET_RATIO = .8
VALID_SET_RATIO = .2
START_EPOCH = 1
NUM_EPOCH = 51
LOSS_PATH = "./loss_history/"


def get_model(model_str: str):
    if model_str == 'Unet':
        return smp.Unet
    elif model_str == 'FPN':
        return smp.FPN
    elif model_str == 'DeepLabV3Plus':
        return smp.DeepLabV3Plus
    elif model_str == 'UnetPlusPlus':
        return smp.UnetPlusPlus
    elif model_str == 'PAN':
        return smp.PAN
    elif model_str == 'MAnet':
        return smp.MAnet
    elif model_str == 'PSPNet':
        return smp.PSPNet
    
    
def get_optimizer(optimizer_str: str):
    if optimizer_str == 'SGD':
        optimizer = torch.optim.SGD
    elif optimizer_str == 'Adam':
        optimizer = torch.optim.Adam
    elif optimizer_str == 'AdamW':
        optimizer = torch.optim.AdamW
    else:
        optimizer = None

    return optimizer

def extract_numbers(string):
    numbers = re.findall(r'\d+', string)
    numbers_list = np.array([int(number) for number in numbers])
    #print(numbers_list)
    return np.max(numbers_list)



# 가장 최근에 저장된 모델의 weight를 가져옵니다.
def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoint_files:
        #raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        return 0, None

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    last_epoch = extract_numbers(latest_checkpoint)
    
    return last_epoch, latest_checkpoint


def get_weight(path): 
    '''
    path : model weight가 저장된 디렉토리
    '''
    last_epoch, last_ckpt_path = get_latest_checkpoint(path)
    if last_epoch == 0:
        return False
    else:
        last_ckpt = torch.load(last_ckpt_path)
        return last_epoch, last_ckpt



def get_multi_gpu_weight(path):
    check_point = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in check_point.items():
        new_state_dict[k[7:]] = v
    return new_state_dict



def get_transform_for_train(version:int=3):
    
    transform_1 = A.Compose(
        [
            A.Normalize(),
            A.OneOf(
                [
                    A.CoarseDropout(50,50,50,15),
                    A.RandomGridShuffle(grid=(2,2))
                ], p = 0.4),
            A.OneOf(
                [
                    # 블러처리
                    A.ColorJitter(),
                    A.ChannelShuffle()             
                ], p = 0.4),
            A.OneOf(
                [
                    #내가 쓰고자하는 모델에서 input size가 어떤게 적합한가
                    A.Resize (224,224),
                    A.RandomCrop(224,224)
                ],p = 1),
            #A.Resize (224,224),
            ToTensorV2()
        ]
    )
    transform_2 = A.Compose(
        [   
            A.Resize(224, 224),
            A.MinMaxNormalize(),
            A.OneOf([
                A.Flip(p=1),
                A.Rotate(p=1),
            ],p = 1),
            ToTensorV2()
        ]
    )
    # for train_20849_2.csv
    transform_3 = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    if version == 1:
        return transform_1
    elif version == 2:
        return transform_2
    elif version == 3:
        return transform_3


def get_transform_for_test():
    transform = A.Compose(
        [   
            A.MinMaxNormalize(),
            ToTensorV2()
        ]
    )
    return transform


def get_dataset(csv_path, transform):
    return SatelliteDataset(csv_file = csv_path, transform = transform)


def random_split_train_valid(dataset):
    data_size = len(dataset)
    train_size = int(data_size*TRAIN_SET_RATIO)
    valid_size = data_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset


def get_data_loader(dataset, is_Train = True):
    return DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = is_Train)
                                                
    
#load weight
is_weight = get_weight(SAVED_MODEL_PATH)
if is_weight == False:
    print('there is no saved model')
    model = get_model(ARCHITECTURE)
    model = model(classes=N_CLASSES,
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHT,
                activation=ACTIVATION)
    
    model = nn.DataParallel(model) # multiple GPUs
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
    
    model = nn.DataParallel(model) # multiple GPUs
    model.load_state_dict(last_ckpt, strict=False)
    model.to(DEVICE)
    START_EPOCH = last_epoch+1

    
# Transform
transform_train = get_transform_for_train(version=3)


# Dataset and DataLoader
dataset = get_dataset(csv_path ='/root/jupyter/Dacon/deeplabv3p/train_img_auged_150K.csv', transform = transform_train)

train_dataset, validation_dataset = random_split_train_valid(dataset)

train_dataloader = get_data_loader(train_dataset, is_Train = True)
validation_dataloader = get_data_loader(validation_dataset, is_Train = True)

                                                 
# OPTIMIZER 
optimizer = get_optimizer(OPTIMIZER)
optimizer = optimizer(model.parameters(),lr=0.0001, weight_decay=5.0e-02)


# LOSS
bceLoss = torch.nn.BCEWithLogitsLoss()
dice_score = DiceScore()
#iouloss = IoULoss()

#object detection sota in google

# LOSSES & SCORES 
pasted_dice_score = [] 
pasted_total_loss = []
pasted_bce_loss = []
pasted_dice_loss = []

#1. augmentation -> secondary memory -> load 
for epoch in range(START_EPOCH, START_EPOCH+NUM_EPOCH):
    
    train_dataset, validation_dataset = random_split_train_valid(dataset)
    train_dataloader = get_data_loader(train_dataset, is_Train = True)
    validation_dataloader = get_data_loader(validation_dataset, is_Train = True)
        
    model.train()
    
    epoch_loss = 0
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_score = 0
    
    for imgs, msks in tqdm(train_dataloader):
        imgs = imgs.to(device=DEVICE, dtype = torch.float)
        msks = msks.to(device=DEVICE, dtype = torch.float)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        bceloss = bceLoss(outputs, msks.unsqueeze(1))
        diceloss = 1 - dice_score(outputs, msks) # 1 - dicescore
        
        # weight of loss
        w1 = 1
        w2 = 1
        loss = (w1*bceloss) + (w2*diceloss)
        
        loss.mean().backward()
        optimizer.step()
        
        epoch_bce_loss += bceloss
        epoch_dice_loss += diceloss
        epoch_loss += loss.item()
        
    pasted_bce_loss.append(epoch_bce_loss/len(train_dataloader))
    pasted_dice_loss.append(epoch_dice_loss/len(train_dataloader))
    pasted_total_loss.append(epoch_loss/len(train_dataloader))
    
    with torch.no_grad():
        model.eval()
        result = []
        for imgs,msks in tqdm(validation_dataloader):
            imgs = imgs.to(device=DEVICE, dtype = torch.float)
            msks = msks.to(device=DEVICE, dtype = torch.float)
            outputs = model(imgs)
            
            dc_sc = dice_score(outputs,msks)
            epoch_score += dc_sc.item()
    
    print(f'Epoch {epoch}')
    print(f'BCE Loss: {epoch_bce_loss/len(train_dataloader)}')
    print(f'DICE Loss: {epoch_dice_loss/len(train_dataloader)}')
    print(f'Total Train Loss: {epoch_loss/len(train_dataloader)}')
    print(f'Validation Dice Score: {epoch_score/len(validation_dataloader)}')
    
    pasted_dice_score.append(epoch_score/len(validation_dataloader))
    
    
    # save a weight every epoch
    path = './model_save/weight_epoch-{num:0004d}.pth'.format(num=epoch)
    torch.save(model.state_dict(), path)
    
    '''
    # save model every 10 epochs
    if epoch % 10 == 0:
    
    # save a epoch weight
        path = './model_save/weight_epoch-{num:0004d}.pth'.format(num=epoch)
        torch.save(model.state_dict(), path)
    '''
    
    
# save epoch losses as .csv
loss_n_score = [pasted_total_loss, pasted_bce_loss, pasted_dice_loss, pasted_dice_score]
loss_df = pd.DataFrame(loss_n_score)
loss_df = loss_df.transpose()
# !!!!파일명 변경 필요!!!!
loss_df.to_csv(path_or_buf=LOSS_PATH + '15k_loss_1.csv' , 
               index=False,
               header=['train_total_loss','train_bce_loss','train_dice_loss','val_Dice'])



'''

transform_test = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize_Z(), #Z-Normalization scailing
        ToTensorV2()
    ]
)




test_dataset = SatelliteDataset(csv_file='/root/jupyter/Dacon/data/test.csv', transform=transform_test, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)




with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        
        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > masks.mean()).astype(np.uint8) # Threshold = 0.35
        
        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)
            
submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./submit.csv', index=False)
'''