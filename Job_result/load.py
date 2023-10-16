import os
import re
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from run_length_encoding import *
import glob as glob
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split



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


def train_valid_seed(csv_file = None, transform_train = None, 
                     transform_valid = None ,test_size = .1, random_seed = 42, 
                     shuffle = True, batch_size=32):
    
    df = pd.read_csv(csv_file)
    train_df, valid_df = train_test_split(df, test_size = .1, random_state = random_seed, shuffle = shuffle)
    Dataset_train = SatelliteDataset_From_Df(train_df, transform = transform_train, infer = False)
    Dataset_valid = SatelliteDataset_From_Df(valid_df, transform = transform_valid, infer = False)
    
    train_data_loader = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(Dataset_valid, batch_size=batch_size, shuffle=True)
    
    return train_data_loader, valid_data_loader


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
class SatelliteDataset_HE(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        merge_array = cv2.merge([cv2.equalizeHist(y), cr, cb])
        image = cv2.cvtColor(merge_array, cv2.COLOR_YCrCb2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


class SatelliteDataset_From_Df(Dataset):
    def __init__(self, df, transform=None, infer=False):
        self.data = df
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


        
def get_transform_for_train(version:int=4):
    
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
    
    transform_4 = A.Compose(
        [
            #A.Resize(224,224),
            #A.Resize(672,672),
            A.MinMaxNormalize(),
            A.RandomGridShuffle(grid=(2,2), p = 0.6),
            A.RandomFog (fog_coef_lower=0.3, fog_coef_upper=0.7,
                         alpha_coef=0.08, p = 0.6),

            A.OneOf(
                [
                    A.ColorJitter(),
                    A.ChannelShuffle()
                ]
            , p = 0.4),
            A.Downscale (scale_min=0.1, scale_max=0.5,interpolation=0, p=1),
            ToTensorV2()
        ]
    )
    
    
    if version == 1:
        return transform_1
    elif version == 2:
        return transform_2
    elif version == 3:
        return transform_3
    elif version == 4:
        return transform_4
    
def get_transform_for_test():
    transform = A.Compose(
        [   
            A.MinMaxNormalize(),
            ToTensorV2()
        ]
    )
    return transform


def get_weight(path): 
    '''
    path : model weight가 저장된 디렉토리
    '''
    def get_latest_checkpoint(checkpoint_dir):
    
        def extract_numbers(string):
            numbers = re.findall(r'\d+', string)
            numbers_list = np.array([int(number) for number in numbers])
            #print(numbers_list)
            return np.max(numbers_list)

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        if not checkpoint_files:
            #raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            return 0, None

        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        last_epoch = extract_numbers(latest_checkpoint)
    
        return last_epoch, latest_checkpoint
    
    last_epoch, last_ckpt_path = get_latest_checkpoint(path)
    if last_epoch == 0:
        return False
    else:
        last_ckpt = torch.load(last_ckpt_path)
        return last_epoch, last_ckpt
    
