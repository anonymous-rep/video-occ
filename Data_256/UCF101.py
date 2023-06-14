import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
import os
from .spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from .UCF101_Dataset_Train import UCF101TRAIN
from .UCF101_Dataset_Test import UCF101TEST
import torch
import torchvision
from .custom_dataset import CustomTest
from torchvision.transforms import Resize
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_ucf101(cl =None,root='Data', frames_path='',occ_dict={},num_frames = 16):
    ## augmentations
    crop_scales = [1.0]
    for _ in range(1, 5):
        crop_scales.append(crop_scales[-1] * 0.53089641525) ##smallest scale is 0.5

    transform_train = Compose([
           # Scale(136),
           # transforms.ColorJitter(brightness = .3,contrast=.3,saturation=.3,hue=.3),
           # transforms.RandomErasing(p=.2),
            MultiScaleRandomCrop(crop_scales, 224),
           # transforms.RandAugment(9,0.5),
           # torchvision.transforms.Lambda(lambda x:x/255.0)
           # RandomHorizontalFlip(),
           #ToTensor(1),
        ])
    
    transform_val = transforms.Compose([
            #Scale(136),
            #esize(128),
            CenterCrop(224),
            #ToTensor(1),
            #torchvision.transforms.Lambda(lambda x:x/255.0)
            #transforms.Normalize(mean=normal_mean, std=normal_std)
        ])
    print(cl)
    train_dataset = UCF101TRAIN (cl =cl,root=root, train=True, fold=1, transform=transform_train, frames_path=frames_path)
    test_dataset = UCF101TEST(cl = cl,root=root, train=False, fold=1, transform=transform_val, frames_path=frames_path,occ_dict=occ_dict,num_frames = num_frames)

    return train_dataset, test_dataset

def custom(cl =None,root='Data', frames_path='',occ_dict={}):
    transform_val = transforms.Compose([
            #Scale(136),
            CenterCrop(224),
            #ToTensor(1),
            #torchvision.transforms.Lambda(lambda x:x/255.0)
            #transforms.Normalize(mean=normal_mean, std=normal_std)
        ])
    test_dataset = CustomTest(root="./dataset/dataset", train=False, fold=1,transform=transform_val)
    return test_dataset

    
if __name__ == "__main__":
    get_ucf101()
