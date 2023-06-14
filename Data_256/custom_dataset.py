from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import random
import pickle, h5py
import cv2
import torch
import torch
from torch.autograd import Variable
import json
from skimage.transform import resize
from skimage import img_as_bool
from PIL import Image
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
import torchvision
from torchvision import transforms
import torch
from .spatial_transforms import ToTensor
from .occlusion import occlude
import decord
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Permute
)


        
class CustomTest(Dataset):
    def __init__(self,cl =None ,root = '', train=True, fold=1, transform=None, frames_path='',occ_dict={}):
        self.cl = cl
        self.root = root
        self.frames_path = frames_path
        self.train = train
        self.fold = fold
        self.transform = Compose(
            [
                    Lambda(lambda x: x / 255.0),
                    Permute((3,0,1,2)),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   # Permute((1,0,2,3)),
                    CenterCrop(224),
            ]
    )
    
        self.occlusion = occlude([112,112],occ_dict.get("occlusion_index",7),occ_dict.get("occlusion_size",60),occ_dict.get("motion","random_placement"))
        self.video_paths, self.targets, self.starts = self.build_paths()
        #print(self.targets)
        #self.targets = np.array(self.targets)
        #self.save_path = "/home/sh009885/code/ucf101-supervised/ucf101-supervised-main/dataset/UCF101-O"
                                 
        
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, video_label, start = self.video_paths[idx], self.targets[idx], self.starts[idx]
        video = self.get_video(video_path, start,idx)
        sample = {"video":video,
                  "label":video_label,
                  "video_path":video_path}
        
        return sample#video, video_label, video_path.replace(self.frames_path,'')

    def get_video(self, video_path, start,idx):
        start_frame = start
        decord.bridge.set_bridge("torch")
        clip = decord.VideoReader(video_path,width=256, height=256)[start:start+16]
        #print(clip.shape)
        
        clip = self.transform(clip)
        #tr_transform = transforms.Compose([ToTensor(1),torchvision.transforms.Lambda(lambda x:x/255.0)])
        #if self.transform is not None:
        #    clip = [self.transform(clip[img]) for img in range(len(clip))]
        #clip = [tr_transform(clip[img]) for img in range(len(clip))]
        #clip = [transforms.functional.normalize(img,normal_mean,normal_std) for img in clip]
        #clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        #self.occlusion.test_randomizer()#initialize
        
        return clip


    def build_paths(self):
        data_paths = []
        targets = []
        startings = []
        path_to_cls = [os.path.join(self.root,i) for i in os.listdir(self.root)]
        video_path = []
        for i in path_to_cls:
            vids_per_class = os.listdir(i)
            for j in vids_per_class:
                if ".avi" in j:
                    video_path.append(os.path.join(i,j))
        #print(video_path)
        #for i in video_path:
        #    print(i.split("/")[-2])
        
        #print(bot)
        
        for path in video_path:
                vr = decord.VideoReader(path)
                n_frame = len(vr)
                start_frames = list(range(0, n_frame, 16))
                if self.cl == int(path.split("/")[-2]) or self.cl is None:    
                
                    while (n_frame-1) - start_frames[-1] < 16:
                        start_frames = start_frames[:-1]
                    for item in start_frames:
                        data_paths.append(path)
                        targets.append(int(path.split("/")[-2])-1)
                        startings.append(item)
                    

        return data_paths, targets, startings#,occ_size,occ_choice,occ_motion

              
if __name__ == "__main__":
    d= CustomTest(root="ucf101-supervised/ucf101-supervised-main/dataset/dataset", train=False, fold=1)
