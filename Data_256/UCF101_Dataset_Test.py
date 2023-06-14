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
normal_mean = [0.5,0.5, 0.5]
normal_std =[0.5, 0.5, 0.5]
import torchvision
from torchvision import transforms
import torch
from .spatial_transforms import ToTensor
from .occlusion import occlude
        
class UCF101TEST(Dataset):
    def __init__(self,cl =None ,root = '', train=True, fold=1, transform=None, frames_path='',occ_dict={},num_frames =16):
        self.cl = cl
        self.root = root
        self.num_frames = num_frames
        self.frames_path = frames_path
        self.train = train
        self.fold = fold
        self.transform = transform
        self.occlusion = occlude([224,224],occ_dict.get("occlusion_index",7),occ_dict.get("occlusion_size",60),occ_dict.get("motion","random_placement"))
        self.oc_size = occ_dict.get("occlusion_size",60)
        
        self.video_paths, self.targets, self.starts = self.build_paths()
        self.targets = np.array(self.targets)
        #self.save_path = "/home/sh009885/code/ucf101-supervised/ucf101-supervised-main/dataset/UCF101-O"
                                 
        
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, video_label, start = self.video_paths[idx], self.targets[idx], self.starts[idx]
        video = self.get_video(video_path, start,idx)
        sample = {"video":video,
                  "label":video_label,
                  "video_path":video_path.replace(self.frames_path,'')}
        
        return sample#video, video_label, video_path.replace(self.frames_path,'')
    def get_video(self, video_path, start,idx):
        start_frame = start
        video_container = []
        for item in range(start_frame, start_frame + self.num_frames):
        
            image_name = "frame"+str(item+1).zfill(6) + '.jpg'
            #print(video_path.split("/"))
            cl_name = video_path.split("/")[6]
            vid_name = video_path.split("/")[7]
            fpath= video_path.split("/",6)[-1]
            #print(cl_name,vid_name,fpath)
            image_path = os.path.join(video_path, image_name)
            current_image = Image.open(image_path).convert('RGB')#.resize((224,224))
            
            video_container.append(current_image)
        tr_transform = transforms.Compose([ToTensor(1),torchvision.transforms.Lambda(lambda x:x/255.0)])
        if self.transform is not None:
            clip = [self.transform(img) for img in video_container]
        #not_occluded_frame_index = np.random.choice(np.array(range(16)),size =2)
        if self.oc_size >0:
            clip = [self.occlusion.occlude_with_objects(img,epoch) for epoch,img in enumerate(clip)]
        #clip_new = []
       #for epoch,img in enumerate(clip):
        #    if epoch not in not_occluded_frame_index:
         #       clip_new.append(self.occlusion.occlude_with_objects(img,epoch))
          #  else:
           #     clip_new.append(img)
            
            
        clip = [tr_transform(img) for img in clip]
        clip = [transforms.functional.normalize(img,normal_mean,normal_std) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        self.occlusion.test_randomizer()#initialize()#
        
        return clip


    def build_paths(self):
        data_paths = []
        targets = []
        startings = []
        occ_size=[]
        occ_choice =[]
        occ_motion = []
        
        if self.train:
            annotation_path = os.path.join(self.root, 'ucfTrainTestlist', f'trainlist0{self.fold}.txt')
        else:
            annotation_path = os.path.join(self.root, 'ucfTrainTestlist', f'testlist0{self.fold}.txt')
        
        class_ind_path = os.path.join(self.root, 'ucfTrainTestlist', 'classInd.txt')
        
        class_mapping = {}
        with open(class_ind_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            for item in data:
                class_mapping[item[1]] = int(item[0])-1
        
        annotation_data = {}
        with open(annotation_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            for item in data:
                label_name = item[0].split('/')[0]
                annotation_data[item[0].replace('.avi','')] = class_mapping[label_name]

        if self.train:
            for key in annotation_data:
                video_path = os.path.join(self.frames_path, key) 
                n_frame = len(os.listdir(video_path))
                start_frames = list(range(0, n_frame, 16))
                if self.cl == int(annotation_data[key]) or self.cl is None:    
                    while (n_frame-1) - start_frames[-1] < 16:
                        start_frames = start_frames[:-1]
                    for item in start_frames:
                        data_paths.append(video_path)
                        targets.append(annotation_data[key])
                        startings.append(item)
                  

        else:
            for key in annotation_data:
                video_path = os.path.join(self.frames_path, key) 
                n_frame = len(os.listdir(video_path))
                start_frames = list(range(0, n_frame, self.num_frames))
                video_path = os.path.join(self.frames_path, key)
                n_frame = len(os.listdir(video_path))
                start_frames = list(range(0, n_frame, self.num_frames))
                if self.cl == int(annotation_data[key]) or self.cl is None:    
                
                    while (n_frame-1) - start_frames[-1] <self.num_frames:
                        start_frames = start_frames[:-1]
                    for item in start_frames:
                        data_paths.append(video_path)
                        targets.append(annotation_data[key])
                        startings.append(item)
                    

        return data_paths, targets, startings#,occ_size,occ_choice,occ_motion
    
    def motion(self,h,w,st_h,oh,ow,st_w,epoch):
        return (st_h+epoch*10)%(h-oh),(st_w+epoch*10)%(w-ow)
    def sine_motion(self,h,w,st_h,oh,ow,st_w,epoch):
        return int((st_h+epoch*1)%(h-oh)),int((np.sin(st_h+epoch*1)*112)%(w-ow))

    
    def occlude(self,img,epoch,occ_per = 70.08):
        global occ,st_h,st_w
        _,h,w= img.shape
        oh = int(occ_per*h//100)
        ow = int(occ_per*w//100)
        occ = occ.resize((oh,ow),resample=0)
        if epoch==0:
            st_h = np.random.randint(0,h-oh)
            st_w = np.random.randint(0,w-ow)
        #st_h = 1
        #st_w = 3
        st_new_h,st_new_w = self.motion(h,w,st_h,oh,ow,st_w,epoch)
        
        _tensor =ToTensor(1)
        occ_tensor = _tensor(occ)
        occ_tensor = occ_tensor/255.0
        img[:,st_new_h:st_new_h+oh,st_new_w:st_new_w+ow] = occ_tensor
        return img
        

