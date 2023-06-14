import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from Data.UCF101 import get_ucf101
from Data_256.UCF101 import get_ucf101
from utils import AverageMeter, accuracy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import AverageMeter, accuracy
import time
import matplotlib.pyplot as plt
from Data.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import os
import pytorchvideo
import argparse
from timm import create_model
from models.MAE import *
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs)) 
class_specific_performance={}
total_class_performance = {}
for i in range(101):
    class_specific_performance[i] = 0
    total_class_performance[i]=0
occ_indx = {0:"Desktop",
            1:"Aeroplane",
            2:"Desktop",
            3:"Human",
            4:"Cat",
            5:"Cat",
            6:"Plant",
            7:"Human",
            8:"Human",
            9: " ",
            10: " ",
            11: " ",
            12: " ",
            13: "motor "
           }

def test(test_loader,model): 
    print("test")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    predicted_target_not_softmax = {}
    video_class_idx={}
    with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data_time.update(time.time() - end)
                model.eval()
                
                inputs = data["video"].cuda()
                targets = data["label"].cuda()
                video_name = data["video_path"]
                
                #print(targets)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                out_prob = F.softmax(outputs, dim=1)
                out_prob = out_prob.cpu().numpy().tolist()
                targets = targets.cpu().numpy().tolist()
                outputs = outputs.cpu().numpy().tolist()
            
                for iterator in range(len(video_name)):
                    if video_name[iterator] not in predicted_target:
                        predicted_target[video_name[iterator]] = []
                    if video_name[iterator] not in video_class_idx:
                        total_class_performance[int(targets[iterator])]+=1
                
                    if video_name[iterator] not in video_class_idx:
                        video_class_idx[video_name[iterator]] =int(targets[iterator])
                    if video_name[iterator] not in predicted_target_not_softmax:
                        predicted_target_not_softmax[video_name[iterator]] = []

                    if video_name[iterator] not in ground_truth_target:
                        ground_truth_target[video_name[iterator]] = []
                    
                    
                    
                    predicted_target[video_name[iterator]].append(out_prob[iterator])
                    predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                    ground_truth_target[video_name[iterator]].append(targets[iterator])
                
                
                losses.update(loss.item(), inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
            
    for key in predicted_target:
        clip_values = np.array(predicted_target[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target[key] = video_pred
    
    for key in predicted_target_not_softmax:
        clip_values = np.array(predicted_target_not_softmax[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target_not_softmax[key] = video_pred
    
    for key in ground_truth_target:
        clip_values = np.array(ground_truth_target[key]).mean(axis=0)
        ground_truth_target[key] = int(clip_values)

    pred_values = []
    pred_values_not_softmax = []
    target_values = []
    for key in predicted_target:
        if predicted_target[key] == ground_truth_target[key]:
            class_specific_performance[video_class_idx[key]] +=1
        
        
    for key in predicted_target:
        pred_values.append(predicted_target[key])
        pred_values_not_softmax.append(predicted_target_not_softmax[key])
        target_values.append(ground_truth_target[key])
    
    pred_values = np.array(pred_values)
    pred_values_not_softmax = np.array(pred_values_not_softmax)
    target_values = np.array(target_values)
    #return(pred_values,target_values)

    secondary_accuracy = (pred_values == target_values)*1
    secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))*100
    print(f'test accuracy after softmax: {secondary_accuracy}')

    secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
    secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
    print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')
def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_argument('--occ_index',default=1,type = int,help = "Occluder to be used")
    parser.add_argument('--occ_size',default=60,type = int,help = "Area of the image to be occluded")
    parser.add_argument('--motion' ,default ="random_placement", type = str , help = "motion followed by the occluder")
    parser.add_argument('--dataset',default = "UCF",type= str)
    parser.add_argument('--arch',default = 'i3d',type=str)
    parser.add_argument('--data_path',default = '',type=str)
    parser.add_argument('--checkpoint',default = '',type=str)
    
    args = parser.parse_args()
    #print(" aug 60")
    print("config occluder {}, occluder size{}, occluder motion {}".format(occ_indx[args.occ_index],args.occ_size,args.motion))
    
    state_dict = torch.load(args.checkpoint)# 
    num_frames = 16
    
    model = model.cuda()
    #state_dict= torch.load('Data_256/checkpoint.pth')['module']
    #model.load_state_dict(state_dict['state_dict'])
    if args.arch == "mvit":
        model  =  torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4",pretrained = True)
        model.head.proj = torch.nn.Linear(768,101,bias =True)
        model = model.cuda()
    
    if args.arch == "i3d":
        print("i3d")
        model  = torch.hub.load("facebookresearch/pytorchvideo",model="i3d_r50",pretrained=True)#torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4",pretrained = True)
        model.blocks[-1].proj = torch.nn.Linear(2048,101,bias =True)
        num_frames = 8

    if args.arch == "x3d":
        model  = torch.hub.load("facebookresearch/pytorchvideo",model="x3d_m",pretrained=True)#torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4",pretrained = True)
#model.proj = torch.nn.Linear(2304,101,bias =True)
        model.blocks[-1].proj = torch.nn.Linear(2048,101,bias =True)
        num_frames =16

        
    model = model.cuda()
    model.load_state_dict(state_dict['state_dict'])
    
    occ_dict= {"occlusion_index":args.occ_index,"occlusion_size":args.occ_size,"motion":args.motion}
    if args.dataset == "UCF":
        train_dataset, test_dataset =  get_ucf101(root='Data_256',frames_path =args.data_path,occ_dict = occ_dict,num_frames= num_frames)
    else:
        from Data_256.UCF101 import custom

        test_dataset = custom(root = args.data_path)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=8,
        num_workers=4,
        pin_memory=True)
    return test(test_loader,model)
    
    
if __name__ == "__main__":
    main()
    
