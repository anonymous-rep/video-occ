import torch
import torch.nn as nn
import numpy as np
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from PIL import Image
import cv2 as cv
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Code.vMFMM import *
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
import torch
from torch.utils.data import DataLoader
import cv2
import os
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, update_clutter_model
from Code.config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures
from Code.config import config as cfg
from torch.utils.data import DataLoader
from Code.losses import ClusterLoss
from utils import AverageMeter, accuracy
import torch.nn.functional as F
from new_model import Classification_model
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from Kinetics.dataset import labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop
)
from Kineticsrun.MVIT import MViT
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
def load_config(path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    return cfg

class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = load_config("Kineticsrun/Mvitv2_kinetics.yaml")
        cfg = assert_and_infer_cfg(cfg)
        model = MViT(cfg).cuda()
        model.load_state_dict(torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth")["model_state"])


        self.feature_model = model
        #self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        #state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_model([x])
        return x,[8,7,7]


class aClassification_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_model()
        self.drop = nn.Dropout(.3)
        self.pool = nn.AvgPool2d(7)
        self.ln  = nn.Linear(768,400)
        #self.occ = nn.Linear(768,1)
    def forward(self,x):
        x,thw = self.feature_extractor(x)
        out = x[:,1:,:]
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        cls_token = x[:,0,:]
        cls_token = cls_token.reshape(-1,1,768,1,1)
        out = out+cls_token
        out = torch.mean(out,dim=1)
        #out = self.pool(out)
        #out = out.reshape(-1,768)
        #x,_ = torch.max(torch.max(x,dim =-1)[0],dim = -1)
        #x = torch.mean(x,dim=1)
        #out = self.drop(out)
        #cls_score = self.ln(out)
        #occ_lik = self.occ(out)
        return out#,occ_lik

#class_specific_performance={}
#total_class_performance = {}
#for i in range(101):
#    class_specific_performance[i] = 0
#    total_class_performance[i]=0
class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw


occ_indx = {0:"Desktop",
            1:"Aeroplane",
            2:"Car",
            3:"Human",
            4:"Motorcycle",
            5:"Cat",
            6:"Plant",
            7:"Human",
            8:"Human",
            9: " ",
            10: " ",
            11: " ",
            12: " ",
            13: " "
           }

class feature_mvit(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = Classification_model(101)
        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        self.feature_model = self.feature_model.cuda()
        #state_dict = torch.load("results/Finern_Augmented_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        state_dict = torch.load("results/NEWER_AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        
        self.feature_model.load_state_dict(state_dict['state_dict'])
        
        #state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x,train = False):
        x,thw = self.feature_model.feature_extractor(x)
        
        #x = self.feature_model.cls_positional_encoding(x)
        #thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        #for blck in self.feature_model.blocks:
         #   x,thw= blck(x,thw)
        out = x[:,1:,:]
        cls_token = x[:,0,:]
        cls_token = cls_token.reshape(-1,1,768,1,1)
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        out = out+cls_token
        out = torch.mean(out,dim=1)
        ##out_avg,_ = torch.max(out,dim =1)
        return out


def test(test_loader,model): 
    classification_loss = nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    predicted_target_not_softmax = {}
 #   video_class_idx={}
    with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data_time.update(time.time() - end)
                model.eval()

                inputs = data["video"].cuda()
                targets = data["label"].cuda()
                video_name = data["video_path"]
                outputs,_,_ = model(inputs)
                label = targets.detach().cpu().numpy()
                    #outputs = torch.max(outputs,0,keepdims = True)[0]
                out = outputs.argmax(1)
                    
                    
                loss = classification_loss(outputs, targets)/ outputs.shape[0]
                    
                out_prob = F.softmax(outputs, dim=1)
                    
                out_prob = out_prob.cpu().numpy().tolist()
                targets = targets.cpu().numpy().tolist()
                outputs = outputs.cpu().numpy().tolist()
            
                for iterator in range(len(video_name)):
  #                  if video_name[iterator] not in video_class_idx:
  #                      total_class_performance[int(targets[iterator])]+=1
                
                    if video_name[iterator] not in predicted_target:
                        predicted_target[video_name[iterator]] = []
                
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
    #for key in predicted_target:
    #    if predicted_target[key] == ground_truth_target[key]:
    #        class_specific_performance[video_class_idx[key]] +=1
    

    for key in predicted_target:
        pred_values.append(predicted_target[key])
        pred_values_not_softmax.append(predicted_target_not_softmax[key])
        target_values.append(ground_truth_target[key])
    
    pred_values = np.array(pred_values)
    pred_values_not_softmax = np.array(pred_values_not_softmax)
    target_values = np.array(target_values)
    #return pred_values,target_values

    secondary_accuracy = (pred_values == target_values)*1
    secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))*100
    print(f'test accuracy after softmax: {secondary_accuracy}')

    secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
    secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
    print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')
    #for i in class_specific_performance:
    #    total_class_performance[i] = class_specific_performance[i]/total_class_performance[i] 
    #print(class_specific_performance)
    #print(total_class_performance)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_argument('--occ_index',default=1,type = int,help = "Occluder to be used")
    parser.add_argument('--occ_size',default=60,type = int,help = "Area of the image to be occluded")
    parser.add_argument('--motion' ,default ="random_placement", type = str , help = "motion followed by the occluder")
    parser.add_argument('--dataset',default = "UCF",type = str)
    parser.add_argument('--mix_model',default ="",type=str)
    parser.add_argument('--dict_path',default ="",type=str)
    parser.add_argument('--checkpoint',default ="",type=str)
    parser.add_argument('--data_dir',default ="",type=str)
    
    args = parser.parse_args()
    occ_dict= {"occlusion_index":args.occ_index,"occlusion_size":args.occ_size,"motion":args.motion}
    
    print("config occluder {}, occluder size{}, occluder motion {}".format(occ_indx[args.occ_index],args.occ_size,args.motion))
    alpha = 3  # vc-loss
    beta = 3 # mix loss
    likely = 0.5 # occlusion likelihood
    lr = 1e-4 # learning rate
    batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
    # Training setup
    vc_flag = True # train the vMF kernels
    mix_flag = True # train mixture components
    ncoord_it = 100 	#number of epochs to train

    bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
    bool_load_pretrained_model = False
    bool_train_with_occluders = False

    extractor = feature_mvit()#Classification_model()
    extractor.cuda()
    dict_dir = dict_dir+'dictionary_{}_{}.pickle'.format(args.dict_path,"768")
    
    weights = getVmfKernels(dict_dir, device_ids)
    
    occ_likely =[]
    for i in range(101): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(.6)

    if args.dataset == "Kinetics":
        num_classes = 400
    else:
        num_classes = 101
    mix_models = getCompositionModel(device_ids,args.mix_model,layer,categories_train,compnet_type=compnet_type,num_classes=num_classes)

    model = Net(extractor, weights,vMF_kappa, occ_likely, mix_models,
              bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, 
          vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    
    t = torch.load(args.checkpoint)
    model.load_state_dict(t['state_dict'])
    if args.dataset == "UCF":
        train_dataset, test_dataset =  get_ucf101(root='Data_256',frames_path =args.data_dir,occ_dict = occ_dict)
    elif args.dataset="Kinetics":
        test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    #Lambda(lambda x: x / 255.0),
                    #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    CenterCrop(224),
                 ]
                ),
              ),
            ]
        )
        test_dataset = labeled_video_dataset(cl = None,
            data_path=os.path.join(args.data_dir, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform",2.5),
           transform=test_transform,
        occ = False
      ) 
    
    
    else:

        from Data_256.UCF101 import custom

        test_dataset = custom(root = args.data_dir)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True)
        
    return test(test_loader,model)
    
    
    
if __name__ == "__main__":
    main()

        
