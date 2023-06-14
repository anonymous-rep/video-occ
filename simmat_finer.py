from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from Initialization_Code.vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Code.vMFMM import *
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pickle
import os
import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
import torchvision
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from new_model import Classification_model
import pytorchvideo
import pytorchvideo.data
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
    RandomHorizontalFlip
)
from Kinetics.dataset import labeled_video_dataset
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
        
    def forward(self,x):

        x = self.feature_model([x])
        return x,[8,7,7]
class feature_modelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        
    def forward(self,x):

        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape()
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw
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

class feature_mvit(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.feature_model = Classification_model()
        self.feature_model = self.feature_model.cuda()
        state_dict = torch.load(checkpoint)
        self.feature_model.load_state_dict(state_dict['state_dict'])
        
       
    def forward(self,x):
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

    
def main_sim(dataset = "UCF101",checkpoint='',data_path='',fname='',save_name = ''):
    if dataset == "UCF101":
        model = feature_mvit(checkpoint)
        num_classes = 101
    elif dataset == "Kinetics":
        model = aClassification_model()
        num_classes = 400#
    #model = torchvision.models.vgg16(pretrained = True).features[:24]
    #model = old_feature_mvit()
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    #Lambda(lambda x: x / 255.0),
                  #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
        )
    
    with open(dict_dir+'dictionary_{}_{}.pickle'.format(fname,"768"), 'rb') as fh:#models/init_vgg/dictionary_vgg/dictionary_finer_mvit_kinetics_prertrained_768.pickleh:#"models/init_vgg/dictionary_vgg/dictionary_mvit_2048.pickle"
        centers = pickle.load(fh)
    
    paral_num = 10
    nimg_per_cat = 100
    imgs_par_cat =np.zeros(400)
    occ_level='ZERO'
    train_loader = DataLoader(
        
        train_dataset,
        sampler = SequentialSampler(train_dataset),
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    model.cuda()
    for cat in range(400):
        N= 100
        if dataset == "UCF101":
            train_dataset, test_dataset =  get_ucf101(cl= cat,root = 'Data_256',frames_path =data_path)
 #      train_loader = DataLoader(
        elif dataset == "Kinetics":
            train_dataset = labeled_video_dataset(cl = cat,data_path = os.path.join(data_path, "train"),clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),transform = train_transform)
        train_loader = DataLoader(

           train_dataset,
       batch_size=1,
       num_workers=4,
        drop_last=True,
        pin_memory=True)

        
    
        savename = os.path.join(sim_dir,'{}_simmat_pretrained_mthrh045_{}_K{}.pickle'.format(save_name,cat,"768"))
        ii =0
        if  not os.path.exists(savename) or True:
            r_set = [None for nn in range(N)]
            
            for iii,i in enumerate(train_loader):
                #x,y,z,_ = i
                x = i["video"]
                y = i["label"]
                y = int(y.detach().numpy())
                cat_idx = cat
                ### for singel frame similarity matrix
                #x = x.reshape(16,3,224,224)
                #print("epoch {} \t class{} \t category{}".format(iii,y,cat))
                if y == cat and imgs_par_cat[cat]<N :
                    with torch.no_grad():
                        x = x.cuda()
                        
                        layer_feature = model(x).detach().cpu().numpy()
                        ###
                        #layer_feature = torch.mean(layer_feature,dim = 0).numpy()
                        layer_feature = layer_feature.squeeze(0)
                        iheight,iwidth = layer_feature.shape[-2:]
                        lff = layer_feature.reshape(layer_feature.shape[0],-1).T
                        lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
                        r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
                        imgs_par_cat[cat_idx]+=1
                    ii+=1
            print('Determine best threshold for binarization - {} ...'.format(cat))
            nthresh=20
            magic_thhs=range(nthresh)
            coverage = np.zeros(nthresh)
            act_per_pix = np.zeros(nthresh)
            layer_feature_b = [None for nn in range(100)]
            magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])
            for idx,magic_thh in enumerate(magic_thhs):
                for nn in range(100):
                    layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
                    coverage[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0)>0)
                    act_per_pix[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0))
            coverage=coverage/100
            act_per_pix=act_per_pix/100
            best_loc=(act_per_pix>2)*(act_per_pix<15)
            if np.sum(best_loc):
                best_thresh = np.min(magic_thhs[best_loc])
            else:
                best_thresh = 0.45
            layer_feature_b = [None for nn in range(N)]
            for nn in range(N):
                layer_feature_b[nn] = (r_set[nn]<best_thresh).astype(int).T
            print('Start compute sim matrix ... magicThresh {}'.format(best_thresh))
            _s = time.time()

            mat_dis1 = np.ones((N,N))
            mat_dis2 = np.ones((N,N))
            N_sub = 200
            sub_cnt = int(math.ceil(N/N_sub))
            for ss1 in range(sub_cnt):
                start1 = ss1*N_sub
                end1 = min((ss1+1)*N_sub, N)
                layer_feature_b_ss1 = layer_feature_b[start1:end1]
                for ss2 in range(ss1,sub_cnt):
                    print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
                    _ss = time.time()
                    start2 = ss2*N_sub
                    end2 = min((ss2+1)*N_sub, N)
                    if ss1==ss2:
                        inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
                        para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
                    else:
                        layer_feature_b_ss2 = layer_feature_b[start2:end2]
                        inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
                        para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))

                    mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
                    mat_dis2[start1:end1, start2:end2] = para_rst[:,1]

                    _ee = time.time()
                    print('comptSimMat iter time: {}'.format((_ee-_ss)/60))

            _e = time.time()
            print('comptSimMat total time: {}'.format((_e-_s)/60))

            with open(savename, 'wb') as fh:
                print('saving at: '+savename)
                pickle.dump([mat_dis1, mat_dis2], fh)


if __name__ == "__main__":
    main_sim("Kinetics")
    
