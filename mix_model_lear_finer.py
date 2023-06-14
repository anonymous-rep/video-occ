from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import os
import pickle
import numpy as np
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor, model_save_dir
from Code.helpers import getImg, imgLoader, Imgset
from torch.utils.data import DataLoader
import cv2
import gc
import matplotlib.pyplot as plt
import scipy.io as sio
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
import torchvision
from new_model import Classification_model
from Kineticsrun.MVIT import MViT
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg

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

vc_num = 768

#with open("models/init_vgg/dictionary_vgg/dictionary_kin_mvitv2_prertrained_768.pickle", 'rb') as fh:#finer_mvit_kinetics_prertrained_768.pickle
#	centers = pickle.load(fh)
#####################
# BEWARE
#####################
bool_pytroch = True
bool_plot_view_p3d=False

mixdir = init_path + 'mix_model_vmf_{}_EM_all/'.format(dataset)
if not os.path.exists(mixdir):
	os.makedirs(mixdir)
occ_level='ZERO'
occ_type=''
spectral_split_thresh=0.05
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


class feature_modelv2(nn.Module):
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



class feature_model(nn.Module):
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
        self.feature_extractor = feature_modelv2()
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
        #out = out+cls_token
        out = torch.mean(out,dim=1)
        ##out_avg,_ = torch.max(out,dim =1)
        return out


def learn_mix_model_vMF(model,category,num_layers = 2,num_clusters_per_layer = 2,frac_data=1.0,data_loader = None,mixdir = None,fname ='',matrix_save='',save_name = ''):

    with open(dict_dir+'dictionary_{}_{}.pickle'.format(fname,"768"), 'rb') as fh:
        centers = pickle.load(fh)
#
    spectral_split_thresh=0.01

    sim_fname = model_save_dir+'init_vgg/'+'similarity_vgg_pool5_pascal3d+/'+'{}_simmat_pretrained_mthrh045_{}_K{}.pickle'.format(matrix_save,category,768)

    # Spectral clustering based on the similarity     matrix_kinetics_mthrh045_{}_K{}.pickle
    with open(sim_fname, 'rb') as fh:
        mat_dis1, _ = pickle.load(fh)

    mat_dis = mat_dis1
    subN = np.int32(mat_dis.shape[0]*frac_data)
    mat_dis = mat_dis[:subN,:subN]
    print('total number of instances for obj {}: {}'.format(category, subN))
    N=subN
    img_idx = np.asarray([nn for nn in range(N)])
    #imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
    #data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
    print("N {}".format(N))
    r_set = []#[None for nn in range(N)]
    #layer_features 	  =	np.zeros((N,featDim,max_1,max_2),dtype=np.float32)
    ii=0
    for iii,data in enumerate(data_loader):
        #if np.mod(ii,10)==0:
         #   print('{} / {}'.format(ii,N))
        #input, label,z,_= data
        input = data["video"]
        label = data["label"]
        y = int(label.detach().numpy())
        #print(y==category)
        if category == y and ii<100:
            input = input.cuda()
            #input =  input.reshape(16,3,224,224)
            model.eval()
            layer_feature = model(input).detach().cpu()
            #layer_feature = torch.mean(layer_feature,0).numpy()
            layer_feature = layer_feature.squeeze(0).numpy() #for non single frame
            
            iheight,iwidth = layer_feature.shape[-2:]
            
            lff = layer_feature.reshape(layer_feature.shape[0],-1).T
                #lff_i = lff[i].T
            lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
               
           # print(lff_norm.shape,centers.shape)
        # compute dot product
            tmp = 1-(cdist(lff_norm, centers, 'cosine')).astype('float32')
        # compute vMF likelihood
            tmp = tmp
            tmp = np.exp(vMF_kappa*tmp)
        # reshape such that the spatial position is preserved during learning
            feat_map = tmp.reshape(iheight, iwidth, -1).astype(np.float32).T
            r_set.append(feat_map)
        if ii >=100:
            break
    print("r_set len {} and shape {}".format(len(r_set),r_set[0].shape))
    # num cluster centers
    max_0 = 768
    # width
    max_1 = max([r_set[nn].shape[1] for nn in range(N)])
    # height
    max_2 = max([r_set[nn].shape[2] for nn in range(N)])
    print(max_0, max_1, max_2)
    layer_feature_vmf = np.zeros((N,max_0,max_1, max_2), dtype=np.float32)
    
    for nn in range(N):
        
        vnum, ww, hh = r_set[nn].shape
        assert (vnum == max_0)
        diff_w1 = int((max_1 - ww) / 2)
        diff_w2 = int(max_1 - ww - diff_w1)
        assert (max_1 == diff_w1 + diff_w2 + ww)
        diff_h1 = int((max_2 - hh) / 2)
        diff_h2 = int(max_2 - hh - diff_h1)
        assert (max_2 == diff_h1 + diff_h2 + hh)
        padded = np.pad(r_set[nn], ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant',constant_values=0)
        r_set[nn] = []
        layer_feature_vmf[nn,:,:,:] = padded
    print("layer feature",layer_feature.shape)
    #layer_feature = layer_feature.reshape(768,7,7)
    
    mat_full = mat_dis + mat_dis.T - np.ones((N,N))
    np.fill_diagonal(mat_full, 0)

    mat_sim = 1. - mat_full

    # setup caching variables
    tmp = list()
    tmp.append(np.zeros(mat_sim.shape[0]))
    LABELS 	= list()
    LABELS.append(tmp)
    tmp = list()
    tmp.append(mat_sim)
    MAT = list()
    MAT.append(tmp)
    tmp = list()
    tmp.append(range(mat_sim.shape[0]))
    IMAGEIDX = list()
    IMAGEIDX.append(tmp)

    # start hierarchical spectral clustering
    FINAL_CLUSTER_ASSIGNMENT=[]
    for i in range(num_layers):
        MAT_SUB = list()
        LABELS_SUB = list()
        IMAGEIDX_SUB = list()

        print('Clustering layer {} ...'.format(i))
        for k in range(np.power(num_clusters_per_layer,i)):
            parent_counter 	= int(np.floor(k / num_clusters_per_layer))
            leaf_counter	= int(np.mod(k,num_clusters_per_layer))
            idx = np.where(LABELS[i][parent_counter] == leaf_counter)[0]
            if len(idx)>spectral_split_thresh*N:
                mat_sim_sub = MAT[i][parent_counter][np.ix_(idx, idx)] # subsample similarity matrix
                MAT_SUB.append(mat_sim_sub)
                IMAGEIDX_SUB.append(np.array(IMAGEIDX[i][parent_counter])[idx])
                cls_solver = SpectralClustering(n_clusters=num_clusters_per_layer, affinity='precomputed', random_state=0)
                cluster_result = cls_solver.fit_predict(mat_sim_sub)
                LABELS_SUB.append(cluster_result)

                print('{} {} {} {}'.format(i,k,sum(cluster_result==0),sum(cluster_result==1)))

                if i==num_layers-1:
                    for ff in range(num_clusters_per_layer):
                        idx_tmp=IMAGEIDX_SUB[k][cluster_result == ff]
                        if len(idx_tmp)>0.001*N:
                            FINAL_CLUSTER_ASSIGNMENT.append(np.array(idx_tmp))
            elif len(idx)>0.001*N:
                FINAL_CLUSTER_ASSIGNMENT.append(np.array(IMAGEIDX[i][parent_counter])[idx])
                LABELS_SUB.append([])
                IMAGEIDX_SUB.append([])
                MAT_SUB.append([])
            else:
                LABELS_SUB.append([])
                IMAGEIDX_SUB.append([])
                MAT_SUB.append([])
        MAT.append(MAT_SUB)
        LABELS.append(LABELS_SUB)
        IMAGEIDX.append(IMAGEIDX_SUB)

    mixmodel_lbs = np.ones(len(LABELS[0][0]))*-1
    K=len(FINAL_CLUSTER_ASSIGNMENT) # number of clusters
    for i in range(K):
        mixmodel_lbs[FINAL_CLUSTER_ASSIGNMENT[i]]=i

    mixmodel_lbs = mixmodel_lbs[:N]

    for kk in range(K):
        print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs==kk)))

        
    alpha = []
    for kk in range(K):
        # get samples for mixture component
        bool_clust = mixmodel_lbs==kk
        bidx = [i for i, x in enumerate(bool_clust) if x]
        num_clusters = 768#vmf.shape[1]
        # loop over samples
        for idx in bidx:
            # compute
            vmf_sum = np.sum(layer_feature_vmf[img_idx[idx]], axis=0)
            vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
            vmf_sum = vmf_sum.repeat(num_clusters, axis=0)+1e-3
            mask = vmf_sum > 0
            layer_feature_vmf[img_idx[idx]] = mask*(layer_feature_vmf[img_idx[idx]]/vmf_sum)

        N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0) # stores the number of samples
        mask = (N_samp > 0)
        vmf_sum = mask * (np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
        alpha.append(vmf_sum)

    '''
    # ML updates of mixture model and vMF mixture coefficients
    '''
    numsteps = 10
    for ee in range(numsteps):
        changed = 0
        mixture_likeli = np.zeros((subN,K))
        print('\nML Step {} / {}'.format(ee, numsteps))
        changed_samples = np.zeros(subN)
        for nn in range(subN):
            if nn % 100 == 0:
                print('{}'.format(nn))
            #compute feature likelihood
            for kk in range(K):
                like_map = layer_feature_vmf[img_idx[nn]]*alpha[kk]
                likeli = np.sum(like_map, axis=0)+1e-10
                mixture_likeli[nn, kk] = np.sum(np.log(likeli))

            #compute new mixture assigment for feature map
            new_assignment = np.argmax(mixture_likeli[nn, :])
            if new_assignment!=mixmodel_lbs[nn]:
                changed+=1
                changed_samples[nn]=1
            mixmodel_lbs[nn] = new_assignment

        for kk in range(K):
            print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs == kk)))
        print('{} changed assignments'.format(changed))

        #update mixture coefficients heres
        for kk in range(K):
            # get samples for mixture component
            bool_clust = mixmodel_lbs == kk
            if np.sum(bool_clust) > 0:
                bidx = [i for i, x in enumerate(bool_clust) if x]
                num_clusters = 768  # vmf.shape[1]
                # loop over samples
                for idx in bidx:
                    # compute
                    vmf_sum = np.sum(alpha[kk]*layer_feature_vmf[img_idx[idx]], axis=0)
                    vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
                    vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-10
                    mask = vmf_sum > 0
                    layer_feature_vmf[img_idx[idx]] = mask * (alpha[kk]*layer_feature_vmf[img_idx[idx]] / vmf_sum)

                N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
                mask = (N_samp > 0)
                alpha[kk]= mask*(np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
                gc.collect()

        if changed/subN<0.01:
            break
    print(np.array(alpha).shape)
    savename = os.path.join(mixdir,'model_{}_K4_FEATDIM{}_{}_specific_view.pickle'.format(category,768, save_name))
    with open(savename, 'wb') as fh:
        pickle.dump(alpha, fh)
def main_mix(dataset = "UCF101",checkpoint='',data_path='',fname= '',matrix_save='',save_name=''):
    if dataset == "UCF101":
        model = feature_mvit(checkpoint)
        num_classes = 101
    elif dataset == "Kinetics":
        model = aClassification_model()#
        num_classes = 400
    #model = torchvision.models.vgg16(pretrained = True).features[:24]
    model = model.cuda()
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
    
    
    
    for category in range(num_classes):
        for num_layers in [1]:
            if dataset == "UCF101":
                train_dataset, test_dataset =  get_ucf101(cl = category,root = 'Data_256',frames_path =data_path)#/home/c3-0/da
            elif dataset == "Kinetics":
                train_dataset = labeled_video_dataset(cl = category,data_path = os.path.join(data_path, "train"),clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),transform = train_transform)
            train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=4,
            drop_last=True,
            pin_memory=True)

            learn_mix_model_vMF(model,category,num_layers=num_layers,num_clusters_per_layer=2,data_loader = train_loader,mixdir =mixdir,fname=fname,matrix_save=matrix_save,save_name=save_name)


if __name__ == "__main__":
    main_mix("Kinetics")
