from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, update_clutter_model
from Code.config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures
from Code.config import config as cfg
from torch.utils.data import DataLoader
from Code.losses import ClusterLoss
from Code.model import resnet_feature_extractor
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from pytorchvideo.layers import convolutions
import time
import os
import torch
import torch.nn as nn
import numpy as np
import random
from utils import AverageMeter, accuracy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torchvision
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
    RandomHorizontalFlip,
    CenterCrop
)
from Kinetics.dataset import labeled_video_dataset


from new_model import Classification_model
vMF_kappa=76

class another_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torchvision.models.video.r3d_18(pretrained = True)
    def forward(self,x):
        x = self.feature.stem(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        x = torch.mean(x,dim = 2)
        return x


class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_model = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        #self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        #state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_model.patch_embed(x)
        x = self.feature_model.cls_positional_encoding(x)
        thw = self.feature_model.cls_positional_encoding.patch_embed_shape()
        for blck in self.feature_model.blocks:
            x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw
    
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


class feature_modelv2(nn.Module):
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


#---------------------
# Training Parameters
#---------------------
alpha = 3  # vc-loss
beta = 3 # mix loss
likely = 0.7 # occlusion likelihood
lr = 1e-6 # learning rate
batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
# Training setup
vc_flag = True # train the vMF kernels
mix_flag = True # train mixture components
ncoord_it = 15 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False
bool_train_with_occluders = False
#print("SGD")

if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']

out_dir = model_save_dir + 'Kinetics_spatial_trial_single_test_mvit_train_only_class_with_model_wiht concat_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
    layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,0)


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, vc_flag=True, mix_flag=False,data_dir=''):
    
    
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
    )
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
    train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(data_dir, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),
           transform=train_transform
    )
    
    test_datasets = [labeled_video_dataset(cl = None,
            data_path=os.path.join(data_Dir, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform",2.5),
           transform=test_transform,
        occ = False
      ) for i in range(1)]
    
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory = True,        
        drop_last=True,
        )

  
    
    
    best_check = {
        'epoch': 0,
        'best': 0,
        'val_acc': 0
    }
    out_file_name = savedir + 'result.txt'
    

    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    for param in model.backbone.parameters():
        param.requires_grad = False
    #model.backbone.t_reduction.requires_grad = True
    
    if not vc_flag:
        model.conv1o1.weight.requires_grad = False
    else:
        model.conv1o1.weight.requires_grad = True

    if not mix_flag:
        model.mix_model.requires_grad = False
    else:
        model.mix_model.requires_grad = True

    classification_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusterLoss()

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)#,weight_decay = 0.05)#torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=.99)#0.98)

    print('Training')

    for epoch in range(1,epochs):
        
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        model.backbone.eval()
        cnt =0
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
        #        print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                start = time.time()

            #input,label,z,_ = data
            input = data["video"]
            label= data["label"]
        
            y = int(label.detach().numpy())
            #if y>100:
            #    continue
            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])
            cnt+=1
      #      if cnt % 10 == 0:
      #          break
            #input = input.reshape(16,3,224,224)
            output, vgg_feat, like = model(input)
            #print(output.shape)
            
            #output = torch.max(output,0,keepdims  = True)[0]
            out = output.argmax(1)
            correct += torch.sum(out == label)
            class_loss = classification_loss(output, label) / output.shape[0]
            
            loss = class_loss
            if alpha != 0:
                clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
                loss += alpha * clust_loss

            if beta!=0:
                mix_loss = like[0,label[0]]
                loss += -beta *mix_loss
            
            #with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            
            # pseudo batches
            if np.mod(index,batch_size)==0:# and index!=0:
                optimizer.step()
                optimizer.zero_grad()
            check = {"state_dict":model.backbone.state_dict(),
            "val_acc":0,
                     "epoch":0}
            #save_checkpoint(check,"backbone_weights.pth",True)
            train_loss += class_loss.detach() * input.shape[0]

        with torch.no_grad():
            
        
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            predicted_target = {}
            ground_truth_target = {}
            predicted_target_not_softmax = {}
                
            for test_dataset in test_datasets:
                test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True)

                cnt = 0
                
                with torch.no_grad():
                    for batch_idx, data in enumerate(test_loader):
                        data_time.update(time.time() - end)
                        model.eval()
                                
                        inputs = data["video"]
                        targets= data["label"]
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        video_name = data["video_name"]
                        #inputs = inputs.reshape(16,3,224,224)
                        
                        
                        outputs,_,_ = model(inputs)
                        #outputs = torch.max(outputs,0,keepdims = True)[0]
                        out = outputs.argmax(1)
                        cnt+=1
                        #if cnt%10 == 0:
                        #    break
                        loss = classification_loss(outputs, targets)/ outputs.shape[0]
                        
                        out_prob = F.softmax(outputs, dim=1)
                        out_prob = out_prob.cpu().numpy().tolist()
                        targets = targets.cpu().numpy().tolist()
                        outputs = outputs.cpu().numpy().tolist()
            
                        for iterator in range(len(video_name)):
                            if video_name[iterator] not in predicted_target:
                                predicted_target[video_name[iterator]] = []
                
                            if video_name[iterator] not in predicted_target_not_softmax:
                                predicted_target_not_softmax[video_name[iterator]] = []

                            if video_name[iterator] not in ground_truth_target:
                                ground_truth_target[video_name[iterator]] = []

                            predicted_target[video_name[iterator]].append(out_prob[iterator])
                            predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                            ground_truth_target[video_name[iterator]].append(targets[iterator])
                        #print(len(predicted_target))
                
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
                pred_values.append(predicted_target[key])
                pred_values_not_softmax.append(predicted_target_not_softmax[key])
                target_values.append(ground_truth_target[key])
        
            pred_values = np.array(pred_values)
            pred_values_not_softmax = np.array(pred_values_not_softmax)
            target_values = np.array(target_values)

            secondary_accuracy = (pred_values == target_values)*1
            secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))*100
            print(f'test accuracy after softmax: {secondary_accuracy}')
            val_acc = secondary_accuracy
            secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
            secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
            print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')
            
            if val_acc>best_check['val_acc']:
                print('BEST: {}'.format(val_acc))
                out_file.write('BEST: {}\n'.format(val_acc))
            best_check = {
                    'state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                    }
            save_checkpoint(best_check, savedir + 'vc' + str(epoch + 1) + '.pth', True)
            
        print('\n')
        out_file.close()
        break
    
        
    #return best_check

def main(dict_name,mix_model,checkpoint):
    
    extractor = aClassification_model()
    extractor.cuda(device_ids[0])
    #state_dict = torch.load("backbone_weights.pth")
    #extractor.load_state_dict(state_dict['state_dict'])
    dict_dir = dict_dir+'dictionary_{}_{}.pickle'.format(dict_name,"768")
    weights = getVmfKernels(dict_dir, device_ids)
    bool_load_pretrained_model = False
    if bool_load_pretrained_model:
        pretrained_file = 'PATH TO .PTH FILE HERE'
    else:
        pretrained_file = ''
    occ_likely = []
    for i in range(400): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    mix_models = getCompositionModel(device_ids,mix_model,layer,categories_train,compnet_type=compnet_type,num_classes=400)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models,
              bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, 
          vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    if bool_load_pretrained_model:
        net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

    net = net.cuda(device_ids[0])

    
	# get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac=0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac=0.1

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info = out_dir + 'config.txt'
    config_file = open(info, 'a')
    config_file.write(dataset)
    out_str = 'layer{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr{}/'.format(layer,alpha,beta,vc_flag,mix_flag,likely,vc_num,lr)
    config_file.write(out_str)
    out_str = 'Train\nDir: {}, vMF_kappa: {}, alpha: {},beta: {}, likely:{}\n'.format(out_dir, vMF_kappa, alpha,beta,likely)
    config_file.write(out_str)
    #print(out_str)
    out_str = 'pretrain{}_file{}'.format(bool_load_pretrained_model,pretrained_file)
    #print(out_str)
    config_file.write(out_str)
    config_file.close()
    train(model=net, train_data=None, val_data=None, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag,data_dir =data_path)

if __name__ == "__main__":
    main()
