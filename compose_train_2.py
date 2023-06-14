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
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torchvision
from new_model import occ_Classification_model
import models.video_resnet as models

class cnn_feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.r2plus1d_18(num_classes=400,pretrained=True)
#self.feature_model.head.proj = torch.nn.Linear(768,101,bias =True)
        #state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
        model.fc = torch.nn.Linear(512,101,bias=True)
        self.feat_model = model.cuda()
        state_dict = torch.load("results/Trial_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feat_model.load_state_dict(state_dict["state_dict"])
    def forward(self,x):
        t = self.feat_model.stem(x)
        t = self.feat_model.layer1(t)
        t = self.feat_model.layer2(t)
        t = self.feat_model.layer3(t)
        out = self.feat_model.layer4(t)
        out = out.mean(dim =2)
        return out
class feature_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.hub.load("facebookresearch/pytorchvideo","mvit_base_16x4",pretrained = True)
        self.feature_extractor.head.proj = torch.nn.Linear(768,101,bias =True)
        self.feature_extractor = self.feature_extractor.cuda()
        
        state_dict = torch.load("results/AUG_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        self.feature_extractor.load_state_dict(state_dict['state_dict'])
        
        #state_dict = torch.load("results/less_MVIT_UCF101_SUPERVISED_TRAINING/model_best.pth.tar")
        #self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x):

        x = self.feature_extractor.patch_embed(x)
        x = self.feature_extractor.cls_positional_encoding(x)
        thw = self.feature_extractor.cls_positional_encoding.patch_embed_shape
        for blck in self.feature_extractor.blocks:
            x,thw= blck(x,thw)
        #out = x[:,1:,:]
        #out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        return x,thw






class feature_mvit(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.feature_model = feature_model()#torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4")#Classification_model()
        state_dict= torch.load(checkpoint)
        self.feature_model.load_state_dict(state_dict['state_dict'])
    def forward(self,x,train=False):
        x,thw = self.feature_model(x)
        
        #x = self.feature_model.cls_positional_encoding(x)
        #thw = self.feature_model.cls_positional_encoding.patch_embed_shape
        #for blck in self.feature_model.blocks:
         #   x,thw= blck(x,thw)
        out = x[:,1:,:]
        cls_token = x[:,0,:]
        cls_token = cls_token.reshape(-1,1,768,1,1)
        out = out.reshape(-1,thw[0],768,thw[1],thw[2])
        out = out+cls_token
        #else:
        #    out = 2*99/100*out+2/100*cls_token
        
        out = torch.mean(out,dim=1)
        ##out_avg,_ = torch.max(out,dim =1)
        return out




#---------------------
# Training Parameters
#---------------------
alpha = 3  # vc-loss
beta = 3 # mix loss
likely = 0.7 # occlusion likelihood
lr = 1e-4 # learning rate
batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
# Training setup
vc_flag = True # train the vMF kernels
mix_flag = True # train mixture components
ncoord_it = 5 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False
bool_train_with_occluders = False
#print("SGD")

if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']

out_dir = model_save_dir + 'trial_mvit_single_test_mvit_train_only_class_with_model_wiht concat_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
    layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, vc_flag=True, mix_flag=False):
    best_check = {
        'epoch': 0,
        'best': 0,
        'val_acc': 0
    }
    out_file_name = savedir + 'result.txt'
    total_train = len(train_data)
    #train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    #val_loaders=[]
    
    #for i in range(len(val_data)):
    #    val_loader = DataLoader(dataset=val_data[i], batch_size=1, shuffle=True)
    #    val_loaders.append(val_loader)
    train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path ="/home/c3-0/datasets/UCF101_frames/frames_256")
    #"/home/yogesh/Naman/ucf-101/frames-128x128",'/home/c3-0/datasets/UCF101_frames/frames_256'
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=4,
        pin_memory=True)
    index = np.random.randint(low = 0,high = len(train_dataset)-1,size=10)
    occ_data = torch.utils.data.Subset(train_dataset, index)
    #state_dict = torch.load("models/train_only_class_with_model_wiht concat_pool5_a3_b3_vcTrue_mixTrue_occlikely0.7_vc2048_lr_1e-07_pascal3d+_pretrainedFalse_epochs_100_occFalse_backbonevgg_0/vc1.pth")
    #model.load_state_dict(state_dict['state_dict'])
    


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

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3)#torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=.9)#0.98)

    print('Training')

    for epoch in range(epochs):
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        model.backbone.eval()
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
        #        print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                start = time.time()

            #input,label,z,_ = data
            input = data["video"]
            label= data["label"]
        
            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])
            #(input)
            #input = input.reshape(16,3,224,224)
            
            output, vgg_feat, like = model(input)
           # print(output.shape,input.shape)
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
            #check = {"state_dict":model.backbone.state_dict(),
            #"val_acc":0,
             #        "epoch":0}
            #save_checkpoint(check,"backbone_weights.pth",True)
            train_loss += class_loss.detach() * input.shape[0]
            

        updated_clutter = update_clutter_model(model,device_ids,occ_data = occ_data)
        #updated_clutter = model.get_clutter_model("vmf",76)
        model.clutter_model = updated_clutter
        scheduler.step()
        train_acc = correct.cpu().item() / total_train
        train_loss = train_loss.cpu().item() / total_train
        out_str = 'Epochs: [{}/{}], Train Acc:{}, Train Loss:{}'.format(epoch + 1, epochs, train_acc, train_loss)
        print(out_str)
        out_file.write(out_str)

        # Evaluate Validation images
        model.eval()
        with torch.no_grad():
 #           correct = 0
#			mval_accs=[]
#			for i in range(len(val_loaders)):
#				val_loader = val_loaders[i]
#				correct_local=0
#				total_local = 0
#				val_loss = 0
#				out_pred = torch.zeros(len(val_data[i].images))
#				for index, data in enumerate(val_loader):
#					input,_, label = data
#					input = input.cuda(device_ids[0])
#					label = label.cuda(device_ids[0])
#					output,_,_ = model(input)
#					out = output.argmax(1)
#					out_pred[index] = out
#					correct_local += torch.sum(out == label)
#					total_local += label.shape[0]

#					class_loss = classification_loss(output, label) / output.shape[0]
#					loss = class_loss
#					val_loss += loss.detach() * input.shape[0]
#				correct += correct_local
#				val_acc = correct_local.cpu().item() / total_local
#				val_loss = val_loss.cpu().item() / total_local
#				val_accs.append(val_acc)
#				out_str = 'Epochs: [{}/{}], Val-Set {}, Val Acc:{} Val Loss:{}\n'.format(epoch + 1, epochs,i , val_acc,val_loss)
#				print(out_str)
#				out_file.write(out_str)
#			val_acc = np.mean(val_accs)
#			out_file.write('Epochs: [{}/{}], Val Acc:{}\n'.format(epoch + 1, epochs, val_acc))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            predicted_target = {}
            ground_truth_target = {}
            predicted_target_not_softmax = {}
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    data_time.update(time.time() - end)
                    model.eval()
                    inputs = data["video"]
                    label= data["label"]
        
                    video_name = data["video_path"]
                    inputs = inputs.cuda()
                    targets = label.cuda()
                    #inputs = inputs.reshape(16,3,224,224)
                    outputs,_,_ = model(inputs)
                    #outputs = torch.max(outputs,0,keepdims = True)[0]
                    out = output.argmax(1)
                    
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
    return best_check

def main(dict_name,mix_model,data_dir,checkpoint):
    
    extractor = feature_mvit(checkpoint)
    extractor.cuda(device_ids[0])
    dict_dir = dict_dir+'dictionary_{}_{}.pickle'.format(dict_name,"768")
    weights = getVmfKernels(dict_dir, device_ids)
    print(weights.shape)
    #print(weights)
    bool_load_pretrained_model = False
    if bool_load_pretrained_model:
        pretrained_file = 'PATH TO .PTH FILE HERE'
    else:
        pretrained_file = ''
    occ_likely = []
    for i in range(101): #changed from len()
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path =data_dir)
    
    train_sampler = RandomSampler
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=4,
        pin_memory=True)
    
    index = np.random.randint(low = 0,high = len(train_dataset)-1,size=10)
    occ_data = torch.utils.data.Subset(train_dataset, index)
    
    
    
    # load the CompNet initialized with ML and spectral clustering#mix_model_path="CompositionalNets/models/init_vgg/mix_model_vmf_pascal3d+_EM_all"
    mix_models = getCompositionModel(device_ids,mix_model,layer,categories_train,compnet_type=compnet_type,num_classes = 101)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models,
              bool_mixture_bg=False,compnet_type="vmf",num_mixtures=num_mixtures, 
          vc_thresholds=cfg.MODEL.VC_THRESHOLD,occ_data = occ_data)
    if bool_load_pretrained_model:
        net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

    net = net.cuda(device_ids[0])

    train_imgs=[]
    train_masks = []
    train_labels = []
    val_imgs = []
    val_labels = []
    val_masks=[]
    
	# get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac=0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac=0.1

   
    
    load_flag = False
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
    train(model=net, train_data=train_loader, val_data=test_loader, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag)

if __name__ == "__main__":
    main()
