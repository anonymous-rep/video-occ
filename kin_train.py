import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle 
import scipy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from Data.UCF101 import get_ucf101
from Data_256.UCF101 import get_ucf101
from utils import AverageMeter, accuracy
from pytorchvideo.transforms import MixVideo
import torchvision
import pytorchvideo
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,Permute
)
from slowfast.config.defaults import get_cfg
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg


from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop
)
from Kinetics.dataset import labeled_video_dataset
DATASET_GETTERS = {'ucf101': get_ucf101}

def save_checkpoint(state, is_best, checkpoint):
    filename=f'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    cosine_end_lr=1.6e-5,
                                    lr=.0016,
                                    offset =0,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., cosine_end_lr+(lr-cosine_end_lr)*(math.cos(math.pi * (current_step-offset)/(num_training_steps-offset))+1)+.5)

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


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
def main_training_testing(EXP_NAME):
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--out', default=f'results/{EXP_NAME}', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--arch', default='resnet3D18', type=str,
                        help='dataset name')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--data_path', default='/home/yogesh/Naman/ucf-101/frames-128x128', type=str,
                        help='video frames path')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')
    parser.add_argument('--cfg_files',default = "",type = str)
    parser.add_argument('--val',default = True,type = bool)
    parser.add_argument('--occ',default= False,type=bool)
    parser.add_argument('--occ_size',default = None , type= int)

    args = parser.parse_args()
    best_acc = 0
    best_acc_2 = 0
    
    def create_model(args):
        if args.arch == 'resnet3D18':
            import models.video_resnet as models
            model = models.r2plus1d_18(num_classes=400,pretrained=True)
       #     for name, module in model.named_modules():
        #        print(name,module)
            #model.fc = torch.nn.Linear(512,101,bias=True)
        if args.arch == "mvit":
            import pytorchvideo
            model  =  torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4",pretrained = True)
            #for param in model.parameters():
            #    param.requires_grad = False
            #model.head.proj = torch.nn.Linear(768,101,bias =True)
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(pytorch_total_params)
        if args.arch == "test_mvit":
            from new_model import occ_Classification_model
            model = occ_Classification_model()
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(pytorch_total_params)
        return model
    print(args.arch)
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    #train_dataset, test_dataset = DATASET_GETTERS[args.dataset]('Data_256', args.frames_path)

    #model = create_model(args)
    #model.to(args.device)
    

    
    args.iteration = 100#len(train_dataset) // args.batch_size // args.world_size
    for path_to_config in args.cfg_files:
        cfg = load_config("Kineticsrun/Mvitv2_kinetics.yaml")
        cfg = assert_and_infer_cfg(cfg)

    args.lr = 1e-4
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay = 0.05)
    #print("occ Model :i3d")
    print(args.arch)
    img_size=224
    num_frames = 16
    if args.arch == "mvit":
        model = torch.hub.load("facebookresearch/pytorchvideo",model = "mvit_base_16x4",pretrained = True)
    elif args.arch == "mvitv2":
        model = build_model(cfg)
        model.load_state_dict(torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth")["model_state"])

    elif args.arch == "videomae":
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    elif args.arch == 'x3d':
        num_frames = 16
        model = model = torch.hub.load("facebookresearch/pytorchvideo",model = "x3d_m",pretrained = True)
    elif args.arch == 'i3d':
        model = torch.hub.load("facebookresearch/pytorchvideo",model = "i3d_r50",pretrained = True)
        num_frames = 8
    elif args.arch == 'r2p1':
        import models.video_resnet as models
        model = models.r2plus1d_18(num_classes=400,pretrained=True)
        img_size =112
                
    
    test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    #Lambda(lambda x: x / 255.0),
                    #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    CenterCrop(img_size),
                 ]
                ),
              ),
            ]
    )
    
    test_datasets = [labeled_video_dataset(
            cl = None,
            data_path=args.data_path,#os.path.join("/home/ak119590/datasets/K400/videos_256/test", "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform",2),
           transform=test_transform,
        occ =args.occ,
        
                                           
      ) for i in range(1)]
    
        
    train_sampler = RandomSampler
    
    test_loaders =[DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True) for test_dataset in test_datasets]  
    
    
    
    
    
    
    #model = torch.hub.load("facebookresearch/pytorchvideo",model = "x3d_m",pretrained = True)
    #state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth')
    #model.load_state_dict(state_dict["model_state"])
    #print("MVITv2")
    #model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
  #  model = build_model(cfg)
    #model.load_state_dict(torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth")["model_state"])
    model.cuda()
    test_accs = []
    model.zero_grad()
    
    test_loss = 0.0
    test_acc = 0.0
    test_model = model
      
    test_loss, test_acc, test_acc_2 = test(args, test_loaders, test_model, epoch)
    print(f'epoch:{epoch},test_loss:{test_loss},test_acc{test_acc}')
    
def test(args, test_loaders, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    predicted_target_not_softmax = {}

    if not args.no_progress:
        test_loader = tqdm(test_loaders[0])

    with torch.no_grad():
        for test_loader in test_loaders:
            print("new")
            for batch_idx,data in enumerate(test_loader):
                #print(batch_idx)
                data_time.update(time.time() - end)
                model.eval()
                inputs = data["video"]
                #print(inputs.shape)
                #inputs = inputs.permute(0,2,1,3,4)
                
                targets= data["label"]
                if args.arch == "mvitv2":
                    inputs = [inputs.cuda()]
                elif args.arch == "videomae":
                    inputs = inputs.permute(0,2,1,3,4).cuda()
                else:
                    inputs = inputs.cuda()
                targets = targets.cuda()
                video_name = data["video_name"]
                if args.arch == "videomae":
                    outputs = model(inputs).logits
                else:
                    outputs = model(inputs)
                if args.arch == "mvitv2":
                    inputs = inputs[0]
                loss = F.cross_entropy(outputs, targets)
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

        secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
        secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
        print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')

    return losses.avg, secondary_accuracy, secondary_accuracy_not_softmax


if __name__ == '__main__':
    cudnn.benchmark = True
    EXP_NAME = 'Kinetics_Augmented_MVIT_UCF101_SUPERVISED_TRAINING'
    print(EXP_NAME)
    main_training_testing(EXP_NAME)
