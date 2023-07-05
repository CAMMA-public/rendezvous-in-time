"""
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France
Website: http://camma.u-strasbg.fr
"""

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pprint import pprint, pformat

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, SequentialLR, LinearLR, ExponentialLR
from sklearn.metrics import average_precision_score

from dataloader import CholecT50

from utils import *
from evaluator import evaluate_model

from network import RiT

def parse_args():
    """
    Parse input arguments
    """
    # model training details
    parser = argparse.ArgumentParser(description='Train a RIT model on CholecT45 dataset')
    parser.add_argument('--exp_name', default="rdv", type=str, 
                      help='experiment_name')
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int,
                      help='starting epoch')
    parser.add_argument('--max_epochs', default=100, type=int,
                      help='number of epochs to train')
    parser.add_argument('--aug_list', default=['rot90', 'hflip', 'contrast', 'original'], 
                      type=list, nargs='+', help='augmentation')    
    parser.add_argument('--disp_interval', default=100, type=int,
                      help='number of iterations to display')                          
    parser.add_argument('--nw', default=2, type=int,
                      help='number of worker to load data')
    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int,
                      default=0, help='gpu ids.')                     
    parser.add_argument('--bs', default=4, type=int,
                      help='batch_size')
    parser.add_argument('--seed', dest='seed', default=324, type=int,
                      help='seed value')       
    parser.add_argument('--m', dest='m', default=3, type=int,
                      help='clip size')                  

    # data and model weight directory details
    parser.add_argument('--data_dir', default="Cholec50", type=str, 
                      help='data directory')
    parser.add_argument('--save_dir', default="checkpoints",nargs=argparse.REMAINDER,
                      help='directory to save models')
    parser.add_argument('--save_folder', default="test", type=str, 
                      help='save folder name inside model folder for saving weights')
    parser.add_argument('--output_dir',default="./output",nargs=argparse.REMAINDER,
                      help='directory to save log file')

    # config optimization
    parser.add_argument('--early_stopping_patience', default=5, type=int,
                      help='num epochs to wait if val metric does not improve')

    # resume trained model
    parser.add_argument('--resume',default=0, type=int,
                      help='resume checkpoint or not')
    parser.add_argument('--evaluate',default=0, type=int,
                      help='evaluate with the provided checkpoint')
    parser.add_argument('--ckp_name', default="rit_rdv_split_weights.pth", 
                      type=str, help='checkpoint name to load model')    
    parser.add_argument('--ckp_folder', default='checkpoints', type=str,
                      help='folder to load checkpoints from')

    # log and display
    parser.add_argument('--log_name', default='test_runs1.log', type=str,
                      help='log file name for storing per epoch results')  

    # model_run_config
    parser.add_argument('--topK', default=5, type=int, 
                      help='topK accuracy')
    parser.add_argument('--ln', default=1, type=int,
                      help='use layer norm') 
    parser.add_argument('--cg', default=1, type=int,
                      help='add basemodel params to cagam') 
    parser.add_argument('--od1', default=1e-6, type=float,
                      help='decay used in the optim1')
    parser.add_argument('--od2', default=1e-6, type=float,
                      help='decay used in the optim2')
    parser.add_argument('--od3', default=1e-6, type=float,
                      help='decay used in the optim3')    
    parser.add_argument('--mom', default=0.95, type=float,
                      help='momentum value in sgd')
    parser.add_argument('--ms1',default=20, type=int,
                      help='milestone value for optim1')
    parser.add_argument('--ms2',default=40, type=int,
                      help='milestone value for optim2')
    parser.add_argument('--ms3',default=60, type=int,
                      help='milestone value for optim3')
    parser.add_argument('--g1',default=0.94, type=float,
                      help='exp gamma for optim1')
    parser.add_argument('--g2',default=0.95, type=float,
                      help='exp gamma for optim2')
    parser.add_argument('--g3',default=0.99, type=float,
                      help='exp gamma for optim3')
    parser.add_argument('--layers',default=8, type=int,
                      help='decoder layers')

    parser.add_argument('--split', default='rdv', type=str, 
                      help='data split for experiment')
    parser.add_argument('--fold', default=1, type=int, 
                      help='data fold') 

    args = parser.parse_args()
    return args

def train_net(net, loader, optimizers, schedulers, args, epoch, mode="train"):
    if mode == "train":
        net.train()
        apply_weights = True
    else:
        net.eval()
        apply_weights = False

    tool_wt, verb_wt, target_wt = get_component_weights(args)

    loss_tracker =  AverageMeter()
    tqdm_loader = tqdm(loader, unit="batch")
    for i, (frames, y_i, y_v, y_t, y_ivt) in enumerate(tqdm_loader):
        y_i    = y_i.float().cuda()
        y_v    = y_v.float().cuda()
        y_t    = y_t.float().cuda()
        y_ivt  = y_ivt.float().cuda()

        b, m, c, h, w = frames.size()
        frames = frames.view(-1, c, h, w).cuda()

        enc_i, enc_v, enc_t, dec_ivt = net(frames)

        # get the predictions from the current frame
        i_p   = enc_i[-1].view(b, m, -1)[:, -1, :]
        v_p   = enc_v[-1].view(b, m, -1)[:, -1, :]
        t_p   = enc_t[-1].view(b, m, -1)[:, -1, :]
        ivt_p = dec_ivt.view(b, m, -1)[:, -1, :]
        
        if mode == 'train':
            loss  = bce_loss(i_p, y_i, pos_wt=tool_wt) + \
                    bce_loss(v_p, y_v, pos_wt=verb_wt) + \
                    bce_loss(t_p, y_t, pos_wt=target_wt) + \
                    bce_loss(ivt_p, y_ivt, pos_wt=None)
        elif mode == 'val':
            loss  = bce_loss(i_p, y_i) + bce_loss(v_p, y_v) + bce_loss(t_p, y_t) + bce_loss(ivt_p, y_ivt)
        
        for opt in optimizers:
                opt.zero_grad(set_to_none=True)

        loss_tracker.update(loss.item())

        if mode == "train":
            loss.backward()
            for opt in optimizers:
                opt.step()

            net.zero_grad()

        tqdm_loader.set_postfix(mode=mode.upper(), epoch=epoch, batch=i, loss=f"{loss_tracker.avg:.3f}")

    if mode == "train": 
        for sch in schedulers:
            sch.step()

    return loss


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()

    global writer
    writer = SummaryWriter(f"checkpoints/{args.exp_name}")

    # check if the model weights path exists
    if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.exp_name))
    else:
        print("folder exists")

    # log file
    if args.log_name == 'test_runs1.log':
        args.log_name = f'{args.exp_name}.log'

    logfile = os.path.join(args.save_dir, args.exp_name, args.log_name)
    args.logfile = logfile
    print(f"Logfile to use is >>>>>>> {logfile}")

    # set seed
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) 
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
        print("cudnn enabled? --> ", torch.backends.cudnn.enabled)

    # print args
    print(">>>>>>>>>>>>>>>>>>>>>>  Args Start <<<<<<<<<<<<<<<<<<<<<<<\n", file=open(logfile, 'a+'))
    print(pformat(vars(args)), file=open(logfile, 'a+'))
    
    # load net
    hr_out = False
    use_ln = True if args.ln else False
    net = RiT(layer_size=args.layers, d_model=128, basename="resnet18", hr_output=False, use_ln=use_ln, m=args.m)

    # count model parameters
    num_params = count_parameters(net)
    print("number of params used >> ", num_params)

    # assign params to 3 param list
    params1, params2, params3 = [], [], []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'wsl' in key: 
                params1 += [{'params':[value]}]
            elif 'cagam' in key:
                params2 += [{'params':[value]}]
            elif 'basemodel' in key:
                if args.cg:
                    params2 += [{'params':[value]}]
                else:
                    params1 += [{'params':[value]}]
            elif 'decoder' in key or 'bottleneck' in key:
                params3 += [{'params':[value]}]
            else:
                print("---- keys missed ------")
                print(key)

    #param length
    print(f"LENGTH >> params1 {len(params1)} | params2 {len(params2)} | params3 {len(params3)}", file=open(logfile, 'a+'))
    print("\n>>>>>>>>>>>>>>>>>>>>>>  Args End <<<<<<<<<<<<<<<<<<<<<<<\n", file=open(logfile, 'a+'))

    opt_dict = {'opt1': {'lr': 0.1, 'sf': 0.1, 'iters': args.ms1, 'gamma': args.g1},
                'opt2': {'lr': 0.1, 'sf': 0.1, 'iters': args.ms2, 'gamma': args.g2},
                'opt3': {'lr': 0.1, 'sf': 0.1, 'iters': args.ms3, 'gamma': args.g3}
            }
    decay1 = args.od1 
    decay2 = args.od2 
    decay3 = args.od3
    mom_y  = args.mom

    optimizer1 = torch.optim.SGD(params1, lr=opt_dict["opt1"]["lr"], weight_decay=decay1, momentum=mom_y)
    scheduler1 = LinearLR(optimizer1, start_factor=opt_dict["opt1"]["sf"], total_iters=opt_dict["opt1"]["iters"]) 
    scheduler2 = ExponentialLR(optimizer1, gamma=opt_dict["opt1"]["gamma"])
    sched1     = SequentialLR(optimizer1, schedulers=[scheduler1, scheduler2], milestones=[opt_dict["opt1"]["iters"]+1])

    optimizer2 = torch.optim.SGD(params2, lr=opt_dict["opt2"]["lr"], weight_decay=decay2, momentum=mom_y) 
    scheduler3 = LinearLR(optimizer2, start_factor=opt_dict["opt2"]["sf"], total_iters=opt_dict["opt2"]["iters"])
    scheduler4 = ExponentialLR(optimizer2, gamma=opt_dict["opt2"]["gamma"])
    sched2     = SequentialLR(optimizer2, schedulers=[scheduler3, scheduler4], milestones=[opt_dict["opt2"]["iters"]+1])

    optimizer3 = torch.optim.SGD(params3, lr=opt_dict["opt3"]["lr"], weight_decay=decay3, momentum=mom_y) 
    scheduler5 = LinearLR(optimizer3, start_factor=opt_dict["opt3"]["sf"], total_iters=opt_dict["opt3"]["iters"])
    scheduler6 = ExponentialLR(optimizer3, gamma=opt_dict["opt3"]["gamma"])
    sched3     = SequentialLR(optimizer3, schedulers=[scheduler5, scheduler6], milestones=[opt_dict["opt3"]["iters"]+1])

    optimizers = [optimizer1, optimizer2, optimizer3]
    schedulers = [sched1, sched2, sched3]
        
    # load net to cuda if available
    if torch.cuda.is_available():
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        net = nn.DataParallel(net, device_ids=args.gpus)
        net = net.cuda()

    aug_list = args.aug_list.copy()
    print(f"Augmentations used  ----> {aug_list} ", file=open(logfile, 'a+'))

    best_metric = 0.0
    no_change_val = 0

    train_records, val_records, test_records = get_video_list(args)

    print("======== Train videos ========")
    print(train_records)
    print("======== Val videos ==========")
    print(val_records)
    print("======== Test videos =========")
    print(test_records)
    print("==============================")

    # load pretrained weights to resume training ..
    if args.resume:
        checkpoint_name = os.path.join(args.ckp_folder, args.ckp_name) 
        load_model_weights(net, checkpoint_name, skip_module=None)

    # evaluate with the checkpoint ..
    if args.evaluate:
        # get checkpoint path 
        checkpoint_name = os.path.join(args.ckp_folder, args.ckp_name) 
        # load model weights from the checkpoint path
        load_model_weights(net, checkpoint_name, skip_module=None)
        # evaluate on test data
        test_results = evaluate_model(net, args, mode="test")
        exit()

    print(f" ------------------ Training for clip size {args.m}------------------- ")

    # start training net
    for epoch in range(args.start_epoch, args.max_epochs+1):

        # one aug for one epoch
        train_aug = aug_list.pop()
        if len(aug_list) == 0:
            aug_list = args.aug_list.copy()
            np.random.shuffle(aug_list)

        # new CholecT50 
        dataset = CholecT50(
                    dataset_dir=args.data_dir, 
                    dataset_variant=args.split,
                    test_fold=args.fold,
                    augmentation_list=[train_aug],
                    normalize=True,
                    m=args.m
                )
        
        train_dataset, val_dataset, test_dataset = dataset.build()

        # create dataloader for train data
        train_loader = DataLoader(
                        train_dataset, 
                        batch_size=args.bs, 
                        num_workers=args.nw, 
                        shuffle=True, 
                        pin_memory=True, 
                        prefetch_factor=4*args.bs, 
                        persistent_workers=True
                    )

        train_loss = train_net(net, train_loader, optimizers, schedulers, args, epoch, mode="train")

        val_results = evaluate_model(net, args, mode='val')
        val_ivtmAP = val_results['triplet_mAP']

        if val_ivtmAP > best_metric:    
            best_metric = val_ivtmAP

            test_results = evaluate_model(net, args, mode='test')
            ivtmAP = test_results['triplet_mAP']

            save_name = os.path.join(args.save_dir, args.exp_name, f'{args.exp_name}_{epoch}.pth')
            exp_state = {
                'epoch': epoch,
                'model': net.state_dict(),
            }
            torch.save(exp_state, save_name)
            print(f"-Checkpoint at epoch {epoch} saved at {save_name}-", file=open(logfile, 'a+'))
            no_change_val = 0

        print(f"Best Metric for Test >>>> {ivtmAP:.3g}", file=open(logfile, 'a+'))

        # for early stopping
        no_change_val += 1
        if no_change_val == args.early_stopping_patience:
            print(f"-- Early Stopping applied at epoch {epoch} as Validation metric did not improve --", file=open(logfile, 'a+'))
            break
        
        print('='*52, file=open(logfile, 'a+'))
        #"""
    print("-- Training has completed --", file=open(logfile, 'a+'))