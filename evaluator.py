"""
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France
Website: http://camma.u-strasbg.fr
"""

import argparse
import os
import json
import random
import sys
import time
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ivtmetrics
from dataloader import CholecT50

def evaluate_model(net, args, mode='val'):
    dataset = CholecT50(
                dataset_dir=args.data_dir, 
                dataset_variant=args.split,
                test_fold=args.fold,
                augmentation_list=['original'],
                normalize=True,
                m=args.m
        )
    _, val_dataset, test_dataset = dataset.build()    

    if mode == 'val':
        ds = val_dataset
        records = dataset.val_records
    else:
        ds = test_dataset
        records = dataset.test_records

    # print("records to use >> ", records)

    loaders = []
    for i, video_ds in enumerate(ds):
        loader = DataLoader(
                        video_ds, 
                        batch_size=args.bs, 
                        num_workers=args.nw, 
                        shuffle=False, 
                        pin_memory=True, 
                        prefetch_factor=4*args.bs, 
                        persistent_workers=True
                    )
        loaders.append((records[i], loader))

    rec = ivtmetrics.Recognition(100)
    rec.reset_global()
    
    print('Eval RDV split:     AP_i  |  AP_v  |  AP_t  | AP_ivt', file=open(args.logfile, 'a+'))
    print('-'*52, file=open(args.logfile, 'a+'))

    print(f"Starting evaluation...")
    for vid, loader in loaders:
        print(f"Processing video {vid} of length: {len(loader.dataset)}")        
        with torch.no_grad():       
            for i, (frames, y_i, y_v, y_t, y_ivt) in enumerate(loader): 
                net.eval()
                y_ivt  = y_ivt.squeeze(1).cuda()

                b, m, c, h, w = frames.size()
                frames = frames.view(-1, c, h, w).cuda()      
                (enc_i_cam, logit_i), (enc_v_cam, logit_v), (enc_t_cam, logit_t), logit_ivt = net(frames)
                logit_ivt = logit_ivt.view(b, m, -1)[:, -1, :]
                preds = torch.sigmoid(logit_ivt).detach().cpu()

                rec.update(y_ivt.float().detach().cpu(), preds)

            ap_i   = rec.compute_AP('i')['mAP']
            ap_v   = rec.compute_AP('v')['mAP']
            ap_t   = rec.compute_AP('t')['mAP']
            ap_ivt = rec.compute_AP('ivt')['mAP']

            print(f'Video # {vid} AP : {ap_i:.4f} | {ap_v:.4f} | {ap_t:.4f} | {ap_ivt:.4f}', file=open(args.logfile, 'a+'))
            rec.video_end()

    # compute the final mAP for all the test videos
    imAP   = rec.compute_video_AP('i')['mAP']
    vmAP   = rec.compute_video_AP('v')['mAP']
    tmAP   = rec.compute_video_AP('t')['mAP']
    ivmAP  = rec.compute_video_AP('iv')['mAP']
    itmAP  = rec.compute_video_AP('it')['mAP']
    ivtmAP = rec.compute_video_AP('ivt')['mAP']
    ivt_ap = rec.compute_video_AP('ivt')['AP']

    # topk values
    itopk   = rec.topK(args.topK, 'i')
    ttopk   = rec.topK(args.topK, 't')
    vtopk   = rec.topK(args.topK, 'v')
    ivttopk = rec.topK(args.topK, 'ivt')

    print(f"MODE: {mode} || mAP ==> tool: {round(imAP,5)} || verb: {round(vmAP,5)} || target: {round(tmAP,5)} || triplet: {round(ivtmAP,5)} || iv: {round(ivmAP,5)} || it: {round(itmAP,5)} ----", file=open(args.logfile, 'a+'))
    print(f"MODE: {mode} || topK ==> tool: {round(itopk,5)} || verb: {round(vtopk,5)} || target: {round(ttopk,5)} || triplet: {round(ivttopk,5)} ----", file=open(args.logfile, 'a+'))

    # NOTE: class wise verb mAP
    # print("verb class wise AP > ", rec.compute_video_AP('v')['AP'], file=open(args.logfile, 'a+'))

    # print("mAP of the triplet >>> ", ivtmAP)
    print(f"MODE: {mode} || mAP ==> tool: {round(imAP,5)} || verb: {round(vmAP,5)} || target: {round(tmAP,5)} || triplet: {round(ivtmAP,5)} || iv: {round(ivmAP,5)} || it: {round(itmAP,5)} ----")

    return {'tool_mAP':imAP, 'verb_mAP':vmAP, 'target_mAP':tmAP, 'triplet_mAP':ivtmAP }