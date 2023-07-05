"""
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France
Website: http://camma.u-strasbg.fr
"""

import argparse
import os
import random
import sys
import time
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from pprint import pprint, pformat

import torch 
import torch.nn as nn
import torch.nn.functional as F

import ivtmetrics

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1.):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_acc(self, val, n=1.):
        self.val = val/n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# get video list
def get_video_list(args):
    video_split  = split_selector(args.split)
    train_videos = sum([v for k,v in video_split.items() if k!=args.fold], []) if 'crossval' in args.split else video_split['train']
    test_videos  = sum([v for k,v in video_split.items() if k==args.fold], []) if 'crossval' in args.split else video_split['test']
    
    if 'crossval' in args.split:
        val_videos   = train_videos[-5:]
        train_videos = train_videos[:-5]
    else:
        val_videos = video_split['val']

    train_videos = sorted(train_videos)
    val_videos   = sorted(val_videos)
    test_videos  = sorted(test_videos)

    # in records format
    train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
    val_records   = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
    test_records  = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]

    return train_records, val_records, test_records

#%% helper functions
def get_weight_balancing(case='cholect50'):
    switcher = {
        'cholect50': {
            'tool'  :   [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
            'verb'  :   [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636, 12.13692946, 0.40462028],
            'target':   [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446, 0.285756500, 12.72368421, 0.6250808,  3.85771277, 6.95683453, 0.84923888, 0.40130032]        
        },
        'cholect50-challenge': {
            'tool':     [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
            'verb':     [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 1.28428410, 7.35822511, 18.67857143, 0.45704490],
            'target':   [0.07333818, 0.87139287, 0.42853950, 1.00000000, 17.67281106, 13.94545455, 1.44880997, 6.04889590, 0.326188650, 16.82017544, 0.63577586, 6.79964539, 6.19547658, 0.96284208, 0.51559559]
        },
        'cholect45-crossval': {
            1: {
                'tool':     [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                'verb':     [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889, 17.07784431, 0.45241117],
                'target':   [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319, 0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
                },
            2: {
                'tool':     [0.0854156, 0.89535362, 0.10995253, 2.74936869, 1.78264429, 1.13234529],
                'verb':     [0.36346863, 0.06771776, 0.07893261, 0.82842725, 1.33892161, 2.13049748, 1.26120359, 5.72674419, 19.7, 0.43189126],
                'target':   [0.07530655, 0.97961957, 0.4325135, 0.99393438, 15.5387931, 14.5951417, 1.53862569, 6.01836394, 0.35184462, 15.81140351, 0.709506, 5.79581994, 8.08295964, 1.0, 0.52689272]
            },
            3: {
                "tool" :   [0.0915228, 0.89714969, 0.12057004, 2.72128174, 1.94092281, 1.12948557],
                "verb" :   [0.43636862, 0.07558554, 0.0891017, 0.81820519, 1.53645582, 2.31924198, 1.28565657, 6.49387755, 18.28735632, 0.48676763],
                "target" : [0.06841828, 0.90980736, 0.38826607, 1.0, 14.3640553, 12.9875, 1.25939394, 5.38341969, 0.29060227, 13.67105263, 0.59168565, 6.58985201, 5.72977941, 0.86824513, 0.47682423]
            },
            4: {
                'tool':     [0.08222218, 0.85414117, 0.10948695, 2.50868784, 1.63235867, 1.20593318],
                'verb':     [0.41154261, 0.0692142, 0.08427214, 0.79895288, 1.33625219, 2.2624166, 1.35343681, 7.63, 17.84795322, 0.43970609],
                'target':   [0.07536126, 0.85398445, 0.4085784, 0.95464422, 15.90497738, 18.5978836, 1.55875831, 5.52672956, 0.33700863, 15.41666667, 0.74755423, 5.4921875, 6.11304348, 1.0, 0.50641118],
            },
            5: {
                'tool':     [0.0804654, 0.92271157, 0.10489631, 2.52302243, 1.60074906, 1.09141982],
                'verb':     [0.50710436, 0.06590258, 0.07981184, 0.81538866, 1.29267277, 2.20525568, 1.29699248, 7.32311321, 25.45081967, 0.46733895],
                'target':   [0.07119395, 0.87450495, 0.43043372, 0.86465981, 14.01984127, 23.7114094, 1.47577277, 5.81085526, 0.32129865, 22.79354839, 0.63304067, 6.92745098, 5.88833333, 1.0, 0.53175798]
            }
        },
        'cholect50-crossval': {
            1:{
                'tool':     [0.0828851, 0.8876, 0.10830995, 2.93907285, 1.63884786, 1.14499484],
                'verb':     [0.29628942, 0.07366916, 0.08267971, 0.83155428, 1.25402434, 2.38358209, 1.34938741, 7.56872038, 12.98373984, 0.41502079],
                'target':   [0.06551745, 1.0, 0.36345711, 0.82434783, 13.06299213, 8.61818182, 1.4017744, 4.62116992, 0.32822238, 45.45205479, 0.67343211, 4.13200498, 8.23325062, 0.88527215, 0.43113306],

            },
            2:{
                'tool':     [0.08586283, 0.87716737, 0.11068887, 2.84210526, 1.81016949, 1.16283571],
                'verb':     [0.30072757, 0.07275414, 0.08350168, 0.80694143, 1.39209979, 2.22754491, 1.31448763, 6.38931298, 13.89211618, 0.39397505],
                'target':   [0.07056703, 1.0, 0.39451115, 0.91977006, 15.86206897, 9.68421053, 1.44483706, 5.44378698, 0.31858714, 16.14035088, 0.7238395, 4.20571429, 7.98264642, 0.91360477, 0.43304307],
            },
            3:{
                'tool':     [0.09225068, 0.87856006, 0.12195811, 2.82669323, 1.97710987, 1.1603972],
                'verb':     [0.34285159, 0.08049804, 0.0928239, 0.80685714, 1.56125608, 2.23984772, 1.31471136, 7.08835341, 12.17241379, 0.43180428],
                'target':   [0.06919395, 1.0, 0.37532866, 0.9830703, 15.78801843, 8.99212598, 1.27597765, 5.36990596, 0.29177312, 15.02631579, 0.64935557, 5.08308605, 5.86643836, 0.86580743, 0.41908257], 
            },
            4:{
                'tool':     [0.08247885, 0.83095539, 0.11050268, 2.58193042, 1.64497676, 1.25538881],
                'verb':     [0.31890981, 0.07380354, 0.08804592, 0.79094077, 1.35928144, 2.17017208, 1.42947103, 8.34558824, 13.19767442, 0.40666428],
                'target':   [0.07777646, 0.95894072, 0.41993829, 0.95592153, 17.85972851, 12.49050633, 1.65701092, 5.74526929, 0.33763901, 17.31140351, 0.83747083, 3.95490982, 6.57833333, 1.0, 0.47139615],
            },
            5:{
                'tool':     [0.07891691, 0.89878025, 0.10267677, 2.53805556, 1.60636428, 1.12691169],
                'verb':     [0.36420961, 0.06825313, 0.08060635, 0.80956984, 1.30757221, 2.09375, 1.33625848, 7.9009434, 14.1350211, 0.41429631],
                'target':   [0.07300329, 0.97128713, 0.42084942, 0.8829883, 15.57142857, 19.42574257, 1.56521739, 5.86547085, 0.32732733, 25.31612903, 0.70171674, 4.55220418, 6.13125, 1.0, 0.48528321],
            }
        }
    }
    return switcher.get(case)

def split_selector(case='cholect50'):
    switcher = {
        'cholect50': {
            'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
            'val':   [8, 12, 29, 50, 78],
            'test':  [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
        },
        'cholect50-challenge': {
            'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
            'val':   [5, 18, 36, 65, 74],
            'test':  [92, 96, 103, 110, 111]
        },
        'cholect45-crossval': {
            1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
            2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
            3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
            4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
            5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
        },
        'cholect50-crossval': {
            1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
            2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
            3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
            4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
            5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
        },
    }
    return switcher.get(case)

# get component wise class weights 
def get_component_weights(args):
    if "crossval" in args.split:    
        wt_dict = get_weight_balancing(case=args.split)[args.fold]
    else:
        wt_dict = get_weight_balancing(case=args.split)

    return wt_dict['tool'], wt_dict['verb'], wt_dict['target']

# binary cross entropy loss
def bce_loss(preds, gt, pos_wt=None):
    wt = torch.tensor(pos_wt, device=gt.device) if pos_wt != None else None
    return F.binary_cross_entropy_with_logits(preds, gt, pos_weight=wt) 

def freeze_net(net, exclude_options=["decoder"]):
    count_grad_vars = 0
    for k,v in net.named_parameters():
        if any([j in k for j in exclude_options]):
            count_grad_vars += 1
            continue
        v.requires_grad = False

def display_net_params(net, show_grad=False):
    for i,j in net.named_parameters():
        if show_grad:
            print(f"{i} >>  {j.requires_grad}")
        else:
            print(i)

# load model weights and test
def load_model_weights(net, checkpoint_name, skip_module="decoder"):
    net_dict = net.state_dict()
    checkpoint = torch.load(checkpoint_name)
    if 'model' in checkpoint: state_dict = checkpoint['model'] 
    else: state_dict = checkpoint

    state_dict_new = OrderedDict({})

    for k,v in state_dict.items():

        state_dict_new[k] = state_dict[k]

    net_dict.update(state_dict_new)
    net.load_state_dict(net_dict)
    print("Pretrained model loading is successful!")

# count params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)