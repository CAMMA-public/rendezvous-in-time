"""
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France
Website: http://camma.u-strasbg.fr
"""

import os
import numpy as np
from typing import Tuple
from collections import OrderedDict
from einops import repeat, rearrange

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as basemodels

OUT_HEIGHT = 8
OUT_WIDTH  = 14

#%% Rendezvous-in-time (RiT)
class RiT(nn.Module):
    def __init__(self, basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, hr_output=False, use_ln=False, m=3):
        super(RiT, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output, m=m)
        self.decoder = Decoder(layer_size, d_model, num_heads, num_triplet, use_ln=use_ln, m=m)     

    def forward(self, inputs):
        enc_i, enc_v, enc_t, enc_ivt = self.encoder(inputs)
        dec_ivt = self.decoder(enc_i, enc_v, enc_t, enc_ivt)
        return enc_i, enc_v, enc_t, dec_ivt

#%% Triplet Components Feature Encoder
class Encoder(nn.Module):
    def __init__(self, basename='resnet18', num_tool=6,  num_verb=10, num_target=15, num_triplet=100, enctype='cagam', is_bottleneck=True, hr_output=False, m=3):
        super(Encoder, self).__init__()
        depth = 64 if basename == 'resnet18' else 128
        def Ignore(x): return None
        self.basemodel  = BaseModel(basename, hr_output)
        self.wsl        = WSL(num_tool, depth)
        self.cagam      = CAG(num_tool, num_verb, num_target) if enctype=='cag' else CAGAM(num_tool, num_verb, num_target, m=m)
        self.bottleneck = Bottleneck(num_triplet) if is_bottleneck else Ignore
        
    def forward(self, x):
        high_x, low_x = self.basemodel(x)
        enc_i         = self.wsl(high_x)
        enc_v, enc_t  = self.cagam(high_x, enc_i[0])
        enc_ivt       = self.bottleneck(low_x)
        return enc_i, enc_v, enc_t, enc_ivt

#%% MultiHead Attention Decoder
class Decoder(nn.Module):
    def __init__(self, layer_size, d_model, num_heads, num_class=100, use_ln=False, m=3):
        super(Decoder, self).__init__()        
        self.projection = nn.ModuleList([Projection(num_triplet=num_class, out_depth=d_model) for i in range(layer_size)])
        self.mhma       = nn.ModuleList([MHMA(num_class=num_class, depth=d_model, num_heads=num_heads, use_ln=use_ln) for i in range(layer_size)])
        self.ffnet      = nn.ModuleList([FFN(k=layer_size-i-1, num_class=num_class, use_ln=use_ln) for i in range(layer_size)])
        self.classifier = Classifier(num_class, m=m)
    
    def forward(self, enc_i, enc_v, enc_t, enc_ivt):
        X = enc_ivt.clone()
        for P, M, F in zip(self.projection, self.mhma, self.ffnet):
            X = P(enc_i[0], enc_v[0], enc_t[0], X)
            X = M(X)
            X = F(X)
        logits = self.classifier(X)
        return logits
 
#%% Feature extraction backbone
class BaseModel(nn.Module):
    def __init__(self, basename='resnet18', hr_output=False, *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {} 
        if basename == 'resnet18':
            self.basemodel      = basemodels.resnet18(pretrained=True)     
            if hr_output: self.increase_resolution()
            self.basemodel.layer1[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))        
        if basename == 'resnet50':
            self.basemodel      = basemodels.resnet50(pretrained=True) 
            # print(self.basemodel)
            self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
        
    def increase_resolution(self):  
        global OUT_HEIGHT, OUT_WIDTH  
        self.basemodel.layer3[0].conv1.stride = (1,1)
        self.basemodel.layer3[0].downsample[0].stride=(1,1)  
        self.basemodel.layer4[0].conv1.stride = (1,1)
        self.basemodel.layer4[0].downsample[0].stride=(1,1)
        OUT_HEIGHT *= 4
        OUT_WIDTH  *= 4
        print("using high resolution output ({}x{})".format(OUT_HEIGHT,OUT_WIDTH))
        

    def get_activation(self, layer_name):
        def hook(module, input: Tuple[torch.Tensor], output:torch.Tensor):
            self.output_feature[layer_name] = output
        return hook
    
    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']

#%% Weakly-Supervised localization
class WSL(nn.Module):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=depth, kernel_size=3, padding=1)
        self.cam   = nn.Conv2d(in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn    = nn.BatchNorm2d(depth)
        self.gmp   = nn.AdaptiveMaxPool2d((1,1))
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam     = self.cam(feature)
        logits  = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits

#%% Unfiltered Bottleneck layer
class Bottleneck(nn.Module):
    def __init__(self, num_class):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, stride=(2,2), kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=1)
        self.elu   = nn.ELU()
        self.bn1   = nn.BatchNorm2d(256)
        self.bn2   = nn.BatchNorm2d(num_class)
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.elu(feature)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.elu(feature)
        return feature


#%% Class Activation Guided Temporal Attention Mechanism (CAGTAM)
# TODO: change name to CAGTAM
class CAGAM(nn.Module):    
    def __init__(self, num_tool, num_verb, num_target, in_depth=512, m=3):
        super(CAGAM, self).__init__()        
        out_depth               = num_tool        
        self.verb_context       = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)        
        self.verb_query         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_key           = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key      = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.verb_cmap          = nn.Conv2d(in_channels=out_depth, out_channels=num_verb, kernel_size=1)       
        self.target_context     = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)     
        self.target_query       = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_query  = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_key         = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_key    = nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=1)        
        self.target_cmap        = nn.Conv2d(in_channels=out_depth, out_channels=num_target, kernel_size=1)        
        self.gmp       = nn.AdaptiveMaxPool2d((1,1))
        self.elu       = nn.ELU()    
        self.soft      = nn.Softmax(dim=1)    
        self.flat      = nn.Flatten(2,3)  
        self.bn1       = nn.BatchNorm2d(out_depth)
        self.bn2       = nn.BatchNorm2d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm2d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm2d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm2d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth)
        self.bn10      = nn.BatchNorm2d(out_depth) 
        self.bn11      = nn.BatchNorm2d(out_depth) 
        self.bn12      = nn.BatchNorm2d(out_depth)        
        self.encoder_cagam_verb_beta   = torch.nn.Parameter(torch.randn(1))
        self.encoder_cagam_target_beta = torch.nn.Parameter(torch.randn(1))
        self.m         = m
        self.bn13      = nn.BatchNorm2d(self.m+1)
        self.gate      = nn.Conv2d(self.m+1, self.m+1, kernel_size=1) if self.m>0 else nn.Identity() 
        self.fc1       = nn.Linear(num_verb, 1)
        self.fc2       = nn.Linear(num_target, 1)

        # NOTE: Temporal Attention Module (TAM)
        self.cmap_attn4 = TAM(in_channels=num_verb, m=self.m, k=1)
        self.cmap_attn5 = TAM(in_channels=num_verb, m=self.m, k=1)
        self.cbn4       = nn.BatchNorm2d(num_verb)

    def get_verb(self, raw, cam):
        x  = self.elu(self.bn1(self.verb_context(raw)))

        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn2(self.verb_query(x)))
        k1 = self.elu(self.bn3(self.verb_key(x)))
        
        w1 = self.flat(k1).matmul(self.flat(q1).transpose(-1,-2))

        q2 = self.elu(self.bn4(self.verb_tool_query(cam)))
        k2 = self.elu(self.bn5(self.verb_tool_key(cam)))
        w2 = self.flat(k2).matmul(self.flat(q2).transpose(-1,-2))
        
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         

        v = self.flat(z)
        e = (attention.matmul(v) * self.encoder_cagam_verb_beta).reshape(sh)
        e = self.bn6(e + z)

        cmap = self.verb_cmap(e) 

        # NOTE: TAM in Late Fusion mode 
        cmap = self.cmap_attn4(cmap)   
        cmap = self.cbn4(cmap)
        cmap = self.elu(cmap)
        
        cmap = self.cmap_attn5(cmap)  

        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        
        return cmap, y  
    
    def get_target(self, raw, cam):
        x  = self.elu(self.bn7(self.target_context(raw)))
        z  = x.clone()
        sh = list(z.shape)
        sh[0] = -1        
        q1 = self.elu(self.bn8(self.target_query(x)))
        k1 = self.elu(self.bn9(self.target_key(x)))
        w1 = self.flat(k1).transpose(-1,-2).matmul(self.flat(q1))   

        q2 = self.elu(self.bn10(self.target_tool_query(cam)))
        k2 = self.elu(self.bn11(self.target_tool_key(cam)))
        w2 = self.flat(k2).transpose(-1,-2).matmul(self.flat(q2))    
        attention = (w1 * w2) / torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)         
        v = self.flat(z)
        e = (v.matmul(attention) * self.encoder_cagam_target_beta).reshape(sh)
        e = self.bn12(e + z)
        cmap = self.target_cmap(e)

        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y                
            
    def forward(self, x, cam):
        cam_v, logit_v = self.get_verb(x, cam)
        cam_t, logit_t = self.get_target(x, cam)
        return (cam_v, logit_v), (cam_t, logit_t)

class TAM(nn.Module):
    """
    Constructs proposed Temporal Attention Module (TAM)
    """
    def __init__(self, in_channels=10, m=3, k=1):
        super(TAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(in_channels, 1, kernel_size=k, bias=False) 
        self.act      = nn.Sigmoid()
        self.m        = m
        self.bn1      = nn.BatchNorm1d(1)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).squeeze()

        # reshape to include time dimension
        B, c = y.shape # 16x10
        b = int(B/(self.m + 1)) # 4
        y = y.reshape([b, self.m+1, c]) # 4 x 4 x 10

        # apply conv to 1 dim value
        y = self.bn1(self.conv(y.transpose(-1, -2))).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.act(y)

        B, c, h, w = x.shape
        x = x.reshape([b, self.m+1, c, h, w])
        past_cmap, curr_cmap = torch.split(x, [self.m, 1], 1)
        x = x * y.unsqueeze(-1).unsqueeze(-1)
        x = x.sum(dim=1).unsqueeze(1)
        x = torch.cat([past_cmap, x], dim=1).view(b*(self.m+1), c, h, w)

        return x

#%% Projection function
class Projection(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128):
        super(Projection, self).__init__()
        self.ivt_value = nn.Conv2d(in_channels=num_triplet, out_channels=out_depth, kernel_size=1)
        self.i_value   = nn.Conv2d(in_channels=num_tool, out_channels=out_depth, kernel_size=1)        
        self.v_value   = nn.Conv2d(in_channels=num_verb, out_channels=out_depth, kernel_size=1)
        self.t_value   = nn.Conv2d(in_channels=num_target, out_channels=out_depth, kernel_size=1)       
        self.ivt_query = nn.Linear(in_features=num_triplet, out_features=out_depth)
        self.dropout   = nn.Dropout(p=0.3)
        self.ivt_key   = nn.Linear(in_features=num_triplet, out_features=out_depth)
        self.i_key     = nn.Linear(in_features=num_tool, out_features=out_depth)
        self.v_key     = nn.Linear(in_features=num_verb, out_features=out_depth)
        self.t_key     = nn.Linear(in_features=num_target, out_features=out_depth)
        self.gap       = nn.AdaptiveAvgPool2d((1,1))
        self.elu       = nn.ELU()          
        self.bn1       = nn.BatchNorm1d(out_depth)
        self.bn2       = nn.BatchNorm1d(out_depth)
        self.bn3       = nn.BatchNorm2d(out_depth)
        self.bn4       = nn.BatchNorm1d(out_depth)
        self.bn5       = nn.BatchNorm2d(out_depth)
        self.bn6       = nn.BatchNorm1d(out_depth)
        self.bn7       = nn.BatchNorm2d(out_depth)
        self.bn8       = nn.BatchNorm1d(out_depth)
        self.bn9       = nn.BatchNorm2d(out_depth) 
        
    def forward(self, cam_i, cam_v, cam_t, X):  
        q = self.elu(self.bn1(self.ivt_query(self.dropout(self.gap(X).squeeze(-1).squeeze(-1)))))  
        k = self.elu(self.bn2(self.ivt_key(self.gap(X).squeeze(-1).squeeze(-1))) )
        v = self.bn3(self.ivt_value(X)) 
        k1 = self.elu(self.bn4(self.i_key(self.gap(cam_i).squeeze(-1).squeeze(-1))) )
        v1 = self.elu(self.bn5(self.i_value(cam_i)) )
        k2 = self.elu(self.bn6(self.v_key(self.gap(cam_v).squeeze(-1).squeeze(-1))))
        v2 = self.elu(self.bn7(self.v_value(cam_v)) )
        k3 = self.elu(self.bn8(self.t_key(self.gap(cam_t).squeeze(-1).squeeze(-1))))
        v3 = self.elu(self.bn9(self.t_value(cam_t)))
        sh = list(v1.shape)
        v  = self.elu(F.interpolate(v, (sh[2],sh[3])))
        X  = self.elu(F.interpolate(X, (sh[2],sh[3])))
        return (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v))

 
#%% Multi-head of self and cross attention
class MHMA(nn.Module):
    def __init__(self, depth, num_class=100, num_heads=4, use_ln=False):
        super(MHMA, self).__init__()        
        self.concat = nn.Conv2d(in_channels=depth*num_heads, out_channels=num_class, kernel_size=3, padding=1)
        self.bn     = nn.BatchNorm2d(num_class)
        self.ln     = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        self.elu    = nn.ELU()    
        self.soft   = nn.Softmax(dim=1) 
        self.heads  = num_heads
        
    def scale_dot_product(self, key, value, query):
        dk        = torch.sqrt(torch.tensor(list(key.shape)[-2], dtype=torch.float32))
        affinity  = key.matmul(query.transpose(-1,-2))                        
        attn_w    = affinity / dk              
        attn_w    = self.soft(attn_w)
        attention = attn_w.matmul(value) 
        return attention
    
    def forward(self, inputs):
        (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v)) = inputs     
        query = torch.stack([q]*self.heads, dim=1) # [B,Head,D]
        query = query.unsqueeze(dim=-1) # [B,Head,D,1]        
        key   = torch.stack([k,k1,k2,k3], dim=1) # [B,Head,D]
        key   = key.unsqueeze(dim=-1) # [B,Head,D,1]        
        value = torch.stack([v,v1,v2,v3], dim=1) # [B,Head,D,H,W]
        dims  = list(value.shape) # [B,Head,D,H,W]
        value = value.reshape([-1,dims[1],dims[2],dims[3]*dims[4]])# [B,Head,D,HW]          
        attn  = self.scale_dot_product(key, value, query)  # [B,Head,D,HW]
        attn  = attn.reshape([-1,dims[1]*dims[2],dims[3],dims[4]]) # [B,DHead,H,W]
        mha   = self.elu(self.bn(self.concat(attn)))
        mha   = self.ln(mha + X.clone())  
        return mha
 
#%% Feed-forward layer
class FFN(nn.Module):
    def __init__(self, k, num_class=100, use_ln=False):
        super(FFN, self).__init__()
        def Ignore(x): return x
        self.conv1 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_class, out_channels=num_class, kernel_size=1) 
        self.elu1  = nn.ELU() 
        self.elu2  = nn.ELU() if k>0 else Ignore     
        self.bn1   = nn.BatchNorm2d(num_class)    
        self.bn2   = nn.BatchNorm2d(num_class)
        self.ln    = nn.LayerNorm([num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        
    def forward(self, inputs,):
        x  = self.elu1(self.bn1(self.conv1(inputs)))
        x  = self.elu2(self.bn2(self.conv2(x)))
        x  = self.ln(x + inputs.clone())
        return x

#%% Classification layer
class Classifier(nn.Module):
    def __init__(self, layer_size, num_class=100, m=3):
        super(Classifier, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Linear(in_features=num_class, out_features=num_class)
        
    def forward(self, inputs):
        x = self.gmp(inputs).squeeze(-1).squeeze(-1)
        y = self.mlp(x)
        return y

