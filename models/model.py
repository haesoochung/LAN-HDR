import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from config import ROOT_DIR

from models.layers import *
from models import FE, Transformer, Recon

import kornia.augmentation as K
ColorJitter = K.ColorJitter(0.3, 0.3, 0.2, 0.2, p=1.)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.mid = args.nframes//2
        
        self.align = Align_sup_00(args)        
        self.merge = Merge_test_res_fft(args)

    def forward(self, inputs, input_ys, input_exp_ys, hdr=None):
        aligned_0, gate0 = self.align(inputs[0], input_ys[0], inputs[self.mid], input_exp_ys[0])
        aligned_1, gate1 = self.align(inputs[1], input_ys[1], inputs[self.mid], input_exp_ys[1])
        aligned_2, gate2 = self.align(inputs[2], input_ys[2], inputs[self.mid], input_exp_ys[2])
        aligned_3, gate3 = self.align(inputs[3], input_ys[3], inputs[self.mid], input_exp_ys[3])
        aligned_4, gate4 = self.align(inputs[4], input_ys[4], inputs[self.mid], input_exp_ys[4])
        output = self.merge([aligned_0, aligned_1, aligned_2, aligned_3, aligned_4])

        return output, [gate0, gate1, gate2, gate3, gate4]
        
class Align_sup_00(nn.Module):
    def __init__(self, args):
        super(Align_sup_00, self).__init__()
        self.num_res_blocks = [5,5,5,5]
        self.T_dim = args.tdim
        self.n_feats = args.ch
        self.FE_KQ = FE.FE_res_fft(n_feats=self.T_dim, args=args, indim=4) # ch * 1 2 4 8
        self.FE_V = FE.FE_res_fft(n_feats=self.T_dim, args=args) # ch * 1 2 4 8
        
        self.Transformer = Transformer.Transformer_swap_once(args=args)
        self.Recon = Recon.Recon_swap_once(num_res_blocks=self.num_res_blocks, n_feats=self.n_feats, T_dim=self.T_dim, args=args)

        self.mov_extractor = FE.FE_res_fft_gc(n_feats=self.T_dim, args=args, indim=7) # ch * 1 2 4 8
        self.ref_extractor = FE.FE_res_fft_gc(n_feats=self.T_dim, args=args, indim=7) # ch * 1 2 4 8

        self.Fuser = Fuser(n_feats=self.n_feats, T_dim=args.tdim, args=args)

    def forward(self, mov_, mov_y, ref_, ref_y):
        mov, mov_mask = torch.split(mov_, [6,1], 1)
        ref, ref_mask = torch.split(ref_, [6,1], 1)
        mov_features = self.mov_extractor(mov_)
        ref_features = self.ref_extractor(ref_)
        _, _, h, w = mov_features[1].shape
        mov = F.interpolate(mov, size = (h,w), mode='bilinear', align_corners=False)
        mov_y = F.interpolate(mov_y, size = (h,w), mode='bilinear', align_corners=False)
        ref_y = F.interpolate(ref_y, size = (h,w), mode='bilinear', align_corners=False)
        movs = self.FE_KQ(mov_y.detach())
        [mov_lv0, mov_lv1, mov_lv2, mov_lv3] = self.FE_V(mov.detach())        
        refs = self.FE_KQ(ref_y.detach())

        S, T = self.Transformer(query=refs[0], key=movs[0], value=mov_lv0)
        aligned = self.Recon(mov_lv0, mov_lv1, mov_lv2, mov_lv3, S, T)
        out, gate = self.Fuser(torch.cat([mov_features[0], ref_features[0]], 1), mov_features[1]+ref_features[1], aligned, mov_features[2]+ref_features[2], mov_features[3]+ref_features[3])
        return out, gate

class Fuser(nn.Module):
    def __init__(self, n_feats, T_dim, args):
        super(Fuser, self).__init__()
        self.num_res_blocks = 2
        self.scale = [1,2,4,8]

        self.RB_head = ResBlock(in_channels=2*self.scale[3]*T_dim, out_channels=self.scale[3]*T_dim)
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=self.scale[3]*T_dim, out_channels=self.scale[3]*T_dim))
        self.conv_tail = conv3x3(self.scale[3]*T_dim, self.scale[2]*T_dim)
        self.up_8_4 = UpsampleBlockv2(self.scale[2]*T_dim, self.scale[2]*T_dim, self.scale[1]*T_dim) # 1/8 -> 1/4
        self.conv_bn = conv1x1(self.scale[1]*T_dim, n_feats)
        self.conv_gate = conv5x5(2*n_feats, 1)
        self.fb = ResBlock(n_feats, self.scale[1]*T_dim)

        self.up_4_2 = UpsampleBlockv2(self.scale[1]*T_dim, self.scale[1]*T_dim, self.scale[0]*T_dim) # 1/8 -> 1/4
        self.up_2_1 = UpsampleBlockv2(self.scale[0]*T_dim, self.scale[0]*T_dim, n_feats) # 1/8 -> 1/4

    def forward(self, F, x2, a, x1, x0):
        F = self.RB_head(F)
        for i in range(self.num_res_blocks):
            F = self.RBs[i](F)
        x = self.conv_tail(F)
        x = self.up_8_4(x, x2)
        x = self.conv_bn(x)
        gate = nn.functional.sigmoid(self.conv_gate(torch.cat([x, a],1)))
        x = gate*(x+a) + (1-gate)*x
        x = self.fb(x)
        x = self.up_4_2(x, x1)
        x = self.up_2_1(x, x0)

        return x, gate

class Merge_test_res_fft(nn.Module):
    def __init__(self, args, num_res_blocks=5):
        super(Merge_test_res_fft, self).__init__()
        self.n_feats = args.ch
        self.num_res_blocks = num_res_blocks
        self.conv1_head = conv3x3(self.n_feats*args.nframes, self.n_feats)
        self.RB1s = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB1s.append(ResBlock_fft_bench(self.n_feats))
        self.conv1_tail = conv3x3(self.n_feats, 3)

    def forward(self, feats):
        x = [k for k in feats]
        x = torch.cat(x, 1)
        x = F.relu(self.conv1_head(x))
        for i in range(self.num_res_blocks):
            x = self.RB1s[i](x)
        x = self.conv1_tail(x)
        x = F.sigmoid(x)
        return x      