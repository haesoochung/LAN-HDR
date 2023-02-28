import torch
import torch.nn as nn
import torch.nn.functional as Fn
import os 
import sys
# from torchvision import models
from config import ROOT_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from models.layers import *


class Single_sup_T_wo_se(nn.Module):
    def __init__(self, num_res_blocks, n_feats, T_dim):
        super(Single_sup_T_wo_se, self).__init__()
        self.num_res_blocks = num_res_blocks 
        self.n_feats = n_feats

        self.conv_head = conv3x3(2*T_dim, T_dim)
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=T_dim, out_channels=T_dim))
        self.conv_tail = conv3x3(T_dim, T_dim//2)

    def forward(self, S, T, sup):
        F = torch.cat((T, sup), dim=1)
        F = self.conv_head(F)
        F = F * S
        for i in range(self.num_res_blocks):
            F = self.RBs[i](F)
        F = self.conv_tail(F)

        return F

class Recon_swap_once(nn.Module):
    def __init__(self, num_res_blocks, n_feats, T_dim, args):
        super(Recon_swap_once, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.n_feats = n_feats
        self.scale = [1,2,4,8]
        self.s3 = Single_sup_T_wo_se(num_res_blocks[0], n_feats, self.scale[3]*T_dim)
        self.up_8_4 = UpsampleBlock(self.scale[2]*T_dim, self.scale[2]*T_dim, self.scale[1]*T_dim) 
        self.up_4_2 = UpsampleBlock(self.scale[1]*T_dim, self.scale[1]*T_dim, self.scale[0]*T_dim) 
        self.up_2_1 = UpsampleBlock(self.scale[0]*T_dim, self.scale[0]*T_dim, n_feats) 
        
        self.conv_final1 = conv3x3(n_feats, n_feats)

    def forward(self, x3, x2, x1, x0, S=None, T_lv3=None):
        x = self.s3(S, T_lv3, x3)
        x = self.up_8_4(x2, x)
        x = self.up_4_2(x1, x)
        x = self.up_2_1(x0, x)
        x = Fn.relu(self.conv_final1(x))
        return x
