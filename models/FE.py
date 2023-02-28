import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

from models.layers import *

class FE_res_fft(torch.nn.Module):
    def __init__(self, n_feats, args, indim=6):
        super(FE_res_fft, self).__init__()
        
        self.conv11 = conv3x3(indim, n_feats)
        self.conv12 = conv3x3(n_feats, n_feats)
        self.block1 = ResBlock_fft_bench(n_feats)

        self.conv21 = conv3x3(n_feats, n_feats*2, 2)
        self.block2 = ResBlock_fft_bench(n_feats*2)

        self.conv31 = conv3x3(n_feats*2, n_feats*4, 2)
        self.block3 = ResBlock_fft_bench(n_feats*4)

        self.conv41 = conv3x3(n_feats*4, n_feats*8, 2)
        self.block4 = ResBlock_fft_bench(n_feats*8)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.block1(x)
        x_lv1 = x
        x = F.relu(self.conv21(x))
        x = (self.block2(x))
        x_lv2 = x
        x = F.relu(self.conv31(x))
        x = (self.block3(x))
        x_lv3 = x
        x = F.relu(self.conv41(x))
        x = (self.block4(x))
        x_lv4 = x
        return [x_lv4, x_lv3, x_lv2, x_lv1]

class FE_res_fft_gc(torch.nn.Module):
    def __init__(self, n_feats, args, indim=6):
        super(FE_res_fft_gc, self).__init__()
        
        self.conv11 = gated_conv(indim, n_feats)
        self.conv12 = gated_conv(n_feats, n_feats)
        self.block1 = ResBlock_fft_bench(n_feats)

        self.conv21 = gated_conv(n_feats, n_feats*2, stride = 2)
        self.block2 = ResBlock_fft_bench(n_feats*2)

        self.conv31 = gated_conv(n_feats*2, n_feats*4, stride = 2)
        self.block3 = ResBlock_fft_bench(n_feats*4)

        self.conv41 = gated_conv(n_feats*4, n_feats*8, stride = 2)
        self.block4 = ResBlock_fft_bench(n_feats*8)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.block1(x)
        x_lv1 = x
        x = F.relu(self.conv21(x))
        x = (self.block2(x))
        x_lv2 = x
        x = F.relu(self.conv31(x))
        x = (self.block3(x))
        x_lv3 = x
        x = F.relu(self.conv41(x))
        x = (self.block4(x))
        x_lv4 = x
        return [x_lv4, x_lv3, x_lv2, x_lv1]
