import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
# from torchvision import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from models.layers import *

class Transformer_swap_once(nn.Module):
    def __init__(self, args):
        super(Transformer_swap_once, self).__init__()
        self.T_dim = args.tdim
        self.embed = nn.Linear(self.T_dim*8*3*3, self.T_dim*4, bias=False)
        
    def gather(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W*s*s]
        # dim: scalar > 0
        # index: [N, H*W]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))] # N, 1, -1
        expanse = list(input.size()) # N, C*k*k, H*W*16
        expanse[0] = -1
        expanse[dim] = -1           # -1, C*k*k, -1
        index = index.view(views).expand(expanse) # [N, H*W] -> [N, 1, H*W] -> [N, C*k*k, H*W] 
        return torch.gather(input, dim, index) #[N, C*k*k, H*W] 

    def swap(self, V, idx, s, k):
        (h, w) = V.size()[-2:]
        temph = s*k+self.H*s-h-s; tempw = s*k+self.W*s-w-s
        ph = max(temph//2 + temph%2, 1); pw = max(tempw//2 + tempw%2, 1)     

        V_unfold = F.unfold(V, kernel_size=(s*k, s*k), padding=(ph, pw), stride=s) # [N, C*s*k*s*k, H*W] 
        T_unfold = self.gather(V_unfold, 2, idx) #[N, C*sk*sk, h*w] 
        divisor = torch.ones_like(T_unfold)  # for overlapping patches
        divisor = F.fold(divisor, output_size=(h, w), kernel_size=(s*k, s*k), padding=(ph, pw), stride=s)
        T = F.fold(T_unfold, output_size=(h, w), kernel_size=(s*k, s*k), padding=(ph, pw), stride=s) / divisor # [N, C, h*s, w*s]
        return T
        
    def forward(self, query, key, value, k=3): # / 1 2 4 8
        (_, _, self.H, self.W) = value.size()
        ### search
        query_unfold  = F.unfold(query, kernel_size=(k, k), padding=k//2) # [N, C, H, W] -> [N, C*k*k, H*W]
        key_unfold = F.unfold(key, kernel_size=(k, k), padding=k//2) # [N, C, H, W] -> [N, C*k*k, H*W]
        key_unfold = key_unfold.permute(0, 2, 1) # [N, H*W, C*k*k]
        key_unfold = self.embed(key_unfold) # [N, H*W, C']
        query_unfold = query_unfold.permute(0, 2, 1) # [N, H*W, C*k*k]
        query_unfold = self.embed(query_unfold) # [N, H*W, C']
        query_unfold = query_unfold.permute(0, 2, 1) # [N, C', H*W]
        
        # normalize
        key_unfold = F.normalize(key_unfold, dim=2) # [N, H*W, C*k*k]
        query_unfold  = F.normalize(query_unfold, dim=1) # [N, C*k*k, H*W]

        ### relevance embedding
        corr = torch.bmm(key_unfold, query_unfold) #[N, H*W, H*W]
        corr_value, corr_argmax = torch.max(corr, dim=1) #[N, H*W], [N, H*W] which Ref pixel is most correlatied with LR

        ### transfer
        T_lv3 = self.swap(V=value, idx=corr_argmax, s=1, k=k) # [N, C, H/8, W/8] <-> [N, C*k*k, M]
        S = corr_value.view(corr_value.size(0), 1, self.H, self.W) # [N, 1, H, W]

        return S, T_lv3
      