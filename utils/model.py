import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
# from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from config import ROOT_DIR, parse_args

from utils.layer import *

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        # gen=utils.import_module_from_file(os.path.join(ROOT_DIR, 'utils/model.py'))
        # Generator_Fine = getattr(gen, "Generator_Fine_"+config.refine_arch)
        self.align_low = Align(config)
        self.align_mid = Align(config)
        self.align_high = Align(config)
        # self.merge = Merge(config)

    def forward(self, low, mid, high, hdr=None):
        aligned_low = self.align_low(low, mid)
        aligned_mid = self.align_mid(mid, mid)
        aligned_high = self.align_high(high, mid)

        # output = self.merge(aligned_low, aligned_mid, aligned_high)
        # return output
        # return aligned_low, aligned_mid, aligned_high, output
        return aligned_low, aligned_mid, aligned_high

class Align(nn.Module):
    def __init__(self, config):
        super(Align, self).__init__()
        print('Align')
        self.num_res_blocks = [3,5,5,5,5]
        self.T_dim = 16
        self.FE = FE(n_feats=self.T_dim) # ch * 1 2 4 8
        # self.FE_copy = FE(requires_grad=False) ### used in transferal perceptual loss
        self.Transformer = Transformer(config)
        self.Recon = Recon(num_res_blocks=self.num_res_blocks, n_feats=16, res_scale=1., T_dim=self.T_dim, config=config)

    def forward(self, mov, ref):
        # if (type(hdr)!=type(None)):
        #     self.FE_copy.load_state_dict(self.FE.state_dict())
        #     hdr_lv1, hdr_lv2, hdr_lv3 = self.FE_copy(hdr)
        #     return hdr_lv1, hdr_lv2, hdr_lv3
        mov_lv0, mov_lv1, mov_lv2, mov_lv3 = self.FE(mov.detach())
        _,       _,       _,       ref_lv3 = self.FE(ref.detach())
        S, T_lv3, T_lv2, T_lv1, T_lv0 = self.Transformer(query_lv3=ref_lv3, key_lv3=mov_lv3, value_lv0=mov_lv0, value_lv1=mov_lv1, value_lv2=mov_lv2, value_lv3=mov_lv3)
        aligned = self.Recon(ref, S, T_lv3, T_lv2, T_lv1, T_lv0)
        return aligned

class FE(torch.nn.Module):
    def __init__(self, n_feats):
        super(FE, self).__init__()
        
        self.conv11 = conv3x3(6, n_feats)
        self.conv12 = conv3x3(n_feats, n_feats)

        self.maxpool1 = maxpool()
        self.conv21 = conv3x3(n_feats, n_feats*2)
        self.conv22= conv3x3(n_feats*2, n_feats*2)

        self.maxpool2 = maxpool()
        self.conv31 = conv3x3(n_feats*2, n_feats*4)
        self.conv32= conv3x3(n_feats*4, n_feats*4)

        self.maxpool3 = maxpool()
        self.conv41 = conv3x3(n_feats*4, n_feats*8)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x_lv1 = x
        x = self.maxpool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x_lv2 = x
        x = self.maxpool2(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x_lv3 = x
        x = self.maxpool3(x)
        x = F.relu(self.conv41(x))
        x_lv4 = x
        return x_lv1, x_lv2, x_lv3, x_lv4
# class FE(torch.nn.Module):
#     def __init__(self, requires_grad=True, rgb_range=1):
#         super(FE, self).__init__()
        
#         ### use vgg19 weights to initialize
#         vgg_pretrained_features = models.vgg19(pretrained=True).features

#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()

#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.slice1.parameters():
#                 param.requires_grad = requires_grad
#             for param in self.slice2.parameters():
#                 param.requires_grad = requires_grad
#             for param in self.slice3.parameters():
#                 param.requires_grad = requires_grad
#         vgg_mean = (0.485, 0.456, 0.406)
#         vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
#         self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

#     def forward(self, x):
#         x = self.sub_mean(x)
#         x = self.slice1(x)
#         x_lv1 = x
#         x = self.slice2(x)
#         x_lv2 = x
#         x = self.slice3(x)
#         x_lv3 = x
#         return x_lv1, x_lv2, x_lv3

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

    def take(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W*s*s]
        # dim: scalar > 0
        # index: [N, H*W]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))] # N, 1, -1
        expanse = list(input.size()) # N, C*k*k, Hr*Wr*16
        expanse[0] = -1
        expanse[dim] = -1           # -1, C*k*k, -1
        index = index.view(views).expand(expanse) # [N, H*W] -> [N, 1, H*W] -> [N, C*k*k, H*W] 
        return torch.gather(input, dim, index) #[N, C*k*k, H*W] 

    def forward(self, query_lv3, key_lv3, value_lv0, value_lv1, value_lv2, value_lv3): # / 1 2 4 8
        ### search
        query_lv3_unfold  = F.unfold(query_lv3, kernel_size=(3, 3), padding=1) # [N, C, H, W] -> [N, C*k*k, H*W]
        key_lv3_unfold = F.unfold(key_lv3, kernel_size=(3, 3), padding=1) # [N, C, Hr, Wr] -> [N, C*k*k, Hr*Wr]
        key_lv3_unfold = key_lv3_unfold.permute(0, 2, 1) # [N, Hr*Wr, C*k*k]

        key_lv3_unfold = F.normalize(key_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        query_lv3_unfold  = F.normalize(query_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        ### relevance embedding
        corr_lv3 = torch.bmm(key_lv3_unfold, query_lv3_unfold) #[N, Hr*Wr, H*W]
        corr_lv3_value, corr_lv3_argmax = torch.max(corr_lv3, dim=1) #[N, H*W], [N, H*W] which Ref pixel is most correlatied with LR

        ### transfer
        value_lv3_unfold = F.unfold(value_lv3, kernel_size=(3, 3), padding=1) # [N, C, Hr/4, Wr/4] -> [N, C*k*k, Hr*Wr/16]
        value_lv2_unfold = F.unfold(value_lv2, kernel_size=(6, 6), padding=2, stride=2) # [N, C, Hr/2, Wr/2] -> [N, C*2k*2k, Hr*Wr/4]
        value_lv1_unfold = F.unfold(value_lv1, kernel_size=(12, 12), padding=4, stride=4) # [N, C, Hr, Wr] -> [N, C*4k*4k, Hr*Wr]
        value_lv0_unfold = F.unfold(value_lv0, kernel_size=(24, 24), padding=8, stride=8) # [N, C, Hr, Wr] -> [N, C*8k*8k, Hr*Wr]

        ### hard attention
        T_lv3_unfold = self.take(value_lv3_unfold, 2, corr_lv3_argmax) #[N, C*k*k, H*W]
        T_lv2_unfold = self.take(value_lv2_unfold, 2, corr_lv3_argmax) #[N, C*2k*2k, H*W]
        T_lv1_unfold = self.take(value_lv1_unfold, 2, corr_lv3_argmax) #[N, C*4k*4k, H*W] 
        T_lv0_unfold = self.take(value_lv0_unfold, 2, corr_lv3_argmax) #[N, C*8k*8k, H*W] Take most correlated Ref content

        T_lv3 = F.fold(T_lv3_unfold, output_size=query_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.) #[N, C*k*k, H*W] -> [N, C, H, W]
        T_lv2 = F.fold(T_lv2_unfold, output_size=(query_lv3.size(2)*2, query_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)  # [N, C, H, W]
        T_lv1 = F.fold(T_lv1_unfold, output_size=(query_lv3.size(2)*4, query_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.) # [N, C, H, W]
        T_lv0 = F.fold(T_lv0_unfold, output_size=(query_lv3.size(2)*8, query_lv3.size(3)*8), kernel_size=(24,24), padding=8, stride=8) / (3.*3.) # [N, C, H, W]

        S = corr_lv3_value.view(corr_lv3_value.size(0), 1, query_lv3.size(2), query_lv3.size(3)) # [N, 1, H, W]

        return S, T_lv3, T_lv2, T_lv1, T_lv0


class Recon(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale, T_dim, config):
        super(Recon, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SE = SE(self.num_res_blocks[0], n_feats, res_scale)

        ### stage11
        self.conv33_head = conv3x3(T_dim*8+n_feats, n_feats)
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.conv22_head = conv3x3(T_dim*4+n_feats, n_feats)
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv22_tail = conv3x3(n_feats, n_feats)

        self.conv32 = conv3x3(n_feats*2, n_feats)

        self.conv11_head = conv3x3(T_dim*2+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats*2, n_feats)

        self.conv00_head = conv3x3(T_dim*1+n_feats, n_feats)
        self.RB00 = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RB00.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv00_tail = conv3x3(n_feats, n_feats)

        self.conv10 = conv3x3(n_feats*2, n_feats)

        self.conv_final1 = conv3x3(n_feats, n_feats)
        self.conv_final2 = conv1x1(n_feats, 3)

        # self.merge_tail = MergeTail(n_feats)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None, T_lv0=None):
        ### shallow feature extraction
        x0, x1, x2, x3 = self.SE(x) # F: feature from LR input

        ### stages3
        x33 = x3
        ### soft-attention # F = F + CONV(CONCAT(F,T)) . S
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv3), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * S
        x33 = x33 + x33_res

        x33_res = x33           # transformer output

        for i in range(self.num_res_blocks[1]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        ### stage21, 22
        # x21 = x11
        # x21_res = x21
        # x22 = self.conv12(x11)
        # x22 = F.relu(self.ps12(x22))

        ### stage2
        x22 = x2
        ### soft-attention
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        x22_res = x22

        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        x33_up = F.interpolate(x33, scale_factor=2, mode='bicubic')
        x32 = torch.cat((x33_up,x22), dim=1)
        x32 = self.conv32(x32)

        ### stage3
        x11 = x1
        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv1), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x11 = x11 + x11_res
        
        x11_res = x11

        for i in range(self.num_res_blocks[3]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        x32_up = F.interpolate(x32, scale_factor=2, mode='bicubic')
        x21 = torch.cat((x32_up,x11), dim=1)
        x21 = self.conv21(x21)

        ### stage3
        x00 = x0
        ### soft-attention
        x00_res = x00
        x00_res = torch.cat((x00_res, T_lv0), dim=1)
        x00_res = self.conv00_head(x00_res) #F.relu(self.conv00_head(x00_res))
        x00_res = x00_res * F.interpolate(S, scale_factor=8, mode='bicubic')
        x00 = x00 + x00_res
        
        x00_res = x00

        for i in range(self.num_res_blocks[4]):
            x00_res = self.RB00[i](x00_res)
        x00_res = self.conv00_tail(x00_res)
        x00 = x00 + x00_res

        x21_up = F.interpolate(x21, scale_factor=2, mode='bicubic')
        x10 = torch.cat((x21_up,x00), dim=1)
        x10 = self.conv10(x10)

        x = F.relu(self.conv_final1(x10))
        x = self.conv_final2(x)
        x = torch.clamp(x, 0, 1)

        # x = self.merge_tail(x11, x22, x33)

        return x

class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3): # x1 x2 x4
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, 0, 1)
        
        return x

class SE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv1_head = conv3x3(6, n_feats)
        self.RB1s = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB1s.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
        self.conv1_tail = conv3x3(n_feats, n_feats)
        
        self.conv2_head = conv3x3(n_feats, n_feats, 2)
        self.conv2_tail = conv3x3(n_feats, n_feats)

        self.conv3_head = conv3x3(n_feats, n_feats, 2)
        self.conv3_tail = conv3x3(n_feats, n_feats)

        self.conv4_head = conv3x3(n_feats, n_feats, 2)
        self.conv4_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv1_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RB1s[i](x)
        x = self.conv1_tail(x)
        x = x + x1
        x1_out = x

        x = F.relu(self.conv2_head(x))
        x = self.conv2_tail(x)
        x2_out = x

        x = F.relu(self.conv3_head(x))
        x = self.conv3_tail(x)
        x3_out = x

        x = F.relu(self.conv4_head(x))
        x = self.conv4_tail(x)
        x4_out = x

        return x1_out, x2_out, x3_out, x4_out

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        

# class Align(nn.Module):
#     def __init__(self, config):
#         super(Align, self).__init__()
#         print('Align')

#         # input_channel = 4 if 'with_mask' in config.refine_input else 3
#         input_cnum = 6
#         cnum = 16

#         self.src_conv1_1 = nn.Conv2d(input_cnum, cnum, 3, 1, 1)
#         self.src_conv1_2 = nn.Conv2d(cnum, cnum, 3, 2, 1)
#         self.src_rdb1 = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         self.src_rdb2 = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)

#         self.ref_conv1_1 = nn.Conv2d(input_cnum, cnum, 3, 1, 1)
#         self.ref_conv1_2 = nn.Conv2d(cnum, cnum, 3, 2, 1)
#         self.ref_rdb1 = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         self.ref_conv2_1 = nn.Conv2d(cnum*2, cnum, 1, 1, 0)
#         self.ref_conv2_2 = nn.Conv2d(cnum, cnum, 3, 1, 1)       
#         self.ref_rdb2 = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         self.ref_conv3_1 = nn.Conv2d(cnum*2, cnum, 1, 1, 0)
#         self.ref_conv3_2 = nn.Conv2d(cnum, cnum, 3, 1, 1)       
#         self.ref_rdb3 = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         # self.src_conv4_1 = nn.Conv2d(cnum*2, cnum, 1, 1, 1)
#         # self.src_conv4_2 = nn.Conv2d(cnum, cnum, 3, 1, 1)    
#         self.fus_conv1_1 = nn.Conv2d(cnum*3, cnum, 3, 1, 1)
#         self.fus_conv1_2 = nn.Conv2d(cnum, cnum, 3, 1, 1)
#         self.fus_conv2 = nn.Conv2d(cnum, cnum, 3, 1, 1)

#         # self.o_conv1 = nn.Conv2d(cnum, cnum, 3, 1, 1)
#         # self.dcn_pack = DCNv2Pack(cnum, cnum, 3, padding=1)
#         self.recon_conv1 = nn.ConvTranspose2d(cnum, cnum, 3, 2, 1, 1)
#         self.recon_conv2 = nn.Conv2d(cnum, 3, 3, 1, 1)

#         # self.rdb = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         self.lrelu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, src, ref):
#         # x = torch.cat((src,src),1)
#         s1 = self.lrelu(self.src_conv1_1(src))
#         s2 = self.lrelu(self.src_conv1_2(s1))
#         S1 = self.src_rdb1(s2)
#         S2 = self.src_rdb2(S1)

#         r1 = self.lrelu(self.ref_conv1_1(ref))
#         r2 = self.lrelu(self.ref_conv1_2(r1))
#         R1 = self.ref_rdb1(r2)
#         cat1 = torch.cat((S1,R1),1)
#         r3 = self.lrelu(self.ref_conv2_1(cat1))
#         r4 = self.lrelu(self.ref_conv2_2(r3))
#         R2 = self.ref_rdb2(r4)    
#         cat2 = torch.cat((S2,R2),1)
#         r5 = self.lrelu(self.ref_conv3_1(cat2))
#         r6 = self.lrelu(self.ref_conv3_2(r5))
#         R3 = self.ref_rdb3(r6) 
#         cat3 = torch.cat((R1,R2,R3),1)
#         f1 = self.lrelu(self.fus_conv1_1(cat3))
#         f2 = self.lrelu(self.fus_conv1_2(f1))
#         f2 += S1
#         feat = self.lrelu(self.fus_conv2(f2))
#         recon1 = self.lrelu(self.recon_conv1(feat))
#         aligned = self.sigmoid(self.recon_conv2(recon1))
        
#         return aligned

# class Merge(nn.Module):
#     def __init__(self, config):
#         super(Merge,self).__init__()
#         print('Merge')

#         # input_channel = 4 if 'with_mask' in config.refine_input else 3
#         input_cnum = 3
#         cnum = 64

#         self.conv_low = nn.Conv2d(input_cnum, cnum, 3, 1, 1)
#         self.conv_mid = nn.Conv2d(input_cnum, cnum, 3, 1, 1)
#         self.conv_high = nn.Conv2d(input_cnum, cnum, 3, 1, 1)

#         self.conv_cat = nn.Conv2d(cnum*3, cnum, 3, 1, 1)
#         self.rbs = nn.Sequential(
#             RB(in_channels=cnum, out_channels=cnum),
#             RB(in_channels=cnum, out_channels=cnum),
#             RB(in_channels=cnum, out_channels=cnum),
#             RB(in_channels=cnum, out_channels=cnum),
#             RB(in_channels=cnum, out_channels=cnum)
#         )
#         self.conv1 = nn.Conv2d(cnum*4, cnum, 1, 1, 0)
#         self.conv2 = nn.Conv2d(cnum, 3, 3, 1, 1)

#         # self.rdb = RDB(nChannels=cnum, nDenselayer=3, growthRate=32)
#         self.lrelu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, low, mid, high):
#         feat_low = self.lrelu(self.conv_low(low))
#         feat_mid = self.lrelu(self.conv_mid(mid))
#         feat_high = self.lrelu(self.conv_high(high))
#         cat = torch.cat((feat_low,feat_mid,feat_high),1)
#         feat = self.lrelu(self.conv_cat(cat))
#         feat = self.rbs(feat)
#         feat = torch.cat((feat,cat),1)
#         feat = self.lrelu(self.conv1(feat))
#         output = self.sigmoid(self.conv2(feat))

#         return output



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator,self).__init__()
        print('Load Global Discrminator')
        # input_channel = 3
        # cnum = 64
        # gen=utils.import_module_from_file(os.path.join(ROOT_DIR, 'utils/layer.py'))
        # disc_conv = getattr(gen, config.disc_conv)
        # conv = getattr(gen, 'conv')

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is (3) x 96 x 96
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (64) x 48 x 48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (128) x 24 x 24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (256) x 12 x 12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (512) x 6 x 6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.lrelu = nn.LeakyReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x 

# if __name__ == "__main__":
#     config = parse_args()
#     a = Generator(config).cuda()

#     input = torch.randn(4,3,256,256).cuda()
#     mask = torch.randn(4,1,256,256).cuda()
#     c,r,o = a(input,mask)

#     print(o.shape)
