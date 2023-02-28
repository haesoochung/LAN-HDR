import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.tools import *
import math

def positionalencoding2d(d_model, x):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    height, width = x.shape[-2:] 
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width).cuda()
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        _, _, h, w = x.shape
        # x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = x + F.interpolate(up_f, size = (h,w), mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x

class UpsampleBlockv2(nn.Module):
    def __init__(self, in_c, skip_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_c, skip_c, kernel_size=3, padding=1)
        self.rb = ResBlock(skip_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, x, skip_f):
        _, _, h, w = skip_f.shape
        x = F.interpolate(x, size = (h,w), mode='bilinear', align_corners=False)
        x = self.skip_conv(x)
        x = x + skip_f
        x = self.rb(x)
        return x
        
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     stride=stride, padding=2, bias=True)

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                        return_indices=False, ceil_mode=True)

class make_dense_fft(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=1):
    super(make_dense_fft, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

class FFT_RDB(nn.Module):
  def __init__(self, nChannels=64, nDenselayer=3, growthRate=32):
    super(FFT_RDB, self).__init__()
    nChannels_ = nChannels
    fft_nChannels_ = nChannels*2
    modules = []
    fft_modules = []
    self.norm='backward'
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate, dilation=2))
        fft_modules.append(make_dense_fft(fft_nChannels_, growthRate))
        nChannels_ += growthRate 
        fft_nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.fft_dense_layers = nn.Sequential(*fft_modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    self.fft_conv_1x1 = nn.Conv2d(fft_nChannels_, nChannels*2, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)

    _, _, H, W = x.shape
    dim = 1
    y = torch.fft.rfft2(x, norm=self.norm)
    y_imag = y.imag
    y_real = y.real
    y = torch.cat([y_real, y_imag], dim=dim)
    y = self.fft_dense_layers(y)
    y = self.fft_conv_1x1(y)
    y_real, y_imag = torch.chunk(y, 2, dim=dim)
    y = torch.complex(y_real, y_imag)
    y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
    return x + y + out

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        return x
        
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x2 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
                        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        if out_channels == None:
            out_channels = in_channels
        if in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        # x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out = out * self.res_scale + x
        return out
        
class RB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(RB, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
    def forward(self, x):
        x = x + self.rb(x)
        return x 

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3, dilation=1):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=((kernel_size-1)*(dilation-1)+kernel_size-1)//2, dilation=dilation, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels=64, nDenselayer=3, growthRate=32):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat*2, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.dim = n_feat
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        # y = torch.fft.rfft2(x, norm='ortho')
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm='ortho')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

class conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3,  
        stride = 1,
        pad_mode ='zeros',
        dilation = 1, 
        activation = nn.ELU()):
        super(conv, self).__init__()


        self.conv2d = nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size =kernel_size, 
                        stride = stride, 
                        dilation = dilation,
                        padding = int(dilation*(kernel_size-1)/2) if pad_mode != 'valid' else 0,
                        padding_mode=pad_mode if pad_mode != 'valid' else 'zeros')

        self.activation = activation if activation is not None else nn.Identity()


    def forward(self,x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x



class gated_conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3,  
        stride = 1,
        pad_mode ='zeros',
        dilation = 1, 
        activation = nn.ELU()):
        super(gated_conv, self).__init__()


        self.conv2d = nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels*2,
                        kernel_size =kernel_size, 
                        stride = stride, 
                        dilation = dilation,
                        padding = int(dilation*(kernel_size-1)/2) if pad_mode != 'valid' else 0,
                        padding_mode=pad_mode if pad_mode != 'valid' else 'zeros')

        self.out_channels = out_channels
        self.activation = activation if activation is not None else nn.Identity()        
        self.sigmoid = nn.Sigmoid()

        ### in last layer sth is different in activation with official code

    def forward(self,x):
        x = self.conv2d(x)
        x,g = torch.split(x,self.out_channels,dim=1)
        x = self.activation(x)
        g = self.sigmoid(g)
        x = x*g
        return x

