import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vgg19 import Vgg19

class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss().cuda()
        elif (type == 'l2'):
            self.loss = nn.MSELoss().cuda()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, pred, gt):
        return self.loss(pred, gt)
        
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg19 = Vgg19(requires_grad=False).cuda()

    def forward(self, pred, gt):
        pred_relu5_1 = self.vgg19(pred)
        gt_relu5_1 = self.vgg19(gt)
        loss = F.mse_loss(pred_relu5_1, gt_relu5_1).cuda()
        return loss

class CosineLoss(nn.Module):
    def __init__(self, args):
        super(CosineLoss, self).__init__()        
        self.y = torch.tensor([1.]).cuda()        
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-08).cuda()

    def forward(self, pred, gt):
        cos = torch.mean(self.sim(pred, gt), dim=(1,2))
        loss = self.y - cos
        return torch.mean(loss)

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.loss = nn.L1Loss().cuda()
        
    def forward(self, pred, gt):
        return self.loss(torch.fft.fft2(pred), torch.fft.fft2(gt))


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)

def get_loss_dict(args):
    loss = {}
    loss['recon_loss'] = ReconstructionLoss(type='l1')
    loss['charbonnier_loss'] = CharbonnierLoss()
    loss['vgg_loss'] = PerceptualLoss()
    loss['cosine_loss'] = CosineLoss(args)
    loss['frequency_loss'] = FrequencyLoss()
    return loss

