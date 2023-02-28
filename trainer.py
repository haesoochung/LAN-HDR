import os
import torch
from utils.loss import *
from config import ROOT_DIR
from utils import *
import glob
import kornia.augmentation as K
import kornia
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import *
from torch.utils.data import DataLoader
from datasets import custom_dataloader, benchmark_loader

import random
import time
import datetime
from utils import utils
import misc
class Trainer():
    def __init__(self, args, load, ckpt_dir, log_dir, val_dir):
        self.args = args
        self.load = load
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.val_dir = val_dir

        if self.args.nframes == 7 and self.args.nexps ==3:
            print('-3E model-')
            module = utils.import_module_from_file(os.path.join(ROOT_DIR, 'models/model_3E.py'))
        else:
            print('-2E model-')
            module = utils.import_module_from_file(os.path.join(ROOT_DIR, 'models/model.py'))
 
        Generator = getattr(module, "Generator")

        self.build_model(Generator)

    def build_model(self, Generator):
        misc.init_distributed_mode(self.args)          
        self.netG = Generator(self.args).cuda()
  
        print(self.args.gpu)
        model_without_ddp = self.netG 
        if self.args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[self.args.gpu])
            model_without_ddp = model.module
        self.optim_g = torch.optim.AdamW(self.netG.parameters(), lr=self.args.lr)
        utils.calc_param(self.netG)

    def prepare_inputs(self, data, split):
        self.split = split
        self.nframes = self.args.nframes
        self.nexps = self.args.nexps
        self.mid = self.nframes // 2

        expos = data['expos'].view(-1, self.nframes, 1, 1).split(1, 1)

        hdrs, log_hdrs = [], []
        ldrs, l2hdrs = [], []        
        for i in range(self.nframes):
            hdrs.append(data['hdr_%d' % i])
            ldrs.append(data['ldr_%d' % i])
        
        tone_aug = self.args.tone_low + self.args.tone_ref
        assert tone_aug <= 1

        if self.split in ['train'] and tone_aug == 1: # add noise
            utils.perturb_low_expo_imgs(self.args, ldrs, expos)
        
        for i in range(self.nframes):
            l2hdrs.append(utils.pt_ldr_to_hdr(ldrs[i], expos[i]))

        assert(len(expos) == len(ldrs))

        input_exp_ys = []
        inputs = []
        input_ys = []
        for i in range(0, len(ldrs)):
            ldr_exp = utils.pt_ldr_to_ldr(ldrs[self.mid], expos[self.mid], expos[i])
            hdr_exp = utils.pt_ldr_to_hdr(ldr_exp, expos[i])
            ldr_exp_y = torch.unsqueeze(kornia.color.rgb_to_ycbcr(ldr_exp)[..., 0, :, :], 1)
            ldr_y = torch.unsqueeze(kornia.color.rgb_to_ycbcr(ldrs[i])[..., 0, :, :], 1)
            input_exp_ys.append(torch.cat([ldr_exp_y, hdr_exp],1)) # 4
            input_ys.append(torch.cat([ldr_y, l2hdrs[i]],1))       # 4
            inputs.append(torch.cat([ldrs[i], l2hdrs[i], ldr_y],1))

        return hdrs, inputs, input_ys, input_exp_ys, expos

    def parse_data(self, sample):
        data = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                data[k] = v.cuda()
            else:
                data[k] = v
        self.data = data
        return data

    def train(self):
        speed = 0
        writer = SummaryWriter(log_dir=self.log_dir, max_queue=1000)
        (saved_iter, saved_epoch) = self.load_model(self.ckpt_dir, self.args.saved_iter) if self.load else (0, 0)
        sampler_train, train_loader, val_loader = custom_dataloader(self.args)

        iter_per_epoch = len(train_loader) 
        start_epoch = saved_iter // iter_per_epoch
        curr_iter = saved_iter
        self.netG.train()
        
        temp = utils.import_module_from_file(os.path.join(ROOT_DIR, 'utils/loss.py'))
        self.loss_dict = get_loss_dict(self.args)
        loss = {}
        try:
            for epoch in range(start_epoch, self.args.max_epoch):
                if self.args.distributed:
                    sampler_train.set_epoch(epoch)
                for iter, items in enumerate(train_loader): 
                    prev_time = time.time()
                    item = self.parse_data(items[0]) # .cuda()
                    item2 = self.parse_data(items[1]) # .cuda()
                    hdrs, inputs, input_ys, input_exp_ys, expos = self.prepare_inputs(item, 'train')
                    hdrs2, inputs2, input_ys2, input_exp_ys2, expos2 = self.prepare_inputs(item2, 'train')
                    output, gates = self.netG(inputs, input_ys, input_exp_ys)
                    output2, _ = self.netG(inputs2, input_ys2, input_exp_ys2)
                    
                    
                    self.optim_g.zero_grad()
                    out_tm = utils.tonemap(output)
                    gt_tm = utils.tonemap(hdrs[self.mid])
                    out2_tm = utils.tonemap(output2)
                    gt2_tm = utils.tonemap(hdrs2[self.mid])

                    loss["g/l1"] = self.loss_dict['recon_loss'](out_tm, gt_tm)
                    loss['g/vgg'] = self.loss_dict['vgg_loss'](out_tm, gt_tm) 
                    loss['g/frequency'] = self.loss_dict['frequency_loss'](out_tm, gt_tm) 
                    loss['g/temporal'] = self.loss_dict['charbonnier_loss'](out_tm-out2_tm, gt_tm-gt2_tm) 

                    loss_G = (
                                loss["g/l1"]
                                + loss["g/vgg"] * 0.1
                                + loss["g/frequency"] * 0.1
                                + loss["g/temporal"] * 0.1
                            )
                    loss_G.backward()
                    self.optim_g.step()

                    speed = speed * 0.6  + (time.time() - prev_time) * 0.4
                    curr_iter += 1

                self.validation(val_loader, epoch, writer)

                if (epoch+1) % self.args.save_epoch == 0:
                    self.save_model(self.ckpt_dir, epoch+1, curr_iter)

        except KeyboardInterrupt:
            print('***********KEY BOARD INTERRUPT *************')
            self.save_model(self.ckpt_dir, epoch, curr_iter)

        writer.close()

    def validation(self, val_dataloader, epoch, writer):
        running_vloss = 0.0
        with torch.no_grad():
            self.netG.eval()
            for v, items in enumerate(val_dataloader):
                item = self.parse_data(items[0]) # .cuda()
                hdrs, inputs, input_ys, input_exp_ys, expos = self.prepare_inputs(item, 'val')
                output, gates = self.netG(inputs, input_ys, input_exp_ys)
                out_tm = utils.tonemap(output)
                gt_tm = utils.tonemap(hdrs[self.mid]) 
                vloss = (
                            self.loss_dict['recon_loss'](out_tm, gt_tm)
                        )
                running_vloss += vloss
            avg_vloss = running_vloss / (v + 1)
            writer.add_scalar('Validation Loss', avg_vloss, epoch + 1)
            writer.flush()

    def test(self):
        output_path = 'results/model_%s/%s' % (self.args.affix, self.args.benchmark)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        test_loader, nscene = benchmark_loader(self.args)
        min_epoch = self.args.saved_epoch + self.args.interval
        max_epoch = 1000
        with torch.no_grad():
            checkpoints = glob.glob(os.path.join(self.ckpt_dir,"ckpt.pt"))
            if len(checkpoints) == 0:
                print('Checkpoint does not exist')
            total_psnrT = 0
            total_psnrL = 0
            total_ssimT = 0
            total_ssimL = 0
            PSNRs = []  
            _, _ = self.load_model(self.ckpt_dir, self.args.saved_iter)                
            for iter, item in enumerate(test_loader): 
                item = self.parse_data(item) # .cuda()
                hdrs, inputs, input_ys, input_exp_ys, expos = self.prepare_inputs(item, 'test')
                output, gates = self.netG(inputs, input_ys, input_exp_ys)
                out_tm = utils.tonemap(output)
                gt_tm = utils.tonemap(hdrs[self.mid])
                input_vis = torch.cat([gt_tm, out_tm], -1)    
                utils.imsave('%s/%03d_out.jpg' % (output_path,iter+1), input_vis) 
                psnrT, psnrL, ssimT, ssimL = utils.evaluate(output, hdrs[self.mid], out_tm, gt_tm)
                total_psnrT += psnrT
                total_psnrL += psnrL
                total_ssimT += ssimT
                total_ssimL += ssimL
                PSNRs.append(psnrT)
            avg_psnrT = total_psnrT/nscene  
            avg_psnrL = total_psnrL/nscene
            avg_ssimT = total_ssimT/nscene
            avg_ssimL = total_ssimL/nscene
            print('Average_PSNR_T : %f' % (avg_psnrT))
            print('Average_SSIM_T : %f' % (avg_ssimT))
            print('Average_PSNR_L : %f' % (avg_psnrL))
            print('Average_SSIM_L : %f\n' % (avg_ssimL))                    

    def save_model(self, path, epoch, iter):
        torch.save({
            'optim_g' :self.optim_g.state_dict(), 
            'g' : self.netG.state_dict(),
            'iter': iter,
            'epoch': epoch}, 
            os.path.join(path, "epoch_{:04d}_iter_{:06d}.pt".format(epoch, iter)))
        print('The %d epoch %d iter trained model is successfully saved!' % (epoch, iter))

    def load_model(self, path, saved_iter):
        if saved_iter != 0:
            checkpoint = torch.load((glob.glob(os.path.join(path,"*%06d.pt" % saved_iter)))[-1])
        else:
            checkpoint = torch.load(sorted(glob.glob(os.path.join(path,"*.pt")))[-1])

        checkpoint_model = checkpoint['g']
        for name in checkpoint_model.keys():
            if 'complex_weight' in name:
                c, h, w = checkpoint_model[name].shape[0:3]
                origin_weight = checkpoint_model[name]
                upsample_h = self.args.h 
                upsample_w = self.args.w // 2 + 1
                origin_weight = origin_weight.permute(0,3,1,2).reshape(1, 2*c, h, w)
                new_weight = torch.nn.functional.interpolate(
                    origin_weight, size=(upsample_h, upsample_w), mode='bicubic', align_corners=True).reshape(c, 2, upsample_h, upsample_w).permute(0, 2, 3, 1)
                checkpoint_model[name] = new_weight
        self.netG.load_state_dict(checkpoint["g"])
        self.optim_g.load_state_dict(checkpoint["optim_g"])
        iter = checkpoint["iter"]
        epoch = checkpoint["epoch"]
        print('The %d iteration trained model is successfully loaded.' % (iter))
        return (iter, epoch)
