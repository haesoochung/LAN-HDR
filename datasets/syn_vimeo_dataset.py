"""
Dataloader for processing vimeo videos into training data
"""
import cv2
import os
import numpy as np
from imageio import imread, imwrite

import torch
from torch.utils.data import Dataset

from datasets import hdr_transforms
from utils import utils
from utils import image_utils as iutils
np.random.seed(0)


class syn_vimeo_dataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(milestones=[5, 10, 20])
        return parser

    def __init__(self, args, split='train'):
        self.root = args.data_dir
        self.split = split
        self.args = args
        self.nframes = args.nframes
        self.nexps = args.nexps
        self._prepare_list(split)
        self._config_mode(split)

    def _prepare_list(self, split='train'):
        if split == 'train':
            list_name = 'sep_trainlist.txt'
        elif split == 'val':
            list_name = 'sep_testlist.txt'

        self.patch_list = utils.read_list(os.path.join(self.root, list_name))

        if split == 'val':
            self.patch_list = self.patch_list[:self.args.batch_size*4] # only use 100 val patches

    def _config_mode(self, split='train'):
        self.l_suffix = ''
        if split == 'train' and self.args.split_train == 'high_only':
            self.l_suffix = '_high.txt'
        elif split == 'train' and self.args.split_train == 'low_only':
            self.l_suffix = '_low.txt'
        elif self.args.split_train not in ['', 'high_only', 'low_only']:
            raise Exception('Unknown split type')

        if self.nexps == 2:
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat
        else:
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat

    def __getitem__(self, index):
        #index = 0
        group1, group2, min_percent, exposures1, exposures2 = self._get_input_path(index)
        hdrs = []

        """ sample parameters for the camera curves"""
        n, sigma = self.sample_camera_curve() # print(n, sigma)

        for img_path in group1:
            img = (imread(img_path).astype(np.float32) / 255.0).clip(0, 1)

            """ convert the LDR images to linear HDR image"""
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        if self.split == 'train':
            hdrs = self._augment_imgs(hdrs)
 
        item = {}
        hdrs, ldrs, anchor = self.re_expose_ldrs(hdrs, min_percent, exposures1)


        for i in range(self.nframes):
            item['ldr_%d' % i] = ldrs[i]
            item['hdr_%d' % i] = hdrs[i]

        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])

        item['expos'] = exposures1

        hdrs = []

        for img_path in group2:
            img = (imread(img_path).astype(np.float32) / 255.0).clip(0, 1)

            """ convert the LDR images to linear HDR image"""
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        if self.split == 'train':
            hdrs = self._augment_imgs(hdrs)
 
        item2 = {}
        hdrs, ldrs, anchor = self.re_expose_ldrs(hdrs, min_percent, exposures2)

        for i in range(self.nframes):
            item2['ldr_%d' % i] = ldrs[i]
            item2['hdr_%d' % i] = hdrs[i]

        for k in item2.keys(): 
            item2[k] = hdr_transforms.array_to_tensor(item2[k])

        item2['expos'] = exposures2
        return [item, item2]

    def re_expose_ldrs(self, hdrs, min_percent, exposures):
        mid = len(hdrs) // 2
        ldr_to_hdrs = []
        new_hdrs = []
        if self.nexps == 3:
            if exposures[mid] == 1:
                factor = np.random.uniform(0.1, 1)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # exposures[mid] == 4 or 8
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)
        else: # nexps == 2
            if exposures[mid] == 1: # low exposure reference
                factor = np.random.uniform(0.1, 1)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # high exposure reference
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)

        for idx, hdr in enumerate(hdrs):
            new_hdr = (hdr / (anchor + 1e-8) * new_anchor).clip(0, 1)
            new_hdrs.append(new_hdr)
        ldrs = []
        for i in range(len(new_hdrs)):
            ldr = iutils.hdr_to_ldr(new_hdrs[i], exposures[i])
            ldrs.append(ldr)
            ldr_to_hdr = iutils.ldr_to_hdr(ldr, exposures[i])
            ldr_to_hdrs.append(ldr_to_hdr)
        return new_hdrs, ldrs, None

    def _augment_imgs(self, imgs):
        h, w, c = imgs[0].shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w

        if self.args.rescale and not (crop_h == h):
            max_h = h * self.args.sc_k
            max_w = w * self.args.sc_k
            sc_h = np.random.randint(crop_h, max_h) if self.args.rand_sc else self.args.scale_h
            sc_w = np.random.randint(crop_w, max_w) if self.args.rand_sc else self.args.scale_w
            imgs = hdr_transforms.rescale(imgs, [sc_h, sc_w])
  
        imgs = hdr_transforms.random_flip_lrud(imgs)
        imgs = hdr_transforms.random_crop(imgs, [crop_h, crop_w])
        imgs = hdr_transforms.random_rotate90(imgs)
        color_permute = np.random.permutation(3)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][:,:,color_permute]
        return imgs

    def _get_input_path(self, index, high_expo=8):
        scene, patch = self.patch_list[index // self.repeat].split('/')
        img_dir = os.path.join(self.root, 'sequences', self.patch_list[index // self.repeat])
        if self.nexps == 2:
            img_idxs = sorted(np.random.permutation(7)[:(self.nframes+1)] + 1)
        else:
            img_idxs = np.concatenate( (np.array([1,2,3,4,5,6,7]) , np.random.randint(low=5, high=8, size=[1], dtype=int))) 
        if self.args.inv_aug and np.random.random() > 0.5: # inverse time order
            img_idxs = img_idxs[::-1]

        img_paths = [os.path.join(img_dir, 'im%d.png' % idx) for idx in img_idxs]

        group1 = img_paths[:-1]
        group2 = img_paths[1:]        
        min_percent = 99.8
        if self.nexps == 2:
            cur_high = True if index % 2 == 0 else False
            exposures1, exposures2 = self._get_2exposures(cur_high)
        elif self.nexps == 3:
            exposures1, exposures2 = self._get_3exposures(index)
        else:
            raise Exception("Unknown exposures")
        return group1, group2, min_percent, exposures1, exposures2

    def _get_2exposures(self, cur_high): # [1 4 1 4 1] or [4 1 4 1 4]

        exposures1 = np.ones(self.nframes, dtype=np.float32)
        exposures2 = np.ones_like(exposures1)
        high_expo = np.random.choice([4., 8.]) if self.args.rstop else 8

        if cur_high:
            for i in range(0, self.nframes, 2):
                exposures1[i] = high_expo
            for i in range(1, self.nframes, 2):
                exposures2[i] = high_expo                
        else:
            for i in range(1, self.nframes, 2):
                exposures1[i] = high_expo
            for i in range(0, self.nframes, 2):
                exposures2[i] = high_expo                
        return exposures1, exposures2

    def _get_3exposures(self, index):
        if index % self.nexps == 0:
            exp1 = [1, 4, 16, 1, 4, 16, 1]
            exp2 = [4, 16, 1, 4, 16, 1, 4]
        elif index % self.nexps == 1:
            exp1 = [4, 16, 1, 4, 16, 1, 4]
            exp2 = [16, 1, 4, 16, 1, 4, 16]
        else:
            exp1 = [16, 1, 4, 16, 1, 4, 16]
            exp2 = [1, 4, 16, 1, 4, 16, 1]
        exposures1 = np.array(exp1).astype(np.float32)
        exposures2 = np.array(exp2).astype(np.float32)
        return exposures1, exposures2

    def sample_camera_curve(self):
        n = np.clip(np.random.normal(0.65, 0.1), 0.4, 0.9)
        sigma = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.8)
        return n, sigma

    def apply_sigmoid_curve(self, x, n, sigma):
        y = (1 + sigma) * np.power(x, n) / (np.power(x, n) + sigma)
        return y

    def apply_inv_sigmoid_curve(self, y, n, sigma):
        x = np.power((sigma * y) / (1 + sigma - y), 1/n)
        return x

    def apply_inv_s_curve(self, y):
        x = 0.5 - np.sin(np.arcsin(1 - 2*y)/3.0)
        return x

    def discretize_to_uint16(self, img):
        max_int = 2**16-1
        img_uint16 = np.uint16(img * max_int).astype(np.float64) / max_int
        return img_uint16

    def __len__(self):
        return len(self.patch_list) * self.repeat
