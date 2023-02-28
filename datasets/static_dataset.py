"""
Dataloader for the captured real-world dataset
It supports static scene with GT, dynamic scene with GT, and dynamic without GT
"""
import os
import numpy as np
from imageio import imread
import torch
import torch.utils.data as data

from datasets import hdr_transforms
from datasets.tog13_online_align_dataset import tog13_online_align_dataset
from utils import utils
np.random.seed(0)

class static_dataset(tog13_online_align_dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, args, split='train'):
        self.root  = os.path.join(args.data_dir)
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.nexps = args.nexps if hasattr(args, 'nexps') else 2
        self._prepare_list()
        self.run_model = args.run_model if hasattr(args, 'run_model') else False
    
    def _prepare_list(self): 
        self.scene_list = utils.read_list(os.path.join(self.root, 'scene_all.txt'))
        skip_headtail = False 
        if self.nframes == 3 or (self.nframes == 5 and self.nexps == 3):
            skip_headtail = True

        self._collect_triple_list(self.scene_list, skip_headtail=skip_headtail)
        print('[%s] totaling  %d triples' % (self.__class__.__name__, len(self.triple_list)))

    def __getitem__(self, index):
        img_paths, exposures, hdr_paths, img_idxs = self._get_input_path(index, get_img_idx=True)

        item = {}
        ldr_start, ldr_end, hdr_start, hdr_end = self.get_ldr_hdr_start_end(index)
        
        imgs, hdrs = [], []
        for i in range(ldr_start, ldr_end):
            if img_paths[i][-4:] == '.tif':
                img = utils.read_16bit_tif(img_paths[i])
            else:
                img = imread(img_paths[i]) / 255.0
            item['ldr_%d' % i] = img

        for i in range(hdr_start, hdr_end):
            if self.nexps == 2:
                hdr_path = hdr_paths[i]
            elif self.nexps == 3:
                hdr_path = hdr_paths[i]

            if os.path.exists(hdr_path):
                hdr = utils.read_hdr(hdr_path)
                if hdr.max() > 1:
                    hdr = hdr / hdr.max()

            else:
                # No GT hdr, use linearized LDR as GT
                if 'ldr_%d'%i in item:
                    hdr = utils.ldr_to_hdr(item['ldr_%d'%i], exposures[i])
                else: # run_model mode
                    hdr = utils.ldr_to_hdr(item['ldr_%d'%ldr_start], exposures[ldr_start])

            item['hdr_%d' % i] = hdr

        origin_hw = (img.shape[0], img.shape[1])
        item = self.post_process(item, img_paths)

        item['hw'] = origin_hw
        item['expos'] = exposures
        item['reuse_cached_data'] = True if ldr_end - ldr_start == 1 else False
        return item

