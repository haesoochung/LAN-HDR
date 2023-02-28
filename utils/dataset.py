import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import utils
import glob
import kornia.augmentation as K
import kornia.geometry.transform as Kt
import random


class TrainingSet(Dataset):
    def __init__(self, args):
        # assert args.mask_type in ALLMASKTYPES
        self.args = args
        self.imglist = utils.get_training_patches(args.train_root) # (10, 74)
        # self.imglist = utils.get_files(args.train_root if train else args.valid_root)
        
    def __len__(self):
        return len(self.imglist[0])

    def __getitem__(self, index):

        input_ldr_lows, input_ldr_mids, input_ldr_highs, input_hdr_lows, input_hdr_mids, input_hdr_highs, \
        gt_ldr_lows, gt_ldr_mids, gt_ldr_highs, gt_hdr_lows, gt_hdr_mids, gt_hdr_highs, gt_hdrs, \
        input_ldr_mid_to_lows, input_ldr_mid_to_highs, exps = self.imglist

        input_LDR_low = self.imread(input_ldr_lows[index])
        input_LDR_mid = self.imread(input_ldr_mids[index])
        input_LDR_high = self.imread(input_ldr_highs[index])
        input_HDR_low = self.hdrread(input_hdr_lows[index])
        input_HDR_mid = self.hdrread(input_hdr_mids[index])
        input_HDR_high = self.hdrread(input_hdr_highs[index])        
        gt_LDR_low = self.imread(gt_ldr_lows[index])     
        gt_LDR_mid = self.imread(gt_ldr_mids[index])
        gt_LDR_high = self.imread(gt_ldr_highs[index])
        gt_HDR = self.hdrread(gt_hdrs[index])
        input_LDR_mid_to_low = self.imread(input_ldr_mid_to_lows[index])
        input_LDR_mid_to_high = self.imread(input_ldr_mid_to_highs[index])
        exp = self.txtread(exps[index])
        # input_low = torch.cat([input_LDR_low,input_HDR_low],0)
        # input_mid = torch.cat([input_LDR_mid,input_HDR_mid],0)
        # input_high = torch.cat([input_LDR_high,input_HDR_high],0)
        return (input_LDR_low, input_LDR_mid, input_LDR_high, 
                input_HDR_low, input_HDR_mid, input_HDR_high, 
                gt_LDR_low, gt_LDR_mid, gt_LDR_high, 
                gt_HDR, input_LDR_mid_to_low, input_LDR_mid_to_high, exp)
        # return (input_low, input_mid, input_high, gt_LDR_low, gt_LDR_mid, gt_LDR_high, gt_HDR)

    def imread(self, filename):        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32)/255.).permute(2, 0, 1).contiguous() # HWC to CHW
        return img
        
    def tifread(self, filename):        
        img = cv2.imread(filename, -1)[:, :, ::-1]
        img = torch.from_numpy(img.astype(np.float32)/65535.).permute(2, 0, 1).contiguous()
        return img

    def hdrread(self, filename):        
        img = cv2.imread(filename,-1)[:, :, ::-1]
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()
        return img

    def txtread(self, filename):
        file = open(filename, 'r')
        a = float(file.readline())
        b = float(file.readline())
        c = float(file.readline())
        file.close()
        return [a, b, c]
class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.imglist = utils.get_test_images(args.test_root) # (10, 74)
        
    def __len__(self):
        return len(self.imglist[0])

    def __getitem__(self, index):

        input_ldr_lows, input_ldr_mids, input_ldr_highs, input_hdr_lows, input_hdr_mids, input_hdr_highs, \
        gt_ldr_lows, gt_ldr_mids, gt_ldr_highs, gt_hdrs, input_ldr_mid_to_lows, input_ldr_mid_to_highs = self.imglist

        input_LDR_low = self.imread(input_ldr_lows[index])
        input_LDR_mid = self.imread(input_ldr_mids[index])
        input_LDR_high = self.imread(input_ldr_highs[index])
        input_HDR_low = self.hdrread(input_hdr_lows[index])
        input_HDR_mid = self.hdrread(input_hdr_mids[index])
        input_HDR_high = self.hdrread(input_hdr_highs[index])        
        gt_LDR_low = self.imread(gt_ldr_lows[index])     
        gt_LDR_mid = self.imread(gt_ldr_mids[index])
        gt_LDR_high = self.imread(gt_ldr_highs[index])
        gt_HDR = self.hdrread(gt_hdrs[index])
        input_LDR_mid_to_low = self.imread(input_ldr_mid_to_lows[index])
        input_LDR_mid_to_high = self.imread(input_ldr_mid_to_highs[index])


        input_low = torch.cat([input_LDR_low,input_HDR_low],0)
        input_mid = torch.cat([input_LDR_mid,input_HDR_mid],0)
        input_high = torch.cat([input_LDR_high,input_HDR_high],0)
        input_mid_to_low = torch.cat([input_LDR_mid_to_low,input_HDR_mid])
        input_mid_to_high = torch.cat([input_LDR_mid_to_high,input_HDR_mid])
        return (input_low, input_mid, input_high, gt_HDR, input_mid_to_low, input_mid_to_high)

    def imread(self, filename):        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.args.crop:
            img = center_crop(img, self.args.w, self.args.h)
        elif self.args.resize:
            img = cv2.resize(img, (self.args.w, self.args.h))
        img = torch.from_numpy(img.astype(np.float32)/255.).permute(2, 0, 1).contiguous() # HWC to CHW
        return img
        
    def tifread(self, filename):        
        img = cv2.imread(filename, -1)[:, :, ::-1]
        if self.args.crop:
            img = center_crop(img, self.args.w, self.args.h)
        elif self.args.resize:
            img = cv2.resize(img, (self.args.w, self.args.h))
        img = torch.from_numpy(img.astype(np.float32)/65535.).permute(2, 0, 1).contiguous()
        return img

    def hdrread(self, filename):        
        img = cv2.imread(filename,-1)[:, :, ::-1]
        if self.args.crop:
            img = center_crop(img, self.args.w, self.args.h)
        elif self.args.resize:
            img = cv2.resize(img, (self.args.w, self.args.h))
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()
        return img

def center_crop(x, crop_w, crop_h):
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)], (crop_w, crop_h))        
    # def spatial_discounting_mask(self,mask):
    #     mask = (mask[0] * 255).astype(np.uint8)
    #     # 이진화된 결과를 dist_transform 함수의 입력으로 사용합니다. 
    #     dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    #     # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다. 
    #     return dist_transform[None,...]

    # def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
    #     """Generate a random free form mask with configuration.
    #     Args:
    #         config: Config should have configuration including IMG_SHAPES,
    #             VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    #     Returns:
    #         tuple: (top, left, height, width)
    #     """
    #     height = shape
    #     width = shape
    #     mask = np.zeros((height, width), np.float32)
    #     times = np.random.randint(times)
    #     for i in range(times):
    #         start_x = np.random.randint(width)
    #         start_y = np.random.randint(height)
    #         for j in range(1 + np.random.randint(5)):
    #             angle = 0.01 + np.random.randint(max_angle)
    #             if i % 2 == 0:
    #                 angle = 2 * 3.1415926 - angle
    #             length = 10 + np.random.randint(max_len)
    #             brush_w = 5 + np.random.randint(max_width)
    #             end_x = (start_x + length * np.sin(angle)).astype(np.int32)
    #             end_y = (start_y + length * np.cos(angle)).astype(np.int32)
    #             cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
    #             start_x, start_y = end_x, end_y
    #     return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    # def center_bbox(self,shape):
    #     mask = np.zeros((shape, shape), np.float32)
    #     mask[shape//3: (shape*2)//3, shape//3: (shape*2)//3] = 1.
    #     return mask[None,...]

    # def random_bbox(self, shape, margin, bbox_shape):
    #     """Generate a random tlhw with configuration.
    #     Args:
    #         config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    #     Returns:
    #         tuple: (top, left, height, width)
    #     """
    #     img_height = shape
    #     img_width = shape
    #     height = bbox_shape
    #     width = bbox_shape
    #     ver_margin = margin
    #     hor_margin = margin
    #     maxt = img_height - ver_margin - height
    #     maxl = img_width - hor_margin - width
    #     t = np.random.randint(low = ver_margin, high = maxt)
    #     l = np.random.randint(low = hor_margin, high = maxl)
    #     h = height
    #     w = width
    #     return (t, l, h, w)

    # def bbox2mask(self, shape, margin, bbox_shape, times):
    #     """Generate mask tensor from bbox.
    #     Args:
    #         bbox: configuration tuple, (top, left, height, width)
    #         config: Config should have configuration including IMG_SHAPES,
    #             MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    #     Returns:
    #         tf.Tensor: output with shape [1, H, W, 1]
    #     """
    #     bboxs = []
    #     for i in range(times):
    #         bbox = self.random_bbox(shape, margin, bbox_shape)
    #         bboxs.append(bbox)
    #     height = shape
    #     width = shape
    #     mask = np.zeros((height, width), np.float32)
    #     for bbox in bboxs:
    #         h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
    #         w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
    #         mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
    #     return mask.reshape((1, ) + mask.shape).astype(np.float32)

# class ValidationSet_with_Known_Mask(Dataset):
#     def __init__(self, args):
#         self.args = args
#         self.namelist = utils.get_names(args.baseroot)

#     def __len__(self):
#         return len(self.namelist)

#     def __getitem__(self, index):
#         # image
#         imgname = self.namelist[index]
#         imgpath = os.path.join(self.args.baseroot, imgname)
#         img = cv2.imread(imgpath)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.args.img_size, self.args.img_size))
#         # mask
#         maskpath = os.path.join(self.args.maskroot, imgname)
#         img = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
#         # the outputs are entire image and mask, respectively
#         img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
#         mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
#         return img, mask, imgname