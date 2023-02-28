import os
import cv2
import torch
import importlib
import datetime
import random
import numpy as np
import glob
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from imageio import imread, imsave
import re



### IO Related ###
def make_file(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def make_files(f_list):
    for f in f_list:
        make_file(f)

def empty_file(name):
    with open(name, 'w') as f:
        f.write(' ')

def read_list(list_path, ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def split_list(in_list, percent=0.99):
    num1 = int(len(in_list) * percent)
    #num2 = len(in_list) - num2
    rand_index = np.random.permutation(len(in_list))
    list1 = [in_list[l] for l in rand_index[:num1]]
    list2 = [in_list[l] for l in rand_index[num1:]]
    return list1, list2

def write_string(filename, string):
    with open(filename, 'w') as f:
        f.write('%s\n' % string)

def save_list(filename, out_list):
    f = open(filename, 'w')
    #f.write('#Created in %s\n' % str(datetime.datetime.now()))
    for l in out_list:
        f.write('%s\n' % l)
    f.close()

def create_dirs(root, dir_list, sub_dirs):
    for l in dir_list:
        makeFile(os.path.join(root, l))
        for sub_dir in sub_dirs:
            makeFile(os.path.join(root, l, sub_dir))

#### String Related #####
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dict_to_string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def float_list_to_string(l):
    strs = ''
    for f in l:
        strs += ',%.2f' % (f)
    return strs

def insert_suffix(name_str, suffix):
    str_name, str_ext = os.path.splitext(name_str)
    return '%s_%s%s' % (str_name, suffix, str_ext)

def insert_char(mystring, position, chartoinsert):
    mystring = mystring[:position] + chartoinsert + mystring[position:] 
    return mystring  

def get_datetime(minutes=False):
    t = datetime.datetime.now()
    dt = ('%02d-%02d' % (t.month, t.day))
    if minutes:
        dt += '-%02d-%02d' % (t.hour, t.minute)
    return dt

def check_in_list(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def remove_slash(string):
    if string[-1] == '/':
        string = string[:-1]
    return string

### Debug related ###
def check_div_by_2exp(h, w):
    num_h = np.log2(h)
    num_w = np.log2(w)
    if not (num_h).is_integer() or not (num_w).is_integer():
        raise Exception('Width or height cannot be devided exactly by 2exponet')
    return int(num_h), int(num_w)

def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_module_from_file(full_path_to_module):
    """
    Import a module given the full path/filename of the .py file
    Python 3.4
    """

    module = None

    try:
        # Get module name and path from full path
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, module_ext = os.path.splitext(module_file)

        # Get module "spec" from filename
        spec = importlib.util.spec_from_file_location(module_name,full_path_to_module)
        module = spec.loader.load_module()

    except Exception as ec:
        # Simple error printing
        # Insert "sophisticated" stuff here
        print(ec)

    finally:
        return module

# Read trainig patches ver.
def get_training_patches(root_folder):
    input_ldr_low_path = os.path.join(root_folder, 'input_LDR_low')
    input_ldr_lows = [os.path.join(input_ldr_low_path, filename) for filename in sorted(os.listdir(input_ldr_low_path))]
    input_ldr_mid_path = os.path.join(root_folder, 'input_LDR_mid')
    input_ldr_mids = [os.path.join(input_ldr_mid_path, filename) for filename in sorted(os.listdir(input_ldr_mid_path))]
    input_ldr_high_path = os.path.join(root_folder, 'input_LDR_high')
    input_ldr_highs = [os.path.join(input_ldr_high_path, filename) for filename in sorted(os.listdir(input_ldr_high_path))]
    input_hdr_low_path = os.path.join(root_folder, 'input_HDR_low')
    input_hdr_lows = [os.path.join(input_hdr_low_path, filename) for filename in sorted(os.listdir(input_hdr_low_path))]
    input_hdr_mid_path = os.path.join(root_folder, 'input_HDR_mid')
    input_hdr_mids = [os.path.join(input_hdr_mid_path, filename) for filename in sorted(os.listdir(input_hdr_mid_path))]
    input_hdr_high_path = os.path.join(root_folder, 'input_HDR_high')
    input_hdr_highs = [os.path.join(input_hdr_high_path, filename) for filename in sorted(os.listdir(input_hdr_high_path))]
    gt_ldr_low_path = os.path.join(root_folder, 'gt_LDR_low')
    gt_ldr_lows = [os.path.join(gt_ldr_low_path, filename) for filename in sorted(os.listdir(gt_ldr_low_path))]
    gt_ldr_mid_path = os.path.join(root_folder, 'gt_LDR_mid')
    gt_ldr_mids = [os.path.join(gt_ldr_mid_path, filename) for filename in sorted(os.listdir(gt_ldr_mid_path))]
    gt_ldr_high_path = os.path.join(root_folder, 'gt_LDR_high')
    gt_ldr_highs = [os.path.join(gt_ldr_high_path, filename) for filename in sorted(os.listdir(gt_ldr_high_path))]
    gt_hdr_low_path = os.path.join(root_folder, 'gt_HDR_low')
    gt_hdr_lows = [os.path.join(gt_hdr_low_path, filename) for filename in sorted(os.listdir(gt_hdr_low_path))]
    gt_hdr_mid_path = os.path.join(root_folder, 'gt_HDR_mid')
    gt_hdr_mids = [os.path.join(gt_hdr_mid_path, filename) for filename in sorted(os.listdir(gt_hdr_mid_path))]
    gt_hdr_high_path = os.path.join(root_folder, 'gt_HDR_high')
    gt_hdr_highs = [os.path.join(gt_hdr_high_path, filename) for filename in sorted(os.listdir(gt_hdr_high_path))]
    gt_hdr_path = os.path.join(root_folder, 'gt_HDR')
    gt_hdrs = [os.path.join(gt_hdr_path, filename) for filename in sorted(os.listdir(gt_hdr_path))]
    input_ldr_mid_to_low_path = os.path.join(root_folder, 'input_LDR_mid_to_low')
    input_ldr_mid_to_lows = [os.path.join(input_ldr_mid_to_low_path, filename) for filename in sorted(os.listdir(input_ldr_mid_to_low_path))]
    input_ldr_mid_to_high_path = os.path.join(root_folder, 'input_LDR_mid_to_high')
    input_ldr_mid_to_highs = [os.path.join(input_ldr_mid_to_high_path, filename) for filename in sorted(os.listdir(input_ldr_mid_to_high_path))]
    
    exp_path = os.path.join(root_folder, 'exposure')
    exps = [os.path.join(exp_path, filename) for filename in sorted(os.listdir(exp_path))]    
    # print('# gt_hdr : ', len(gt_hdrs))     
    return input_ldr_lows, input_ldr_mids, input_ldr_highs, \
           input_hdr_lows, input_hdr_mids, input_hdr_highs, \
           gt_ldr_lows, gt_ldr_mids, gt_ldr_highs, \
           gt_hdr_lows, gt_hdr_mids, gt_hdr_highs, \
           gt_hdrs, input_ldr_mid_to_lows, input_ldr_mid_to_highs, exps

# Read full images and randomly crop ver.
def get_test_images(root_folder):
    # read a folder, return the complete path
    scenes = sorted(os.listdir(root_folder))
    input_ldr_lows = []; input_ldr_mids = []; input_ldr_highs = [] 
    input_hdr_lows = []; input_hdr_mids = []; input_hdr_highs = []
    gt_ldr_lows = []; gt_ldr_mids = []; gt_ldr_highs = []
    gt_hdrs = []; input_ldr_mid_to_lows = []; input_ldr_mid_to_highs = []

    print('# scenes: ', len(scenes))
    for scene in scenes:
        scene_path = os.path.join(root_folder, scene)
        ldrs = sorted(glob.glob('%s/*.tif' % scene_path))
        hdrs = sorted(glob.glob('%s/input_*.hdr' % scene_path))
        gt_ldrs = sorted(glob.glob('%s/gt_*.png' % scene_path))
        gt_hdr = '%s/HDRImg.hdr' % scene_path
        input_ldr_mid_to_low = '%s/input_LDR_mid_to_low.png' % scene_path
        input_ldr_mid_to_high = '%s/input_LDR_mid_to_high.png' % scene_path
        input_ldr_lows.append(ldrs[0]); input_ldr_mids.append(ldrs[1]); input_ldr_highs.append(ldrs[2])
        input_hdr_lows.append(hdrs[0]); input_hdr_mids.append(hdrs[1]); input_hdr_highs.append(hdrs[2])
        gt_ldr_lows.append(gt_ldrs[0]); gt_ldr_mids.append(gt_ldrs[1]); gt_ldr_highs.append(gt_ldrs[2])
        gt_hdrs.append(gt_hdr)
        input_ldr_mid_to_lows.append(input_ldr_mid_to_low); input_ldr_mid_to_highs.append(input_ldr_mid_to_high)
        
    return input_ldr_lows, input_ldr_mids, input_ldr_highs, \
           input_hdr_lows, input_hdr_mids, input_hdr_highs, \
           gt_ldr_lows, gt_ldr_mids, gt_ldr_highs, gt_hdrs, \
           input_ldr_mid_to_lows, input_ldr_mid_to_highs

# def get_files(path):
#     # read a folder, return the complete path
#     files = sorted(os.listdir(path))
#     ret = []
#     for file in files:
#         ret.append(os.path.join(path, file))
#     # ret = []
#     # for root, dirs, files in os.walk(path):
#     #     for filespath in files:
#     #         ret.append(os.path.join(root, filespath))
#     return ret

# def save_model(net, epoch, opt):
#     """Save the model at "checkpoint_interval" and its multiple"""
#     model_name = 'deepfillv2_LSGAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
#     model_name = os.path.join(save_folder, model_name)
#     if opt.multi_gpu == True:
#         if epoch % opt.checkpoint_interval == 0:
#             torch.save(net.module.state_dict(), model_name)
#             print('The trained model is successfully saved at epoch %d' % (epoch))
#     else:
#         if epoch % opt.checkpoint_interval == 0:
#             torch.save(net.state_dict(), model_name)
#             print('The trained model is successfully saved at epoch %d' % (epoch))
def print_weights(model):
    for name, param in model.named_parameters():
        if 'conv3_2.bias' in name:
            print(name, param)

def to_vis(image):
    i = image*255.
    i = i.type(torch.uint8)
    return i

def tonemap(img):
    # x = 1+5000.*img
    return torch.log(1+5000.*img) / torch.log(torch.tensor(1+5000.).cuda())

def calc_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# total parameters: %d K' % (total_params/1000))
    print('# trainable parameters: %d K' % (trainable_params/1000))

def evaluate(hdr, gt, hdr_tm, gt_tm):
    # mseT = np.mean( (hdr_tm - gt_tm) ** 2 )
    # if mseT == 0:
    #     return 100
    # PIXEL_MAX = 1.0 # input -1~1
    # psnrT = 20 * math.log10(PIXEL_MAX / math.sqrt(mseT))
    # mseL = np.mean((gt-hdr)**2)
    # psnrL = -10.*math.log10(mseL)
    hdr = tensor2numpy(hdr); gt = tensor2numpy(gt); hdr_tm = tensor2numpy(hdr_tm); gt_tm = tensor2numpy(gt_tm)    
    psnrT = psnr(gt_tm, hdr_tm)
    psnrL = psnr(gt, hdr)
    ssimT = ssim(gt_tm, hdr_tm, multichannel=True, data_range=gt_tm.max() - gt_tm.min())
    ssimL = ssim(gt, hdr, multichannel=True, data_range=gt.max() - gt.min())

    return psnrT, psnrL, ssimT, ssimL

def evaluate_alignment(aligned_low, aligned_mid, aligned_high, gt_low, gt_mid, gt_high):
    aligned_low = tensor2numpy(aligned_low)
    aligned_mid = tensor2numpy(aligned_mid)
    aligned_high = tensor2numpy(aligned_high)
    gt_low = tensor2numpy(gt_low)
    gt_mid = tensor2numpy(gt_mid)
    gt_high = tensor2numpy(gt_high)
    psnrs = (psnr(gt_low, aligned_low) 
            + psnr(gt_mid, aligned_mid) 
            + psnr(gt_high, aligned_high))/3
    ssims = (ssim(gt_low, aligned_low, multichannel=True) 
            + ssim(gt_mid, aligned_mid, multichannel=True) 
            + ssim(gt_high, aligned_high, multichannel=True))/3
    return psnrs, ssims

def make_list(iteration, PSNRs, avg_psnrT, avg_ssimT, avg_psnrL, avg_ssimL):
  list = []
  list.append(iteration)
  list.extend(PSNRs)
  list.extend([avg_psnrT, avg_ssimT, avg_psnrL, avg_ssimL])
  list = tuple(list)
  return list

def write_csv(csv_path, final_list, nscene):
  if os.path.isfile(csv_path) == False:
    col_0 = []
    col_0.append('epoch')
    col_0.extend(range(1, nscene+1))
    col_0.extend(['psnrT','ssimT','psnrL','ssimL'])
    col_0 = tuple(col_0)
    final_list.insert(0, col_0)
  zipped = list(zip(*final_list))  
  with open(csv_path,'a',newline='') as f:
      wr = csv.writer(f)
      for row in zipped:
        wr.writerow([*row])

def tensor2numpy(tensor, out_type=np.float32, min_max=(0, 1)):
    # 4D: grid (1, C, H, W), 3D: (C, H, W), 2D: (H, W)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        img_np = tensor.squeeze().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def ldr2hdr(x, exp):
    return torch.div(torch.pow(x, torch.tensor(2.2,dtype=torch.float32).cuda()), exp.float().view(-1,1,1,1))


# Conversion between LDRs and HDRs
def pt_ldr_to_hdr(ldr, expo, gamma=2.2):
    #expo = epo.view(-1, 1, 1, 1)
    ldr = ldr.clamp(0, 1)
    ldr = torch.pow(ldr, gamma)
    hdr = ldr / expo
    return hdr

def pt_ldr_to_hdr_clamp(ldr, expo, gamma=2.2):
    #expo = epo.view(-1, 1, 1, 1)
    ldr = ldr.clamp(1e-8, 1)
    ldr = torch.pow(ldr, gamma)
    hdr = ldr / expo
    return hdr

def pt_hdr_to_ldr_clamp(hdr, expo, gamma=2.2,):
    ldr = torch.pow((hdr * expo).clamp(1e-8), 1.0 / gamma)
    ldr = ldr.clamp(0, 1)
    return ldr

def pt_hdr_to_ldr(hdr, expo, gamma=2.2):
    ldr = torch.pow(hdr * expo, 1.0 / gamma)
    ldr = ldr.clamp(0, 1)
    return ldr

def pt_ldr_to_ldr(ldr, expo_l2h, expo_h2l, gamma=2.2):
    ldr = ldr.clamp(0, 1)
    #if expo_l2h == expo_h2l:
    #    return ldr
    gain = torch.pow(expo_h2l / expo_l2h, 1.0 / gamma)
    ldr = (ldr * gain).clamp(0, 1)
    return ldr

    
##### HDR related #####
def ldr_to_hdr(img, expo, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, gamma) # linearize
    img /= expo
    return img


def hdr_to_ldr(img, expo, gamma=2.2):
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img


def ldr_to_ldr(img, expo_l2h, expo_h2l):
    if expo_l2h == expo_h2l:
        return img
    img = ldr_to_hdr(img, expo_l2h)
    img = hdr_to_ldr(img, expo_h2l)
    return img


## Data augmentation used in this work
def pt_tone_ref_tone_augment(ldr, d=0.1):
    n, c, h, w = ldr.shape
    # d: [-0.7, 0.7] -> gamma [0.49, 2.0]
    gammas = torch.exp(torch.rand(n, 3, 1, 1) * 2 * d - d)
    gammas = gammas.to(ldr.device)
    ldr_tone_aug = torch.pow(ldr, 1.0 / gammas)
    return ldr_tone_aug

def pt_tone_ref_add_gaussian_noise(img, stdv1=1e-3, stdv2=1e-2, max_thres=0.08, scale=True):
    stdv = torch.rand(img.shape, device=img.device) * (stdv2 - stdv1) + stdv1
    noise = torch.normal(0, stdv)
    out = torch.pow(img.clamp(0, 1), 2.2) # undo gamma
    out = (out + noise).clamp(0, 1)
    out = torch.pow(out, 1/2.2)
    return out


def perturb_low_expo_imgs(args, ldrs, expos):
    need_aug = (args.aug_prob == 1.0) or (np.random.uniform() < args.aug_prob)
    if not need_aug:
        return

    for i in range(args.nexps):
        cur_l_idx = torch.zeros(expos[0].shape[0], device=expos[0].device).byte()
        if i > 0:
            cur_l_idx = cur_l_idx | (expos[i] < expos[i-1]).view(-1)
        if i < args.nframes-1:
            cur_l_idx = cur_l_idx | (expos[i] < expos[i+1]).view(-1)

        if cur_l_idx.sum() > 0:
            tone_d = None
            params = {}
            for j in range(i, args.nframes, args.nexps): # e.g., [0,2], [0,3]
                if args.tone_low:
                    ldrs[j][cur_l_idx] = pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-4, stdv2=1e-3, scale=False)
                elif args.tone_ref:
                    ldrs[j][cur_l_idx] = pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-3, stdv2=1e-3, scale=False)
                else:
                    raise Exception('Unknown tone low mode')

# image read/save

###### Image Loading ######
def read_16bit_tif(img_name, crf=None):
    img = cv2.imread(img_name, -1) #/ 65535.0 # normalize to [0, 1]
    img = img[:, :, [2, 1, 0]] # BGR to RGB
    #print('before', img.max(), img.mean(), img.min())
    if crf is not None:
        img = reverse_crf(img, crf)
        img = img / crf.max()
    else:
        img = img / 65535.0
    #print('after', img.max(), img.mean(), img.min())
    return img

def reverse_crf(img, crf):
    img = img.astype(int)
    out = img.astype(float)
    for i in range(img.shape[2]):
        out[:,:,i] = crf[:,i][img[:,:,i]]
    return out

def read_hdr(filename, use_cv2=True):
    ext = os.path.splitext(filename)[1]
    if use_cv2:
        hdr = cv2.imread(filename, -1)[:,:,::-1].clip(0)
    # elif ext == '.exr':
    #     hdr = read_exr(filename) 
    elif ext == '.hdr':
        hdr = cv2.imread(filename, -1)
    elif ext == '.npy':
        hdr = np.load(filename) 
    else:
        raise_not_defined()
    return hdr

# def read_exr(filename):
#     hdr_file = OpenEXR.InputFile(filename)
#     img = to_array(hdr_file)
#     return img

def imsave(path, x):
    if x.ndim == 4:
        x = x.squeeze()
    x = 255.*x
    x = np.transpose(x.cpu().numpy(), (1, 2, 0)) # C H W -> H W C
    x = x[...,::-1] # RGB to BGR
    cv2.imwrite(path, x)


# def to_vis(image):
#     i = (image + 1)*127.5
#     i = i.type(torch.uint8)
#     return i

# if __name__ == "__main__":

#     import cv2 as cv
#     import numpy as np

#     e = np.zeros((128,128,1)).astype(np.uint8)
#     e[32:96,32:96] = 255


#     ret, thresh = cv.threshold(e, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


#     # 이진화된 결과를 dist_transform 함수의 입력으로 사용합니다. 

#     dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 3)
#     # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다. 
#     result = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)

#     cv.imshow("dist_transform", result)
#     cv.imshow("src", e)

#     cv.waitKey(0)