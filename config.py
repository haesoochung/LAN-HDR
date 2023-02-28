from colorama import init, Fore
init(autoreset=True)
import argparse
import os
import json
from argparse import Namespace
import shutil
import datetime
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='****')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--new", action='store_true')
    parser.add_argument('--gpu', default='0', type=str) 
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument('--ch', default=64, type=int)
    parser.add_argument('--tdim', default=16, type=int)
    parser.add_argument('--affix', default='2E', type=str)
    
    ### video ver.
    parser.add_argument('--dataset', default='syn_vimeo_dataset') 
    parser.add_argument('--data_dir', default='./data/vimeo_septuplet')
    parser.add_argument('--split_train', default='', help='low_only|high_only, for debug') 
    parser.add_argument('--tone_low', default=False, action='store_true')
    parser.add_argument('--tone_ref', default=True, action='store_true')
    parser.add_argument('--aug_prob', default=0.9, type=float, help='prob of perturbing tone of input')

    parser.add_argument('--nframes', type=int, default=5)
    parser.add_argument('--nexps', type=int, default=2)
    parser.add_argument('--repeat', default=1, type=int, help='repeat dataset in an epoch')
    parser.add_argument('--rescale', default=True, action='store_false')
    parser.add_argument('--sc_k', default=1.0, type=float, help='scale factor')
    parser.add_argument('--rand_sc', default=True, action='store_false')
    parser.add_argument('--scale_h', default=320, type=int)
    parser.add_argument('--scale_w', default=320, type=int)
    parser.add_argument('--crop', default=True, action='store_false')
    parser.add_argument('--crop_h', default=128, type=int)
    parser.add_argument('--crop_w', default=128, type=int)
    parser.add_argument('--inv_aug', default=True, action='store_false', help='inverse time order')
    parser.add_argument('--rstop', default=True, action='store_false', help='random stop value for high exposure')

    parser.add_argument('--benchmark', default='synthetic_dataset')

    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--num_patch', default=240, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--saved_iter', default=0, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument('--step_disp', default=5000, type=int)
    parser.add_argument("--visual_batch", type=int, default=2)
    parser.add_argument('--max_iter', dest='max_iteration', default=5000000, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)

    parser.add_argument('--gan_loss', default='wgan', type=str)
    parser.add_argument('--w_py', default=0, type=float)
    parser.add_argument('--w_gan', default=0, type=float)
    parser.add_argument('--w_align', default=1, type=float)

    parser.add_argument('--saved_epoch', default=0, type=int)
    parser.add_argument('--interval', default=5, type=int)

    args = parser.parse_args()
    return args

def initialize():
    args = parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    EXP_PATH, CKPT_DIR, LOG_DIR, VAL_DIR = set_path(args.affix)
    load = False
    if args.train:
        if os.path.exists(CKPT_DIR) and len(os.listdir(CKPT_DIR)) > 0:
            if args.new:
                if os.path.exists(LOG_DIR):
                    shutil.rmtree(LOG_DIR)
                print(Fore.YELLOW + "Restart")
            else:
                print(Fore.YELLOW + "Load model_{}".format(args.affix))
                load_args(EXP_PATH, args)
                load = True
        else:
            print(Fore.YELLOW + "No checkpoint")
            save_args(EXP_PATH, args)
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            if not os.path.exists(CKPT_DIR):
                os.makedirs(CKPT_DIR)
        if not os.path.exists(VAL_DIR):
            os.makedirs(VAL_DIR)
        print("Arguments:")
        for key, value in args.__dict__.items():
            print('\t%15s:\t%s' % (key, value))
    else:
        print(Fore.YELLOW + "Model_{} evaluation".format(args.affix))
    return args, load, CKPT_DIR, LOG_DIR, VAL_DIR

def save_args(SAVE_ROOT, _args):
    model_arg_root = os.path.join(SAVE_ROOT,'args.json')
    if not os.path.exists(os.path.dirname(model_arg_root)):
        os.makedirs(os.path.dirname(model_arg_root))
    with open(model_arg_root, 'w') as f:
        args_str = '{}'.format(json.dumps(vars(_args), sort_keys=False, indent=4))
        print(args_str, file=f)

def load_args(LOAD_ROOT, _args):
    load_path = os.path.join(LOAD_ROOT,'args.json')
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            arg_str = f.read()
            model_args = json.loads(arg_str)
            model_args["batch_size"] = _args.batch_size
            args_ = Namespace(**model_args)
        return args_
    else:
        raise ValueError(
            "LOAD_ROOT not exist"
        )

def set_path(affix):
    exp_path = 'exps/model_%s' % ( affix)
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')
    val_dir = os.path.join(exp_path, 'validation')

    return exp_path, ckpt_dir, log_dir, val_dir
