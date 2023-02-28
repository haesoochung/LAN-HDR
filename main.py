import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from config import *
from utils import utils
from trainer import Trainer

if __name__ == '__main__':
    utils.set_seed()
    args, Load, CKPT_DIR, LOG_DIR, VAL_DIR = initialize()
    trainer = Trainer(args, Load, CKPT_DIR, LOG_DIR, VAL_DIR)

    if args.train:
        print('train')
        trainer.train()
    else:
        trainer.test()

