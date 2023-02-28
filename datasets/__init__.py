import torch.utils.data
import importlib
from torch.utils.data import DistributedSampler

def find_dataset_from_string(dataset_name):
    datasetlib = importlib.import_module('datasets.%s' % (dataset_name))
    dataset_class = getattr(datasetlib, dataset_name)
    return dataset_class

def get_option_setter(dataset_name):
    dataset_class = find_dataset_from_string(dataset_name)
    return dataset_class.modify_commandline_options

def custom_dataloader(args):
    print("=> fetching img pairs in %s" % (args.dataset))
    datasets = __import__('datasets.' + args.dataset)
    dataset_file = getattr(datasets, args.dataset)
    train_set = getattr(dataset_file, args.dataset)(args, 'train')
    val_set   = getattr(dataset_file, args.dataset)(args, 'val')
    print('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    print('\t Train Batch: %d, Val Batch: %d' % (args.batch_size, args.batch_size//2))
    if args.distributed:
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    test_loader  = torch.utils.data.DataLoader(val_set, 
        sampler=sampler_val, batch_size=args.batch_size//2,
        num_workers=args.num_workers, shuffle=False)
    return sampler_train, train_loader, test_loader

def benchmark_loader(args):
    print("=> fetching img pairs in '%s'" % (args.benchmark))
    datasets = __import__('datasets.' + args.benchmark)
    dataset_file = getattr(datasets, args.benchmark)
    test_set = getattr(dataset_file, args.benchmark)(args, 'test')

    nscene = len(test_set)
    print('Found Benchmark Data: %d samples' % (nscene))

    use_gpu = len(args.gpu) > 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
        num_workers=args.num_workers, pin_memory=use_gpu, shuffle=False)
    return test_loader, nscene
