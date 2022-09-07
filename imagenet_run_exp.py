# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import numpy as np
import os, sys
import json
from thop import profile

import torch

from models import *
from run_manager import RunManager
import warnings
import time
from models import get_net_by_name
from utils.pytorch_utils import create_exp_dir, check_dir_exists, console_log

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='output/proxyless-grad14,4,4,4,1/learned_net/')
parser.add_argument('--dataset_location', type=str, default='/your_dataset/imagenet/', help='imagenet dataset path')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')

parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--load_search_weight', action='store_true')
parser.add_argument('--latency', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=125)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='imagenet_100',
                    choices=['imagenet_100', 'imagenet'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--valid_size', type=int, default=None)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='strong', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument(
    '--net', type=str, default='proxyless_mobile',
    choices=['proxyless_gpu', 'proxyless_cpu', 'proxyless_mobile', 'proxyless_mobile_14']
)

args = parser.parse_args()
args.save_env = 'env_dir/retrain-{}-{}'.format(args.train_batch_size, time.strftime("%Y%m%d-%H%M%S"))
args.target_hardware = None
if args.target_hardware is not None:
    args.save_env = 'env_dir/retrain-{}-{}-{}'.format(args.train_batch_size, args.target_hardware,
                                                      time.strftime("%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.makedirs(args.path, exist_ok=True)

    ## prepare run config
    run_config_path = '%s/run.config' % args.path
    if os.path.isfile(run_config_path):
        print('load run config from file')
        run_config = json.load(open(run_config_path, 'r'))
        run_config['init_lr'] = args.init_lr
        run_config['valid_size'] = None
        run_config['n_epochs'] = args.n_epochs
        run_config['train_batch_size'] = args.train_batch_size
        run_config['test_batch_size'] = args.test_batch_size
        run_config['n_worker'] = args.n_worker
        run_config['dataset_location'] = args.dataset_location
        run_config['dataset'] = args.dataset
        print('run_config is ', run_config)
        run_config = ImagenetRunConfig(**run_config)
        if args.valid_size:
            run_config.valid_size = args.valid_size
    else:
        # build run config from args
        args.lr_schedule_param = None
        args.opt_param = {
            'momentum': args.momentum,
            'nesterov': not args.no_nesterov,
        }
        if args.no_decay_keys == 'None':
            args.no_decay_keys = None
        run_config = ImagenetRunConfig(
            **args.__dict__
        )

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path = '%s/net.config' % args.path
    net_config = json.load(open(net_config_path, 'r'))
    net = get_net_by_name(net_config['name']).build_from_config(net_config)
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(net, inputs=(input,))
    print('macs:', macs / 1e6)
    # build run manager
    run_manager = RunManager(args.path, net, run_config)
    run_manager.save_config(print_info=True)

    # load checkpoints
    init_path = '%s/init.pth.tar' % args.path
    if args.resume:
        run_manager.load_model()
    elif os.path.isfile(init_path) and args.load_search_weight:
        if torch.cuda.is_available():
            checkpoint = torch.load(init_path)
        else:
            checkpoint = torch.load(init_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        run_manager.net.module.load_state_dict(checkpoint)
    else:
        run_manager.write_log('Random initialization and without load_model', prefix='without_load_model')

    # train
    try:
        flops = run_manager.net_flops()
        run_manager.write_log('normal_net flops: %.4fM' % (flops / 1e6), prefix='net_info')
    except Exception as e:
        print('Exception about measure flops:', e)

    run_manager.train()
    run_manager.save_model()

    output_dict = {}
    # validate
    if run_config.valid_size:
        loss, acc1, acc5 = run_manager.validate(is_test=False, return_top5=True)
        log = 'valid_loss: %f\t valid_acc1: %f\t valid_acc5: %f' % (loss, acc1, acc5)
        run_manager.write_log(log, prefix='valid')
        output_dict = {
            **output_dict,
            'valid_loss': ' % f' % loss, 'valid_acc1': ' % f' % acc1, 'valid_acc5': ' % f' % acc5,
            'valid_size': run_config.valid_size
        }

    # test
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)
