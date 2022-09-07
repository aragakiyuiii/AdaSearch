# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import argparse
import glob, os, warnings, sys
import numpy as np

from models import ImagenetRunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets
from utils.pytorch_utils import create_exp_dir, check_dir_exists, console_log, get_out_channel_settings, get_expand_rate_settings

warnings.filterwarnings('ignore')
# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {'1.00': 80, },
    'gpu8': {'1.00': 80, },
}

parser = argparse.ArgumentParser()
parser.add_argument('--warmup', action='store_true', help='if have not warmup, please set it True')
parser.add_argument('--path', type=str, default='./output/proxyless-', help='checkpoint save path')
parser.add_argument('--save_env', type=str, default='EXP', help='experiment time name')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true', help='load last checkpoint')  # load last checkpoint
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--previous_total_epoch_sum', default=None, type=int,
                    help='sum of previous period total epoch numbers(including warmup and search)')
parser.add_argument('--search_epoch', default=None, type=int, help='epoch number for search')
parser.add_argument('--period', type=str, default='b0', choices=['b0', 'b1', 'b2', 'b3', 'b4'],
                    help='set the period of search')
parser.add_argument('--wo_progressive', action='store_true', help='if use, it will directly search any period')
parser.add_argument('--add_se', action='store_true',
                    help='if use, it will apply mobile_inverted_conv_SE as operation candidates')
parser.add_argument('--add_h_swish', action='store_true', help='if use, it will apply h-swish as active function')

""" run config """
parser.add_argument('--dataset_location', type=str, default='/your_dataset/imagenet/',
                    help='dataset path')
parser.add_argument('--n_epochs', type=int, default=75, help="total epoch number for cosine lr_schedule")
parser.add_argument('--init_lr', type=float, default=0.025, help='learning rate for weight parameters')
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--lr_schedule_param', type=int, default=None)
# lr_schedule_param
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet_100'])
parser.add_argument('--train_batch_size', type=int, default=256,
                    help="1024 for 1 GPU, 2048 for 2 GPUs, 4096 for 4 GPUs")
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--valid_size', type=int, default=50000, help='number of valid images in search')

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32, help='n_worker for data_loader')
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--width_stages', type=str, default='28,40,96,128,216,216',
                    help="width (output channels) of each cell stage in the block, also last_channel = make_divisible(400 * width_mult, 8) if width_mult > 1.0 else 400")
parser.add_argument('--n_cell_stages', type=str, default='3,3,5,5,6,3',
                    help="maximum number possible of cells in each cell stage")
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1', help="stride of each cell stage in the block")
parser.add_argument('--width_mult', type=float, default=1.0, help="the scale factor of width")
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)
""" period config """
parser.add_argument('--b0', type=str, default='1,2,3,2,2,1', help="number of cells in b0 period")  # 11
parser.add_argument('--b1', type=str, default='2,3,4,3,2,1', help="number of cells in b1 period")  # 15
parser.add_argument('--b2', type=str, default='2,3,3,4,4,2', help="number of cells in b2 period")  # 18
parser.add_argument('--b3', type=str, default='3,3,4,5,4,2', help="number of cells in b3 period")  # 21
parser.add_argument('--b4', type=str, default='3,3,5,5,6,3', help="number of cells in b4 period")  # 25
parser.add_argument('--b0_width', type=str, default='28,40,96,128,184,184', help="number of cells in b0 period")
parser.add_argument('--b1_width', type=str, default='32,48,112,136,224,256', help="number of cells in b1 period")
parser.add_argument('--b2_width', type=str, default='36,56,112,144,256,256', help="number of cells in b2 period")
parser.add_argument('--b3_width', type=str, default='36,64,128,160,256,336', help="number of cells in b3 period")
parser.add_argument('--b4_width', type=str, default='40,64,128,160,256,336', help="number of cells in b4 period")
parser.add_argument('--this_period_n_cell_stages', type=str, default='2,2,2,1,2,1',
                    help="number of cells in this period")
parser.add_argument('--fix_depth', action='store_true', help='if fix_depth, the operation candidates will not use Zero')
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'], help='gradient-based or rl-based')
parser.add_argument('--warmup_epochs', type=int, default=25)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3, help='learning rate for arch-alpha parameters')
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['sample_flops', 'expected_flops', None],
                    help='expected_flops use weighted sum flops by all ops probability. sample_flops only sum flops of sampled ops.')
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5,
                    help='in one epoch, Take a one arch-params update iteration every 5 weight params update iterations')
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='gumbel_1_path', choices=['gumbel_1_path', 'gumbel_2_path'],
                    help="choose the Mixed operations active mode. Gumbel will acivate 1 path. 2-path will acivate 2 path.")
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='add#linear', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=2e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
parser.add_argument('--blocks_expand_ratio_step', help='gpu available', default=None)
parser.add_argument('--blocks_out_channels_step', help='gpu available', default=None)
parser.add_argument('--knowledge_distillation', action='store_true',
                    help="if knowledge_distillation, the will use teacher model and soft_label for training")
parser.add_argument('--teacher_model_period', type=str, default='b0', choices=['b0', 'b1', 'b2', 'b3', 'b4'],
                    help='set the period of teacher model period. Usually it should be last period model')
parser.add_argument('--teacher_model_ckpt_path', type=str, default='./output/expname/checkpoint_name.pth.tar',
                    help='checkpoint save path')
parser.add_argument('--unify_warmup_train', action='store_true',
                    help="if unify_warmup_train, the warmup and train will share the same for-loop")
parser.add_argument('--freeze_old_block_tau', action='store_true',
                    help='if freeze_old_block_tau, old block tau of last period will always be 0.1')
parser.add_argument('--wo_alpha_kd', action='store_true', help='if use, alpha will not updated by knowledge_distillation')
parser.add_argument('--old_period', type=str, default='b0', choices=['b0', 'b1', 'b2', 'b3', 'b4'],
                    help='set the period of last model period. It helps freeze_old_block_tau.')
parser.add_argument('--exp_name', type=str, default='archlr_period', help="name the log folder")
parser.add_argument('--test_derive_normal_net', action='store_true', help="just for unit test")
args = parser.parse_args()
args.save_env = 'env_dir/search-{}-{}-{}'.format(args.arch_algo, args.train_batch_size, time.strftime("%Y%m%d-%H%M%S"))
args.path = "./output/" + args.exp_name
check_dir_exists('./output/')
check_dir_exists('./env_dir/')
if args.unify_warmup_train:
    # n_epochs = previous_period_warmup_epoch + previous_period_search_epoch + this_period_warmup_epoch +this_period_search_epoch
    # search for small arch
    if args.period == 'b0':
        args.search_epoch = 75 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b0
        args.width_stages = args.b0_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.90
        if args.previous_total_epoch_sum is not None:
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
        else:
            # previous_total_epoch_sum = 0 by default
            args.previous_total_epoch_sum = 0
            args.n_epochs = args.search_epoch + args.warmup_epochs
    # search for medium arch
    elif args.period == 'b1':
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b1
        args.width_stages = args.b1_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.50
        if args.previous_total_epoch_sum is not None:
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
        else:
            if args.wo_progressive:
                # increase b1 search epoch to 100 when directly search
                args.previous_total_epoch_sum = 0
                args.search_epoch = 100 if args.search_epoch is None else args.search_epoch
            else:
                # previous_total_epoch_sum = 100 by default
                args.previous_total_epoch_sum = 100
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
    # search for large arch
    elif args.period == 'b2':
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b2
        args.width_stages = args.b2_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.50
        if args.previous_total_epoch_sum is not None:
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
        else:
            if args.wo_progressive:
                # increase b2 search epoch to 100 when directly search
                args.previous_total_epoch_sum = 0
                args.search_epoch = 100 if args.search_epoch is None else args.search_epoch
            else:
                # previous_total_epoch_sum = 175 by default
                args.previous_total_epoch_sum = 175
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
    elif args.period == 'b3':
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b3
        args.width_stages = args.b3_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.20
        if args.previous_total_epoch_sum is not None:
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
        else:
            if args.wo_progressive:
                # increase b3 search epoch to 100 when directly search
                args.previous_total_epoch_sum = 0
                args.search_epoch = 100 if args.search_epoch is None else args.search_epoch
            else:
                # previous_total_epoch_sum = 250 by default
                args.previous_total_epoch_sum = 250
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
    elif args.period == 'b4':
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b4
        args.width_stages = args.b4_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.10
        if args.previous_total_epoch_sum is not None:
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
        else:
            if args.wo_progressive:
                # increase b3 search epoch to 100 when directly search
                args.previous_total_epoch_sum = 0
                args.search_epoch = 100 if args.search_epoch is None else args.search_epoch
            else:
                # previous_total_epoch_sum = 250 by default
                args.previous_total_epoch_sum = 325
            args.n_epochs = args.previous_total_epoch_sum + args.search_epoch + args.warmup_epochs
    else:
        raise NotImplementedError
else:
    # n_epochs = previous_period_search_epoch + previous_period_search_epoch
    if args.period == 'b0':
        args.n_epochs = 75
        args.search_epoch = 75 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b0
        args.width_stages = args.b0_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.90
    # search for medium arch
    elif args.period == 'b1':
        args.n_epochs = 125
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b1
        args.width_stages = args.b1_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.50
    # search for large arch
    elif args.period == 'b2':
        args.n_epochs = 175
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b2
        args.width_stages = args.b2_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.50
    elif args.period == 'b3':
        args.n_epochs = 225
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b3
        args.width_stages = args.b3_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.20
    elif args.period == 'b4':
        args.n_epochs = 275
        args.search_epoch = 50 if args.search_epoch is None else args.search_epoch
        args.this_period_n_cell_stages = args.b4
        args.width_stages = args.b4_width
        args.grad_reg_loss_lambda = args.grad_reg_loss_lambda * 0.10
    else:
        raise NotImplementedError
# see flops-latency
console = console_log(logs_path=args.path)
args.save_env = 'env_dir/search-{}-{}-{}-{}'.format(args.arch_algo, args.train_batch_size, args.target_hardware,
                                                    time.strftime("%Y%m%d-%H%M%S"))
console.write_log('args.path: %s' % args.path, prefix='console')
console.write_log('args.save_env: %s' % args.save_env, prefix='console')
console.write_log('args.exp_name: %s' % args.exp_name, prefix='console')
try:
    create_exp_dir(args.save_env, scripts_to_save=glob.glob('*.py'))
except Exception as e:
    console.write_log('Exception of create_exp_dir: %s' % e, prefix='console')

if __name__ == '__main__':
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = ImagenetRunConfig(
        **args.__dict__
    )

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.this_period_n_cell_stages = [int(val) for val in args.this_period_n_cell_stages.split(',')]
    args.conv_candidates = ['3x3_MBConv6', '5x5_MBConv6', ]

    if args.add_se:
        se_conv_candidates = ['3x3_MBConv6_se', '5x5_MBConv6_se']
        args.conv_candidates = args.conv_candidates + se_conv_candidates
    if args.add_h_swish:
        se_conv_candidates = args.conv_candidates
        args.conv_candidates = args.conv_candidates + [op + '_h_swish' for op in se_conv_candidates]

    # 2,2,2,1,2,1
    # 3,3,4,6,6,2
    args.blocks_expand_ratio_step = get_expand_rate_settings(period=args.period)
    args.blocks_out_channels_step = get_out_channel_settings(period=args.period)
    assert sum(args.n_cell_stages) == len(args.blocks_expand_ratio_step), \
        'number of blocks_expand_rate_configs should be equal to blocks number'
    assert sum(args.n_cell_stages) == len(args.blocks_out_channels_step), \
        'number of blocks_out_channels_configs should be equal to blocks number'
    if args.freeze_old_block_tau:
        if args.old_period == 'b0':
            old_period_n_cell_stages = args.b0
        elif args.old_period == 'b1':
            old_period_n_cell_stages = args.b1
        elif args.old_period == 'b2':
            old_period_n_cell_stages = args.b2
        elif args.old_period == 'b3':
            old_period_n_cell_stages = args.b3
        else:
            # no period is higher than b4, so b4 can not be last period
            raise NotImplementedError
        old_period_n_cell_stages = [int(val) for val in old_period_n_cell_stages.split(',')]
    else:
        old_period_n_cell_stages = None
    super_net = SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, period=args.period,
        this_period_n_cell_stages=args.this_period_n_cell_stages, fix_depth=args.fix_depth,
        blocks_expand_ratio_step=args.blocks_expand_ratio_step,
        blocks_out_channels_step=args.blocks_out_channels_step,
        freeze_old_block_tau=args.freeze_old_block_tau, old_period_n_cell_stages=old_period_n_cell_stages
    )
    if args.knowledge_distillation:
        if args.teacher_model_period == 'b0':
            teacher_width_stages = args.b0_width
            teacher_model_this_period_n_cell_stages = args.b0
        elif args.teacher_model_period == 'b1':
            teacher_width_stages = args.b1_width
            teacher_model_this_period_n_cell_stages = args.b1
        elif args.teacher_model_period == 'b2':
            teacher_width_stages = args.b2_width
            teacher_model_this_period_n_cell_stages = args.b2
        elif args.teacher_model_period == 'b3':
            teacher_width_stages = args.b3_width
            teacher_model_this_period_n_cell_stages = args.b3
        else:
            # no period is higher than b4, so b4 can not be teacher
            raise NotImplementedError
        teacher_model_blocks_out_channels_step = get_out_channel_settings(period=args.teacher_model_period)
        teacher_model_blocks_expand_ratio_step = get_expand_rate_settings(period=args.teacher_model_period)
        teacher_width_stages = [int(val) for val in teacher_width_stages.split(',')]
        teacher_model_this_period_n_cell_stages = [int(val) for val in
                                                   teacher_model_this_period_n_cell_stages.split(',')]
        teacher_model = SuperProxylessNASNets(
            width_stages=teacher_width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes,
            width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, period=args.teacher_model_period,
            this_period_n_cell_stages=teacher_model_this_period_n_cell_stages, fix_depth=args.fix_depth,
            blocks_expand_ratio_step=teacher_model_blocks_expand_ratio_step,
            blocks_out_channels_step=teacher_model_blocks_out_channels_step,
        )
    else:
        teacher_model = None
        console.write_log('do not use knowledge_distillation', prefix='console')

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    elif 'flops' in args.target_hardware:
        args.ref_value = ref_values['flops']['%.2f' % args.width_mult]
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig

        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError

    print('Run config:')
    console.write_log('Run config:', prefix='console')
    for k, v in run_config.config.items():
        console.write_log('\t%s: %s' % (k, v), prefix='console')
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        console.write_log('\t%s: %s' % (k, v), prefix='console')

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config,
                                                   warmup=args.warmup, teacher_model=teacher_model, wo_alpha_kd=args.wo_alpha_kd)
    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception as e:
            console.write_log('Exception of load_model: %s' % e, prefix='fail_to_load_model')
    else:
        console.write_log('without load_model', prefix='without_load_model')
    if args.knowledge_distillation:
        arch_search_run_manager.load_teacher_model(model_fname=args.teacher_model_ckpt_path)
    if args.test_derive_normal_net:
        arch_search_run_manager.derive_normal_net()
        sys.exit(0)

    # joint training
    if args.unify_warmup_train:
        if arch_search_run_manager.warmup:
            args.warmup_epochs = args.warmup_epochs
        else:
            args.warmup_epochs = 0
        arch_search_run_manager.unify_warmup_train(previous_epoch=args.previous_total_epoch_sum,
                                                   warmup_epochs=args.warmup_epochs, last_epoch=args.n_epochs)
    else:
        if arch_search_run_manager.warmup:
            arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)
        arch_search_run_manager.train(last_epoch=args.n_epochs)
