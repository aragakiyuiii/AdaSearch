# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils import *
from utils.pytorch_utils import data_prefetcher, dual_data_prefetcher
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
from tensorboardX import SummaryWriter


class RunConfig:

    def __init__(self, dataset_location, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency, print_frequency):
        self.dataset_location = dataset_location
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
            if epoch < 5:
                lr = ((epoch + 1) / 5) * self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'imagenet':
                from data_providers.imagenet import ImagenetDataProvider
                self._data_provider = ImagenetDataProvider(**self.data_config)
            elif self.dataset == 'imagenet_100':
                from data_providers.imagenet_100 import imagenet_100DataProvider
                self._data_provider = imagenet_100DataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0},
                ], lr=self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True, teacher_model=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0

        # initialize model (default)
        self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # move network to GPU if available
        self.device = torch.device('cuda:0')
        self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)

        if teacher_model is not None:
            self.teacher_model = teacher_model
            self.teacher_model.init_model(run_config.model_init, run_config.init_div_groups)
            self.device = torch.device('cuda:0')
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)
            self.teacher_model.to(self.device)
        else:
            self.teacher_model = None

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                self.net.module.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.module.get_parameters(keys, mode='include'),  # parameters without weight decay
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())

        self.print_net_info()

    """ save path and log path """

    @property
    def save_path(self):  # search checkpoint save location
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net
        net.eval()
        input_var = torch.zeros(data_shape)
        input_var = input_var.cuda(non_blocking=True)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop

    def print_net_info(self, measure_latency=None):
        # network architecture
        if self.out_log:
            print(self.net)

        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            self.write_log('Total training params: %.2fM' % (total_params / 1e6), prefix='net_info')
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                self.write_log("=> loaded checkpoint '{}'".format(model_fname), prefix='loaded_checkpoint')
        except Exception as e:
            if self.out_log:
                self.write_log('fail to load checkpoint from %s because %s' % (self.save_path, e),
                               prefix='fail_to_load_checkpoint')

    def load_teacher_model(self, model_fname=None):
        assert self.teacher_model is not None
        assert model_fname is not None
        if self.out_log:
            print("=> loading teacher_model checkpoint '{}'".format(model_fname))
        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')
        self.teacher_model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        if self.out_log:
            self.write_log("=> loaded teacher_model checkpoint '{}'".format(model_fname), prefix='loaded_checkpoint')

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        # json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4, default=set_to_list)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4, default=set_to_list)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
            print("USE: test_loader")
        else:
            data_loader = self.run_config.valid_loader

        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()

        net = net.cuda()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                output = net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train(self):
        best_acc = 0
        nBatch = len(self.run_config.train_loader)
        writerTf = SummaryWriter(comment='retrain')
        self.write_log('tensorboardX logdir:%s' % writerTf.logdir, prefix='tensorboardX_logdir')
        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            localtime = time.localtime()
            self.write_log(time.strftime("%Y-%m-%d %H:%M:%S", localtime), prefix='retrain')
            self.write_log('\n' + '-' * 30 + 'Train epoch: %d' % (epoch + 1) + '-' * 30 + '\n', prefix='retrain')

            train_acc_top1, train_acc_top5, train_losses, lr = self.train_run_manager_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch)
            )
            writerTf.add_scalar('Train top1', train_acc_top1.avg, epoch)
            writerTf.add_scalar('Train top5', train_acc_top5.avg, epoch)
            writerTf.add_scalar('Train loss', train_losses.avg, epoch)
            writerTf.add_scalar('lr', lr, epoch)

            self.write_log(
                'Train top1 {:.4f}, Train top5 {:.4f}, Train loss {:.4f}, lr {:.4f}'.format(
                    train_acc_top1.avg, train_acc_top5.avg, train_losses.avg, lr), prefix='retrain')
            if epoch % 3 == 0 or epoch > 100:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
                writerTf.add_scalar('Val loss', val_loss, epoch)
                writerTf.add_scalar('Val top1', val_acc, epoch)
                writerTf.add_scalar('Val top5', val_acc5, epoch)
                self.write_log(
                    'val loss {:.4f}, val_acc {:.4f}, val_acc5 {:.4f}'.format(val_loss, val_acc, val_acc5),
                    prefix='retrain_valid')
                if val_acc > best_acc:
                    self.save_model({
                        'epoch': epoch,
                        'weight_optimizer': self.optimizer.state_dict(),
                        'state_dict': self.net.module.state_dict(),
                    }, is_best=True)
                    best_acc = val_acc
                self.save_model({
                    'epoch': epoch,
                    'weight_optimizer': self.optimizer.state_dict(),
                    'state_dict': self.net.module.state_dict(),
                })

    def train_run_manager_one_epoch(self, adjust_lr_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        lr = AverageMeter()

        # switch to train mode
        self.net.train()
        data_loader = self.run_config.train_loader
        prefetcher = data_prefetcher(data_loader)
        images, labels = prefetcher.next()
        i = 0
        end = time.time()
        while images is not None:
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            lr.update(new_lr)

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            images, labels = prefetcher.next()
            i += 1
        return top1, top5, losses, lr.avg


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
