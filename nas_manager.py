# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from run_manager import *
import time
from tensorboardX import SummaryWriter
from utils.pytorch_utils import data_prefetcher, dual_data_prefetcher, cross_entropy_loss_with_soft_target
from thop import profile
import math


class ArchSearchConfig:

    def __init__(self, arch_init_type, arch_init_ratio, arch_opt_type, arch_lr,
                 arch_opt_param, arch_weight_decay, target_hardware, ref_value):
        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, nBatch):
        raise NotImplementedError

    def build_optimizer(self, params):
        """
        :param params: architecture parameters
        :return: arch_optimizer
        """
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError


class GradientArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 grad_update_arch_param_every=20, grad_update_steps=1, grad_binary_mode='gumbel_2_path',
                 grad_data_batch=None,
                 grad_reg_loss_type=None, grad_reg_loss_params=None, **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params

        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if nBatch < self.update_arch_param_every:
            self.update_arch_param_every = nBatch
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self, ce_loss, expected_value):
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == 'mul#log':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('Do not support: %s' % self.reg_loss_type)


class ArchSearchRunManager:

    def __init__(self, path, super_net, run_config: RunConfig, arch_search_config: ArchSearchConfig, warmup=False,
                 teacher_model=None, wo_alpha_kd=False):
        # init weight parameters & build weight_optimizer
        self.run_manager = RunManager(path, super_net, run_config, True, teacher_model=teacher_model)

        self.arch_search_config = arch_search_config

        # init architecture parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,
        )

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())

        self.warmup = warmup
        self.warmup_epoch = 0
        self.start_epoch = 0
        self.wo_alpha_kd = wo_alpha_kd

    @property
    def net(self):
        return self.run_manager.net.module

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            self.write_log("=> loading checkpoint '{}'".format(model_fname), prefix='console')

        checkpoint = torch.load(model_fname, map_location=torch.device('cpu'))  # reduce max gpu-cost

        model_dict = self.net.state_dict()
        # # when change the period of supernet, filter out irrelevant keys from last-period model
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict, strict=False)
        if self.run_manager.out_log:
            self.write_log("=> loaded checkpoint '{}'".format(model_fname), prefix='console')

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
            self.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            try:
                self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
            except Exception as e:
                # when change the period of supernet, MAYBE loaded state dict contains a parameter group that doesn't match the size of optimizer's group
                self.write_log("Exception about weight_optimizer state_dict loading: %s" % e, prefix='console')
        if 'arch_optimizer' in checkpoint:
            try:
                self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
            except Exception as e:
                # when change the period of supernet, MAYBE loaded state dict contains a parameter group that doesn't match the size of optimizer's group
                self.write_log("Exception about arch_optimizer state_dict loading: %s" % e, prefix='console')
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch'] + 1

    def load_teacher_model(self, model_fname=None):
        self.run_manager.load_teacher_model(model_fname=model_fname)

    def get_normal_net(self):
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        self.write_log('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6), prefix='normal_net')
        os.makedirs(os.path.join(self.run_manager.path, self.net.period + '_learned_net'), exist_ok=True)
        json.dump(normal_net.config,
                  open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/net.config'), 'w'),
                  indent=4,
                  default=set_to_list)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/run.config'), 'w'), indent=4,
            default=set_to_list
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, self.net.period + '_learned_net/init.pth.tar')
        )

    def derive_normal_net(self):
        normal_net = self.net.cpu().convert_to_normal_net()
        self.write_log('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6), prefix='train')
        os.makedirs(os.path.join(self.run_manager.path, self.net.period + '_learned_net'), exist_ok=True)
        normal_net_config = normal_net.config()
        json.dump(normal_net_config,
                  open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/net.config'), 'w'),
                  indent=4,
                  default=set_to_list)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/run.config'), 'w'), indent=4,
            default=set_to_list
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, self.net.period + '_learned_net/init.pth.tar')
        )

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        valid_res = self.run_manager.validate(is_test=False, return_top5=True)
        flops, latency = 0, 0
        return valid_res, flops, latency

    def gradient_step(self, images, labels, kd_ratio=0.5):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        if self.arch_search_config.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_batch_size
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.data_batch
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # Mix edge mode
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        time2 = time.time()  # time
        # compute output

        output = self.run_manager.net(images)
        time3 = time.time()  # time
        # loss
        ce_loss = self.run_manager.criterion(output, labels)
        if self.run_manager.teacher_model is not None:
            if self.wo_alpha_kd:
                ce_loss = ce_loss
            else:
                kd_ratio = kd_ratio
                self.run_manager.teacher_model.module.set_mix_op_tau(tau=0.1, tau_soft=0.1)
                self.run_manager.teacher_model.module.reset_probs()
                self.run_manager.teacher_model.train()
                with torch.no_grad():
                    soft_logits = self.run_manager.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)
                    kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                ce_loss = kd_ratio * kd_loss + (1 - kd_ratio) * ce_loss
        else:
            ce_loss = ce_loss
        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == 'expected_flops':
            data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.expected_flops(input_var)
        elif self.arch_search_config.target_hardware == 'sample_flops':
            data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.sample_flops(input_var)
        else:
            raise NotImplementedError
        loss = self.arch_search_config.add_regularization_loss(ce_loss, expected_value)
        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()

        self.arch_optimizer.step()

        time4 = time.time()  # time
        self.write_log(
            '(%.4f, %.4f, %.4f)' % (time2 - time1, time3 - time2, time4 - time3), 'gradient',
            should_print=False, end='\t'
        )
        return loss.data.item(), expected_value.item() if expected_value is not None else None

    def warm_up(self, warmup_epochs=25):
        MixedEdge.MODE = self.arch_search_config.binary_mode

        writerTf = SummaryWriter(comment='warm_up')
        self.write_log('tensorboardX logdir: %s' % (writerTf.logdir), prefix='tensorboardX_logdir')
        self.write_log('self.warmup_epoch=%d, warmup_epochs=%d' % (self.warmup_epoch, warmup_epochs),
                       prefix='epoch_info')
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader

        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch
        checkpoint = None
        kd_ratio = 0.5
        for epoch in range(self.warmup_epoch, warmup_epochs):
            self.write_log('\n' + '-' * 30 + 'Warmup epoch: %d' % (epoch + 1) + '-' * 30 + '\n', prefix='warm_up')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            epoch_time = time.time()
            # switch to train mode
            self.run_manager.net.train()
            prefetcher = dual_data_prefetcher(data_loader)
            images, labels, arch_images, arch_labels = prefetcher.next()
            i = 0
            end = time.time()
            while images is not None:
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                self.net.reset_probs()
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                # compute output
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)

                # knowledge_distillation
                if self.run_manager.teacher_model is not None:
                    kd_ratio = max(0.5 - 0.5 * (epoch - self.warmup_epoch) / (warmup_epochs - 1 - self.warmup_epoch),
                                   0.01)
                    if i == 0:
                        print('kd_ratio=', kd_ratio)
                    self.run_manager.teacher_model.module.set_mix_op_tau(tau=0.1, tau_soft=0.1)
                    self.run_manager.teacher_model.module.reset_probs()
                    self.run_manager.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = self.run_manager.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    loss = kd_ratio * kd_loss + (1 - kd_ratio) * ce_loss
                else:
                    loss = ce_loss
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    self.write_log(batch_log, 'train')
                images, labels, arch_images, arch_labels = prefetcher.next()
                i += 1
            (val_loss, val_top1, val_top5), flops, latency = self.validate()

            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\tflops: {5:.1f}M'. \
                format(epoch + 1, warmup_epochs, val_loss, val_top1, val_top5, flops / 1e6, top1=top1, top5=top5)
            if self.arch_search_config.target_hardware not in [None, 'flops']:
                val_log += '\t' + self.arch_search_config.target_hardware + ': %.3fms' % latency
            self.write_log(val_log, 'valid')
            self.warmup = epoch + 1 < warmup_epochs

            writerTf.add_scalar('train loss', losses.avg, epoch)
            writerTf.add_scalar('train top1', top1.avg, epoch)
            writerTf.add_scalar('train top5', top5.avg, epoch)
            writerTf.add_scalar('Valid Loss', val_loss, epoch)
            writerTf.add_scalar('val_top1', val_top1, epoch)
            writerTf.add_scalar('val_top5', val_top5, epoch)

            # ========== print time ==========
            this_epoch_time = time.time() - epoch_time
            self.write_log('epoch time {:.3f} min'.format(this_epoch_time / 60), prefix='warm_up')
            localtime = time.localtime()
            self.write_log(time.strftime("%Y-%m-%d %H:%M:%S", localtime), prefix='warm_up')

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'warmup': self.warmup,
                'warmup_epoch': epoch,
                'state_dict': state_dict,
            }
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')
        self.run_manager.save_model(checkpoint, model_name=str(warmup_epochs) + 'warmup.pth.tar')
        writerTf.close()

    def train(self, last_epoch=75):
        MixedEdge.MODE = self.arch_search_config.binary_mode
        lr, exp_value = None, None
        writerTf = SummaryWriter(comment='search')
        self.write_log('tensorboardX logdir: %s' % (writerTf.logdir), prefix='tensorboardX_logdir')
        self.write_log('self.start_epoch=%d, last_epoch=%d' % (self.start_epoch, last_epoch), prefix='epoch_info')
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)

        arch_param_num = len(list(self.net.architecture_parameters()))
        weight_param_num = len(list(self.net.weight_parameters()))

        self.write_log('#arch_params: %d\t#weight_params: %d' %
                       (arch_param_num, weight_param_num), prefix='train')

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        tau_max = 10
        tau_min = 0.1
        tau_soft_max = 5.0
        soft_exponential = math.pow((0.1 / tau_soft_max), (1.0 / (last_epoch - self.start_epoch)))

        kd_ratio = 0.5
        for epoch in range(self.start_epoch, last_epoch):
            self.write_log('\n' + '-' * 30 + 'Train epoch: %d' % (epoch + 1) + '-' * 30 + '\n', prefix='train')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()
            tau_soft = tau_soft_max * (soft_exponential ** (epoch - self.start_epoch))
            self.net.set_mix_op_tau(tau=max(tau_soft, 0.1), tau_soft=tau_soft)
            end = time.time()
            epoch_time = time.time()
            localtime = time.localtime()
            self.write_log(time.strftime("%Y-%m-%d %H:%M:%S", localtime) + '\n', prefix='train')
            prefetcher = dual_data_prefetcher(data_loader)
            images, labels, arch_images, arch_labels = prefetcher.next()
            i = 0
            end = time.time()
            while images is not None:
                data_time.update((time.time() - end) / 60)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                self.net.reset_probs()
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                # compute output
                output = self.run_manager.net(images)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)
                    # measure accuracy and record loss
                # knowledge_distillation
                if self.run_manager.teacher_model is not None:
                    kd_ratio = max(0.5 - 0.5 * (epoch - self.start_epoch) / (last_epoch - 1 - self.start_epoch), 0.01)
                    self.run_manager.teacher_model.module.set_mix_op_tau(tau=0.1, tau_soft=0.1)
                    self.run_manager.teacher_model.module.reset_probs()
                    self.run_manager.teacher_model.train()
                    if i == 0:
                        print('kd_ratio=', kd_ratio)
                    with torch.no_grad():
                        soft_logits = self.run_manager.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    loss = kd_ratio * kd_loss + (1 - kd_ratio) * ce_loss
                else:
                    loss = ce_loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters

                # skip architecture parameter updates in the first epoch
                if epoch >= 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        self.net.reset_probs()
                        arch_loss, exp_value = self.gradient_step(arch_images, arch_labels, kd_ratio=kd_ratio)
                # measure elapsed time
                batch_time.update((time.time() - end) / 60)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    self.write_log('Train [{0}][{1}/{2}]\t' \
                                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                   'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                   'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                   'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                   'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                   'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                                   format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                                          losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr), prefix='train')
                images, labels, arch_images, arch_labels = prefetcher.next()
                i += 1
            writerTf.add_scalar('Train Loss', losses.avg, epoch)
            writerTf.add_scalar('Entropy', entropy.avg, epoch)
            writerTf.add_scalar('train top1 acc', top1.avg, epoch)
            writerTf.add_scalar('train top5 acc', top5.avg, epoch)
            writerTf.add_scalar('lr', lr, epoch)
            if exp_value is not None:
                writerTf.add_scalar('exp_flops', exp_value, epoch)

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                if exp_value is not None:
                    flops = exp_value
                self.write_log('Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                               'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                               'Entropy {entropy.val:.5f}\t' \
                               'Latency-{6}: {7:.3f}ms\texp_flops: {8:.2f}M'. \
                               format(epoch + 1, last_epoch, val_loss, val_top1,
                                      self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                                      latency, flops / 1e6, entropy=entropy, top1=top1, top5=top5), prefix='train')
                writerTf.add_scalar('Valid Loss', val_loss, epoch)
                writerTf.add_scalar('val top1 acc', val_top1, epoch)
                writerTf.add_scalar('val top5 acc', val_top5, epoch)
            # save model
            self.run_manager.save_model({
                'warmup': self.warmup,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })
            this_epoch_time = time.time() - epoch_time
            self.write_log('epoch time {:.3f} min'.format(this_epoch_time / 60), prefix='train')

        # save a period model(small, medium, or large)
        self.run_manager.save_model({
            'warmup': True,
            'epoch': last_epoch,
            'weight_optimizer': self.run_manager.optimizer.state_dict(),
            'arch_optimizer': self.arch_optimizer.state_dict(),
            'state_dict': self.net.state_dict()
        }, model_name=self.net.period + '_checkpoint.pth.tar')
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        self.write_log('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6), prefix='train')
        os.makedirs(os.path.join(self.run_manager.path, self.net.period + '_learned_net'), exist_ok=True)
        normal_net_config = normal_net.config()
        json.dump(normal_net_config,
                  open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/net.config'), 'w'),
                  indent=4,
                  default=set_to_list)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/run.config'), 'w'), indent=4,
            default=set_to_list
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, self.net.period + '_learned_net/init.pth.tar')
        )
        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(normal_net, inputs=(input,))
        print('macs:', macs / 1e6)
        self.write_log('normal_net_flops {:.4f} M'.format(macs / 1e6), prefix='normal_net_flops')
        self.write_log('normal_net_params {:.4f} M'.format(params / 1e6), prefix='normal_net_params')

    def unify_warmup_train(self, previous_epoch=0, warmup_epochs=25, last_epoch=75):
        MixedEdge.MODE = self.arch_search_config.binary_mode
        warmup_lr, lr, exp_value = None, None, None
        writer_unify_warmup_train = SummaryWriter(comment='unify_warmup_train')
        self.write_log('writer_unify_warmup_train tensorboardX logdir:%s' % (writer_unify_warmup_train.logdir),
                       prefix='tensorboardX_logdir')
        warmup = warmup_epochs + previous_epoch
        self.write_log('self.start_epoch=%d, warmup=%d, last_epoch=%d' % (self.start_epoch, warmup, last_epoch),
                       prefix='epoch_info')
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)

        arch_param_num = len(list(self.net.architecture_parameters()))
        weight_param_num = len(list(self.net.weight_parameters()))

        self.write_log('#arch_params: %d\t#weight_params: %d' %
                       (arch_param_num, weight_param_num), prefix='train')

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        tau_max = 10
        tau_min = 0.1
        tau_soft_max = 5.0
        soft_exponential = math.pow((0.1 / tau_soft_max), (1.0 / (last_epoch - warmup)))
        kd_ratio = 0.5
        for epoch in range(self.start_epoch, warmup):
            # warmup period
            self.write_log('\n' + '-' * 30 + 'Warmup epoch: %d' % (epoch + 1) + '-' * 30 + '\n',
                           prefix='unify_warmup_train')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            epoch_time = time.time()
            # switch to train mode
            self.run_manager.net.train()
            prefetcher = dual_data_prefetcher(data_loader)
            images, labels, arch_images, arch_labels = prefetcher.next()
            i = 0
            end = time.time()
            while images is not None:
                data_time.update(time.time() - end)
                # warmup_lr
                warmup_lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                self.net.reset_probs()
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                # compute output
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)

                # knowledge_distillation
                if self.run_manager.teacher_model is not None:
                    kd_ratio = max(0.5 - 0.5 * (epoch - self.start_epoch) / (last_epoch - 1 - self.start_epoch), 0.05)
                    if i == 0:
                        print('kd_ratio=', kd_ratio)
                    self.run_manager.teacher_model.module.set_mix_op_tau(tau=0.1, tau_soft=0.1)
                    self.run_manager.teacher_model.module.reset_probs()
                    self.run_manager.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = self.run_manager.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    loss = kd_ratio * kd_loss + (1 - kd_ratio) * ce_loss
                else:
                    loss = ce_loss
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    self.write_log(batch_log, 'unify_warmup_train')
                images, labels, arch_images, arch_labels = prefetcher.next()
                i += 1
            (val_loss, val_top1, val_top5), flops, latency = self.validate()

            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\tflops: {5:.1f}M'. \
                format(epoch + 1, warmup_epochs, val_loss, val_top1, val_top5, flops / 1e6, top1=top1, top5=top5)
            if self.arch_search_config.target_hardware not in [None, 'flops']:
                val_log += '\t' + self.arch_search_config.target_hardware + ': %.3fms' % latency
            self.write_log(val_log, 'valid')
            self.warmup = epoch + 1 < warmup_epochs

            writer_unify_warmup_train.add_scalar('train loss', losses.avg, epoch)
            writer_unify_warmup_train.add_scalar('train top1', top1.avg, epoch)
            writer_unify_warmup_train.add_scalar('train top5', top5.avg, epoch)
            writer_unify_warmup_train.add_scalar('Valid Loss', val_loss, epoch)
            writer_unify_warmup_train.add_scalar('val_top1', val_top1, epoch)
            writer_unify_warmup_train.add_scalar('val_top5', val_top5, epoch)
            writer_unify_warmup_train.add_scalar('lr', warmup_lr, epoch)

            # ========== print time ==========
            this_epoch_time = time.time() - epoch_time
            self.write_log('epoch time {:.3f} min'.format(this_epoch_time / 60), prefix='warmup')
            localtime = time.localtime()
            self.write_log(time.strftime("%Y-%m-%d %H:%M:%S", localtime), prefix='warmup')
            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)

                self.write_log('Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                               'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                               'Latency-{6}: {7:.3f}ms\tFlops: {8:.2f}M'. \
                               format(epoch + 1, last_epoch, val_loss, val_top1,
                                      self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                                      latency, flops / 1e6, top1=top1, top5=top5), prefix='valid')
                writer_unify_warmup_train.add_scalar('Valid Loss', val_loss, epoch)
                writer_unify_warmup_train.add_scalar('val top1 acc', val_top1, epoch)
                writer_unify_warmup_train.add_scalar('val top5 acc', val_top5, epoch)
            # save model
            self.run_manager.save_model({
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })
            if epoch == (warmup-1):
                self.run_manager.save_model({
                    'epoch': epoch,
                    'weight_optimizer': self.run_manager.optimizer.state_dict(),
                    'state_dict': self.net.state_dict()
                }, model_name=str(warmup) + 'warmup.pth.tar')
        if self.start_epoch >= warmup:
            warmup = self.start_epoch
        else:
            self.run_manager.save_model({
                'epoch': warmup,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            }, model_name=str(warmup) + 'warmup.pth.tar')
        for epoch in range(warmup, last_epoch):
            # in search period
            self.write_log('\n' + '-' * 30 + 'Train epoch: %d' % (epoch + 1) + '-' * 30 + '\n', prefix='train')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()
            tau_soft = tau_soft_max * (soft_exponential ** (epoch - warmup))
            self.net.set_mix_op_tau(tau=max(tau_soft, 0.1), tau_soft=tau_soft)
            epoch_time = time.time()
            localtime = time.localtime()
            self.write_log(time.strftime("%Y-%m-%d %H:%M:%S", localtime) + '\n', prefix='train')
            prefetcher = dual_data_prefetcher(data_loader)
            images, labels, arch_images, arch_labels = prefetcher.next()
            i = 0
            end = time.time()
            while images is not None:
                data_time.update((time.time() - end) / 60)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                self.net.reset_probs()
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                # compute output
                output = self.run_manager.net(images)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)
                    # measure accuracy and record loss

                # knowledge_distillation
                if self.run_manager.teacher_model is not None:
                    kd_ratio = max(0.5 - 0.5 * (epoch - self.start_epoch) / (last_epoch - 1 - self.start_epoch), 0.05)
                    if i == 0:
                        print('kd_ratio=', kd_ratio)
                    self.run_manager.teacher_model.module.set_mix_op_tau(tau=0.1, tau_soft=0.1)
                    self.run_manager.teacher_model.module.reset_probs()
                    self.run_manager.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = self.run_manager.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    loss = kd_ratio * kd_loss + (1 - kd_ratio) * ce_loss
                else:
                    loss = ce_loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters

                # skip architecture parameter updates in the first epoch
                if epoch >= 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        self.net.reset_probs()
                        arch_loss, exp_value = self.gradient_step(arch_images, arch_labels, kd_ratio=kd_ratio)
                # measure elapsed time
                batch_time.update((time.time() - end) / 60)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    self.write_log('Train [{0}][{1}/{2}]\t' \
                                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                   'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                   'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                   'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                   'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                   'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                                   format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                                          losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr),
                                   prefix='train')

                images, labels, arch_images, arch_labels = prefetcher.next()
                i += 1
            writer_unify_warmup_train.add_scalar('Train Loss', losses.avg, epoch)
            writer_unify_warmup_train.add_scalar('Entropy', entropy.avg, epoch)
            writer_unify_warmup_train.add_scalar('train top1 acc', top1.avg, epoch)
            writer_unify_warmup_train.add_scalar('train top5 acc', top5.avg, epoch)
            writer_unify_warmup_train.add_scalar('lr', lr, epoch)
            if exp_value is not None:
                writer_unify_warmup_train.add_scalar('exp_flops', exp_value, epoch)

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                if exp_value is not None:
                    flops = exp_value
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                self.write_log('Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                               'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                               'Entropy {entropy.val:.5f}\t' \
                               'Latency-{6}: {7:.3f}ms\texp_flops: {8:.2f}M'. \
                               format(epoch + 1, last_epoch, val_loss, val_top1,
                                      self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                                      latency, flops / 1e6, entropy=entropy, top1=top1, top5=top5), prefix='valid')

                writer_unify_warmup_train.add_scalar('Valid Loss', val_loss, epoch)
                writer_unify_warmup_train.add_scalar('val top1 acc', val_top1, epoch)
                writer_unify_warmup_train.add_scalar('val top5 acc', val_top5, epoch)
            # save model
            self.run_manager.save_model({
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })
            this_epoch_time = time.time() - epoch_time
            self.write_log('epoch time {:.3f} min'.format(this_epoch_time / 60), prefix='train')
        # over the search period
        # save a period model(small, medium, or large)
        self.run_manager.save_model({
            'epoch': last_epoch,
            'weight_optimizer': self.run_manager.optimizer.state_dict(),
            'arch_optimizer': self.arch_optimizer.state_dict(),
            'state_dict': self.net.state_dict()
        }, model_name=self.net.period + '_checkpoint.pth.tar')
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        self.write_log('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6), prefix='train')
        os.makedirs(os.path.join(self.run_manager.path, self.net.period + '_learned_net'), exist_ok=True)
        normal_net_config = normal_net.config()
        json.dump(normal_net_config,
                  open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/net.config'), 'w'),
                  indent=4,
                  default=set_to_list)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, self.net.period + '_learned_net/run.config'), 'w'), indent=4,
            default=set_to_list
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, self.net.period + '_learned_net/init.pth.tar')
        )
        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(normal_net, inputs=(input,))
        print('macs:', macs / 1e6)
        self.write_log('normal_net_flops {:.4f} M'.format(macs / 1e6), prefix='normal_net_flops')
        self.write_log('normal_net_params {:.4f} M'.format(params / 1e6), prefix='normal_net_params')


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
