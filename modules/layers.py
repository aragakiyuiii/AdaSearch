# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import torch
import torch.nn.functional as F
from utils import *
from utils.pytorch_utils import gumbel_softmax, Hsigmoid, Hswish  # build_activation
from collections import OrderedDict
from torch.nn.parameter import Parameter
from modules.dynamic_op import DynamicPointConv2d, DynamicSeparableConv2d, SwitchableBatchNorm2d
import warnings


def get_list_from_step(low=None, high=None, step=None):
    n_steps = 1 + int((high - low) // step)
    step2list = list(range(n_steps))
    for i in range(n_steps):
        step2list[i] = low + i * step
    # print("step2list", step2list)
    return step2list


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        # build normal block from config
        MBInvertedConvLayer.__name__: NormalMBInvertedConvLayer,
        NormalMBInvertedConvLayer.__name__: NormalMBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    def wider_in_channel(self, wider_out_channel, random_init=False):
        self.conv = wider_in_channel_conv(self.conv, wider_out_channel, random_init=random_init)

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    # @property
    def config(self, in_channel=None):
        if in_channel is not None:
            self.in_channels = in_channel
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)
        self.tau = 1.0
        self.soft_gumbel_out = None

    def wider_out_channel(self, wider_out_channel, random_init=False):
        return None

    def wider_in_channel(self, wider_in_channel, random_init=False):
        return None

    def wider_expand_rate(self, wider_expand_rate, random_init=False):
        return None

    def weight_op(self):
        return None

    def forward(self, x):
        return x

    @property
    def module_str(self):
        return 'Identity'

    # @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            # maybe the config of identity/My2DLayer/my_modules need to be unified
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x, in_channel_effective_num=None):
        return 0, self.forward(x)

    def get_soft_gumbel_out(self):
        return None

    def set_soft_gumbel_out(self, soft_gumbel_out=None):
        return None


class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 expand_ratio=6, mid_channels=None, use_se=False, act_func='relu6', expand_ratio_step=None,
                 out_channels_step=None, stage_head=True,
                 stage_tail=False):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        if expand_ratio == 1:
            # if expand_ratio is 1, don't search expand channel
            self.expand_ratio = expand_ratio
        else:
            self.expand_ratio = expand_ratio_step[1]
        self.mid_channels = mid_channels
        self.use_se = use_se
        self.act_func = act_func
        self.stage_head = stage_head
        self.stage_tail = stage_tail
        self.tau = 5.0

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            stage_head = True
            self.stage_head = True
            self.inverted_bottleneck = None
        else:
            # set alpha_expand_channel
            assert len(expand_ratio_step) == 3, 'expand_ratio_step should have [0.75, 3.25, 0.5]'
            assert self.expand_ratio == expand_ratio_step[1]
            # float list [0.75, 3.25, 0.5]
            self.expand_ratio_step2list = get_list_from_step(low=expand_ratio_step[0], high=expand_ratio_step[1],
                                                             step=expand_ratio_step[2])
            # int list [16, 20, 24, 28]
            self.sub_expand_channel_list = [round(self.in_channels * expand) for expand in self.expand_ratio_step2list]
            self.mask_expand_channel_list = torch.zeros(
                # or register these masks as internal parameter
                (len(self.sub_expand_channel_list), self.sub_expand_channel_list[-1], 1, 1, 1),
                requires_grad=False).detach().cuda()
            for i in range(len(self.sub_expand_channel_list)):
                self.mask_expand_channel_list[i][:self.sub_expand_channel_list[i]][:][:][:] = 1
            self.alpha_expand = Parameter(
                torch.Tensor(len(self.sub_expand_channel_list)))  # AP_path_alpha_gumbel_expand
            nn.init.uniform_(self.alpha_expand, -0.25, 0.25)
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv',
                 DynamicPointConv2d(max_in_channels=self.in_channels, max_out_channels=feature_dim, kernel_size=1,
                                    stride=1)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                # SwitchableBatchNorm2d(feature_dim, switchable_channel_list=self.sub_expand_channel_list)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))
            assert len(self.inverted_bottleneck.conv.conv.weight) == len(self.mask_expand_channel_list[0]), \
                'inverted_bottleneck point_conv have the same size(0) as any expand mask size(0)'

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_in_channels=feature_dim, kernel_size=kernel_size, stride=stride)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(feature_dim))
        # set alpha_out_channel
        assert len(out_channels_step) == 3
        assert out_channels_step[1] == out_channels  # out_channels is initialized as max_out_channels
        # int list [16, 28, 4]
        out_channels_step2list = get_list_from_step(low=out_channels_step[0], high=out_channels_step[1],
                                                    step=out_channels_step[2])
        # int list [16, 20, 24, 28]
        self.sub_out_channel_list = out_channels_step2list
        self.mask_out_channel_list = torch.zeros(
            (len(self.sub_out_channel_list), self.sub_out_channel_list[-1], 1, 1, 1),
            requires_grad=False).detach().cuda()
        for i in range(len(self.sub_out_channel_list)):
            self.mask_out_channel_list[i][:self.sub_out_channel_list[i]][:][:][:] = 1

        self.active = False
        self.point_linear = nn.Sequential(OrderedDict([
            ('conv',
             DynamicPointConv2d(max_in_channels=feature_dim, max_out_channels=out_channels, kernel_size=1, stride=1)
             ),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
        assert len(self.point_linear.conv.conv.weight) == len(self.mask_out_channel_list[0]), \
            'MBconv last point_conv have the same size(0) as any out mask size(0)'
        self.alpha_out = None

        self.soft_gumbel_out = None
        self.soft_gumbel_expand = None

    def forward(self, x):
        if self.inverted_bottleneck:
            # 1.compute expand_ratio mask
            # expand_ratio can double choose
            soft_gumbel_expand = self.soft_gumbel_expand.cuda(x.device)
            mask_expand_channel_list = self.mask_expand_channel_list.cuda(x.device)
            expand_mask = sum(w * mask_expand_channel_i for w, mask_expand_channel_i in
                              zip(soft_gumbel_expand, mask_expand_channel_list))
            # 2.set inverted_bottleneck and depth_conv's expand_ratio mask on expand filters
            self.inverted_bottleneck.conv.mask = expand_mask
            self.depth_conv.conv.mask = expand_mask
            if self.use_se:
                self.depth_conv.se.mask = expand_mask

            # 3.forward inverted_bottleneck
            x = self.inverted_bottleneck(x)

        # 1.compute out_channels mask
        if self.stage_head:
            # active stage_head block in this stage
            self.active = True
            assert self.soft_gumbel_out is not None
        else:
            assert self.soft_gumbel_out is not None
        soft_out = self.soft_gumbel_out.cuda(x.device)
        mask_out_channel_list = self.mask_out_channel_list.cuda(x.device)
        out_channel_mask = sum(w * mask_out_channel_i for w, mask_out_channel_i in
                               zip(soft_out, mask_out_channel_list))

        # 2.set out_channel mask on output filters
        self.point_linear.conv.mask = out_channel_mask
        # 3.forward depth_conv and point_linear
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def set_probs(self):
        # print('set_probs')
        if self.inverted_bottleneck:
            self.soft_gumbel_expand = gumbel_softmax(self.alpha_expand, tau=self.tau,
                                                     hard=False,
                                                     dim=0)

    @property
    def module_str(self):
        if self.expand_ratio == 1:
            self.normal_expand_ratio = 1
            probs_chosen_expand = 1.00
        else:
            probs_expand = F.softmax(self.alpha_expand.data, dim=0)
            index = torch.argmax(self.alpha_expand.data).item()
            self.normal_expand_ratio = self.expand_ratio_step2list[index]
            probs_chosen_expand = probs_expand[index]
        if self.stage_head:
            # stage_head can set this stage's out_channel
            probs_out = F.softmax(self.alpha_out.data, dim=0)
            index = torch.argmax(self.alpha_out.data).item()
            self.normal_out_channels = self.sub_out_channel_list[index]
            probs_chosen_out = probs_out[index]
            if self.use_se:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_SE_hswish%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
                else:
                    return '%dx%d_MBConv_SE%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
            else:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_hswish%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
                else:
                    return '%dx%d_MBConv%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
        else:
            if self.use_se:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_SE_hswish%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
                else:
                    return '%dx%d_MBConv_SE%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
            else:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_hswish%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
                else:
                    return '%dx%d_MBConv%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)

    # @property
    def config(self, in_channel=None):
        self.normal_in_channels = in_channel
        if self.normal_in_channels is None:
            self.normal_in_channels = self.in_channels
        if self.stage_head:
            # stage_head can set this stage's out_channel
            index = torch.argmax(self.alpha_out).item()
            self.normal_out_channels = self.sub_out_channel_list[index]
        else:
            # if is stage_tail, out_channel = stage_head_out_channel
            self.normal_out_channels = in_channel
        if self.expand_ratio == 1:
            self.normal_expand_ratio = 1
        else:
            index = torch.argmax(self.alpha_expand).item()
            self.normal_expand_ratio = self.expand_ratio_step2list[index]

        return {
                   'name': MBInvertedConvLayer.__name__,
                   'in_channels': self.normal_in_channels,
                   'out_channels': self.normal_out_channels,
                   'kernel_size': self.kernel_size,
                   'stride': self.stride,
                   'expand_ratio': self.normal_expand_ratio,
                   'mid_channels': self.mid_channels,
                   'use_se': self.use_se,
                   'act_func': self.act_func,
               }, self.normal_out_channels

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def set_tau(self, tau=None, tau_soft=None):
        # use exponential decay for channel
        self.tau = tau_soft

    def get_flops(self, x, in_channel_effective_num=None):
        bottleneck_expand_channel_effective_num = None
        if self.inverted_bottleneck:
            bottleneck_in_channel_effective_num = in_channel_effective_num
            bottleneck_expand_channel_probs = self.soft_gumbel_expand  # F.softmax(self.alpha_expand, dim=0)
            bottleneck_expand_channel_effective_num = sum(
                probs * mask_expand_channel_num_i for probs, mask_expand_channel_num_i in
                zip(bottleneck_expand_channel_probs,
                    self.sub_expand_channel_list))
            flop1 = count_conv_flop(self.inverted_bottleneck.conv.conv, x) * (
                    bottleneck_in_channel_effective_num / self.inverted_bottleneck.conv.max_in_channels) * (
                            bottleneck_expand_channel_effective_num / self.inverted_bottleneck.conv.max_out_channels)
            x = self.inverted_bottleneck(x)
        else:
            # no inverted_bottleneck use full depth_conv
            # depth_conv_in_channel_effective_num = self.depth_conv.conv.max_in_channels
            # give block_0 depth_conv_in_channel_effective_num
            bottleneck_expand_channel_effective_num = self.depth_conv.conv.max_in_channels
            flop1 = 0
        # flops*(c_in_e / c_in)*(c_out_e / c_out)
        if bottleneck_expand_channel_effective_num is None:
            depth_conv_in_channel_effective_num = in_channel_effective_num
        else:
            depth_conv_in_channel_effective_num = bottleneck_expand_channel_effective_num
        if self.use_se:
            x_se = self.depth_conv(x)
            x_se = x_se.mean(3, keepdim=True).mean(2, keepdim=True)
            flop1 += count_conv_flop(self.depth_conv.se.fc.reduce, x_se) * \
                     (depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * (
                             depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * 2
        flop2 = count_conv_flop(self.depth_conv.conv.conv, x) * \
                (depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * (
                        depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels)
        x = self.depth_conv(x)

        # flops*(c_in_e / c_in)*(c_out_e / c_out)
        point_linear_in_channel_effective_num = depth_conv_in_channel_effective_num
        # stage_head should have own out_channels
        if self.stage_head:
            point_linear_out_channel_probs = F.softmax(self.alpha_out, dim=0)
            point_linear_out_channel_effective_num = sum(
                probs * mask_out_channel_num_i for probs, mask_out_channel_num_i in
                zip(point_linear_out_channel_probs,
                    self.sub_out_channel_list))
        else:
            point_linear_out_channel_effective_num = in_channel_effective_num
        flop3 = count_conv_flop(self.point_linear.conv.conv, x) * (
                point_linear_in_channel_effective_num / self.point_linear.conv.max_in_channels) * (
                        point_linear_out_channel_effective_num / self.point_linear.conv.max_out_channels)

        x = self.point_linear(x)
        out_channel_effective_num = point_linear_out_channel_effective_num
        return flop1 + flop2 + flop3, x, out_channel_effective_num

    @staticmethod
    def is_zero_layer():
        return False

    def net2deepernet(self):
        # deepered MBInvertedConvLayer is default to have self.inverted_bottleneck, self.depth_conv and self.point_linear
        self.inverted_bottleneck.conv.conv, self.inverted_bottleneck.bn = identify_inverted_bottleneck(
            inverted_bottleneck=self.inverted_bottleneck,
            noise=True)
        self.depth_conv.conv.conv, self.depth_conv.bn = identify_depth_conv(depth_conv=self.depth_conv, noise=True)
        if self.use_se:
            self.depth_conv.se.fc.reduce = identify_fc(fc=self.depth_conv.se.fc.reduce)
            self.depth_conv.se.fc.expand = identify_fc(fc=self.depth_conv.se.fc.expand)
        self.point_linear.conv.conv, self.point_linear.bn = identify_point_linear(point_linear=self.point_linear,
                                                                                  noise=True)
        return None

    def wider_in_channel(self, wider_in_channel, random_init=False):
        self.inverted_bottleneck = wider_in_channel_inverted_bottleneck(inverted_bottleneck=self.inverted_bottleneck,
                                                                        wider_in_channel=wider_in_channel, noise=True,
                                                                        random_init=random_init)
        feature_dim = round(wider_in_channel * self.expand_ratio)
        # wider_out_channel_point_linear also can wider self.inverted_bottleneck because both they are point_wise_conv
        self.inverted_bottleneck = wider_out_channel_point_linear(point_linear=self.inverted_bottleneck,
                                                                  wider_out_channel=feature_dim,
                                                                  random_init=random_init)
        self.depth_conv = wider_depth_conv(depth_conv=self.depth_conv, wider_channel=feature_dim, noise=True,
                                           random_init=random_init)
        if self.use_se:
            self.depth_conv.se.wider_in_channel(wider_in_channel=feature_dim)
        # wider_in_channel_inverted_bottleneck also can wider self.point_linear because both they are point_wise_conv
        self.point_linear = wider_in_channel_inverted_bottleneck(inverted_bottleneck=self.point_linear,
                                                                 wider_in_channel=feature_dim, noise=True,
                                                                 random_init=random_init)
        return None

    def wider_out_channel(self, wider_out_channel, random_init=False):
        self.point_linear = wider_out_channel_point_linear(point_linear=self.point_linear,
                                                           wider_out_channel=wider_out_channel, noise=True,
                                                           random_init=random_init)
        return None

    def wider_expand_rate(self, wider_expand_rate, random_init=False):
        wider_expand_channel = round(wider_expand_rate*self.inverted_bottleneck.conv.conv.in_channels)
        self.inverted_bottleneck = wider_out_channel_point_linear(point_linear=self.inverted_bottleneck,
                                                                  wider_out_channel=wider_expand_channel,
                                                                  random_init=random_init)
        self.depth_conv = wider_depth_conv(depth_conv=self.depth_conv, wider_channel=wider_expand_channel, noise=True,
                                           random_init=random_init)
        if self.use_se:
            self.depth_conv.se.wider_in_channel(wider_in_channel=wider_expand_channel)
        self.point_linear = wider_in_channel_inverted_bottleneck(inverted_bottleneck=self.point_linear,
                                                                 wider_in_channel=wider_expand_channel, noise=True,
                                                                 random_init=random_init)
        return None

    def get_soft_gumbel_out(self):
        return self.soft_gumbel_out

    def entropy(self, eps=1e-8):
        entropy = 0
        if self.inverted_bottleneck:
            probs = F.softmax(self.alpha_expand.data, dim=0)  # softmax to probability
            log_probs = torch.log(probs + eps)
            entropy += - torch.sum(torch.mul(probs, log_probs))
        if self.stage_head:
            probs = F.softmax(self.alpha_out.data, dim=0)
            log_probs = torch.log(probs + eps)
            entropy += - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def set_soft_gumbel_out(self, soft_gumbel_out):
        self.soft_gumbel_out = soft_gumbel_out
        assert self.soft_gumbel_out is not None


class MBInvertedConvLayer_1x1(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 expand_ratio=6, mid_channels=None, use_se=False, act_func='relu6', expand_ratio_step=None,
                 out_channels_step=None, stage_head=True,
                 stage_tail=False):
        super(MBInvertedConvLayer_1x1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        if expand_ratio == 1:
            # if expand_ratio is 1, don't search expand channel
            self.expand_ratio = expand_ratio
        else:
            self.expand_ratio = expand_ratio_step[1]
        self.mid_channels = mid_channels
        self.use_se = use_se
        self.act_func = act_func
        self.stage_head = stage_head
        self.stage_tail = stage_tail
        self.tau = 1.0

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            stage_head = True
            self.stage_head = True
            self.inverted_bottleneck = None
        else:
            # set alpha_expand_channel
            assert len(expand_ratio_step) == 3, 'expand_ratio_step should have [0.75, 3.25, 0.5]'
            assert self.expand_ratio == expand_ratio_step[1]
            # float list [0.75, 3.25, 0.5]
            self.expand_ratio_step2list = get_list_from_step(low=expand_ratio_step[0], high=expand_ratio_step[1],
                                                             step=expand_ratio_step[2])
            # int list [16, 20, 24, 28]
            self.sub_expand_channel_list = [round(self.in_channels * expand) for expand in self.expand_ratio_step2list]
            self.mask_expand_channel_list = torch.zeros(
                # or register these masks as internal parameter
                (len(self.sub_expand_channel_list), self.sub_expand_channel_list[-1], 1, 1, 1),
                requires_grad=False).detach().cuda()
            for i in range(len(self.sub_expand_channel_list)):
                self.mask_expand_channel_list[i][:self.sub_expand_channel_list[i]][:][:][:] = 1
            self.alpha_expand = Parameter(
                torch.Tensor(len(self.sub_expand_channel_list)))  # AP_path_alpha_gumbel_expand
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv',
                 DynamicPointConv2d(max_in_channels=self.in_channels, max_out_channels=feature_dim, kernel_size=1,
                                    stride=1)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                # SwitchableBatchNorm2d(feature_dim, switchable_channel_list=self.sub_expand_channel_list)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))
            assert len(self.inverted_bottleneck.conv.conv.weight) == len(self.mask_expand_channel_list[0]), \
                'inverted_bottleneck point_conv have the same size(0) as any expand mask size(0)'

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_in_channels=feature_dim, kernel_size=kernel_size, stride=stride)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(feature_dim))
        # set alpha_out_channel
        assert len(out_channels_step) == 3
        assert out_channels_step[1] == out_channels  # out_channels is initialized as max_out_channels
        # int list [16, 28, 4]
        out_channels_step2list = get_list_from_step(low=out_channels_step[0], high=out_channels_step[1],
                                                    step=out_channels_step[2])
        # int list [16, 20, 24, 28]
        self.sub_out_channel_list = out_channels_step2list
        self.mask_out_channel_list = torch.zeros(
            (len(self.sub_out_channel_list), self.sub_out_channel_list[-1], 1, 1, 1),
            requires_grad=False).detach().cuda()
        for i in range(len(self.sub_out_channel_list)):
            self.mask_out_channel_list[i][:self.sub_out_channel_list[i]][:][:][:] = 1

        self.active = False
        self.point_linear = nn.Sequential(OrderedDict([
            ('conv',
             DynamicPointConv2d(max_in_channels=feature_dim, max_out_channels=out_channels, kernel_size=1, stride=1)
             ),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
        assert len(self.point_linear.conv.conv.weight) == len(self.mask_out_channel_list[0]), \
            'MBconv last point_conv have the same size(0) as any out mask size(0)'

        self.alpha_out = Parameter(torch.Tensor(len(self.sub_out_channel_list)))
        nn.init.uniform_(self.alpha_out, -0.25, 0.25)
        self.soft_gumbel_out = None
        self.soft_gumbel_expand = None

    def forward(self, x):
        if self.inverted_bottleneck:
            # 1.compute expand_ratio mask
            # expand_ratio can double choose
            soft_gumbel_expand = self.soft_gumbel_expand.cuda(x.device)
            mask_expand_channel_list = self.mask_expand_channel_list.cuda(x.device)
            expand_mask = sum(w * mask_expand_channel_i for w, mask_expand_channel_i in
                              zip(soft_gumbel_expand, mask_expand_channel_list))
            # 2.set inverted_bottleneck and depth_conv's expand_ratio mask on expand filters
            self.inverted_bottleneck.conv.mask = expand_mask
            self.depth_conv.conv.mask = expand_mask
            if self.use_se:
                self.depth_conv.se.mask = expand_mask

            # 3.forward inverted_bottleneck
            x = self.inverted_bottleneck(x)

        # 1.compute out_channels mask
        if self.stage_head:
            # active stage_head block in this stage
            self.active = True
            assert self.soft_gumbel_out is not None
        else:
            assert self.soft_gumbel_out is not None
        soft_out = self.soft_gumbel_out.cuda(x.device)
        mask_out_channel_list = self.mask_out_channel_list.cuda(x.device)
        out_channel_mask = sum(w * mask_out_channel_i for w, mask_out_channel_i in
                               zip(soft_out, mask_out_channel_list))

        # 2.set out_channel mask on output filters
        self.point_linear.conv.mask = out_channel_mask
        # 3.forward depth_conv and point_linear
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def net2deepernet(self):
        # deepered MBInvertedConvLayer is default to have self.inverted_bottleneck, self.depth_conv and self.point_linear
        self.inverted_bottleneck.conv.conv, self.inverted_bottleneck.bn = identify_inverted_bottleneck(
            inverted_bottleneck=self.inverted_bottleneck,
            noise=True)
        self.depth_conv.conv.conv, self.depth_conv.bn = identify_depth_conv(depth_conv=self.depth_conv, noise=True)
        if self.use_se:
            self.depth_conv.se.fc.reduce = identify_fc(fc=self.depth_conv.se.fc.reduce)
            self.depth_conv.se.fc.expand = identify_fc(fc=self.depth_conv.se.fc.expand)
        self.point_linear.conv.conv, self.point_linear.bn = identify_point_linear(point_linear=self.point_linear,
                                                                                  noise=True)
        return None

    def set_probs(self):
        # print('set_probs')
        if self.inverted_bottleneck:
            self.soft_gumbel_expand = gumbel_softmax(self.alpha_expand, tau=self.tau,
                                                     hard=False,
                                                     dim=0)
        if self.stage_head:
            self.soft_gumbel_out = gumbel_softmax(self.alpha_out, tau=self.tau,
                                                  hard=False, dim=0)

    @property
    def module_str(self):
        if self.expand_ratio == 1:
            self.normal_expand_ratio = 1
            probs_chosen_expand = 1.00
        else:
            probs_expand = F.softmax(self.alpha_expand.data, dim=0)
            index = torch.argmax(self.alpha_expand.data).item()
            self.normal_expand_ratio = self.expand_ratio_step2list[index]
            probs_chosen_expand = probs_expand[index]
        if self.stage_head:
            # stage_head can set this stage's out_channel
            probs_out = F.softmax(self.alpha_out.data, dim=0)
            index = torch.argmax(self.alpha_out.data).item()
            self.normal_out_channels = self.sub_out_channel_list[index]
            probs_chosen_out = probs_out[index]
            if self.use_se:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_SE_hswish%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
                else:
                    return '%dx%d_MBConv_SE%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
            else:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_hswish%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
                else:
                    return '%dx%d_MBConv%.2f(%.2f)_outchannel_%d(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand,
                        self.normal_out_channels, probs_chosen_out)
        else:
            if self.use_se:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_SE_hswish%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
                else:
                    return '%dx%d_MBConv_SE%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
            else:
                if self.act_func == 'h_swish':
                    return '%dx%d_MBConv_hswish%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)
                else:
                    return '%dx%d_MBConv%.2f(%.2f)' % (
                        self.kernel_size, self.kernel_size, self.normal_expand_ratio, probs_chosen_expand)

    # @property
    def config(self, in_channel=None):
        self.normal_in_channels = in_channel
        if self.normal_in_channels is None:
            self.normal_in_channels = self.in_channels
        if self.stage_head:
            # stage_head can set this stage's out_channel
            index = torch.argmax(self.alpha_out).item()
            self.normal_out_channels = self.sub_out_channel_list[index]
        else:
            # if is stage_tail, out_channel = stage_head_out_channel
            self.normal_out_channels = in_channel
        if self.expand_ratio == 1:
            self.normal_expand_ratio = 1
        else:
            index = torch.argmax(self.alpha_expand).item()
            self.normal_expand_ratio = self.expand_ratio_step2list[index]

        return {
                   'name': MBInvertedConvLayer.__name__,
                   'in_channels': self.normal_in_channels,
                   'out_channels': self.normal_out_channels,
                   'kernel_size': self.kernel_size,
                   'stride': self.stride,
                   'expand_ratio': self.normal_expand_ratio,
                   'mid_channels': self.mid_channels,
                   'use_se': self.use_se,
                   'act_func': self.act_func,
               }, self.normal_out_channels

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def set_tau(self, tau=None, tau_soft=None):
        # use exponential decay for channel
        self.tau = tau_soft

    def get_flops(self, x, in_channel_effective_num=None):
        bottleneck_expand_channel_effective_num = None
        if self.inverted_bottleneck:
            bottleneck_in_channel_effective_num = in_channel_effective_num
            bottleneck_expand_channel_probs = self.soft_gumbel_expand  # F.softmax(self.alpha_expand, dim=0)
            bottleneck_expand_channel_effective_num = sum(
                probs * mask_expand_channel_num_i for probs, mask_expand_channel_num_i in
                zip(bottleneck_expand_channel_probs,
                    self.sub_expand_channel_list))
            flop1 = count_conv_flop(self.inverted_bottleneck.conv.conv, x) * (
                    bottleneck_in_channel_effective_num / self.inverted_bottleneck.conv.max_in_channels) * (
                            bottleneck_expand_channel_effective_num / self.inverted_bottleneck.conv.max_out_channels)
            x = self.inverted_bottleneck(x)
        else:
            # no inverted_bottleneck use full depth_conv
            # depth_conv_in_channel_effective_num = self.depth_conv.conv.max_in_channels
            # give block_0 depth_conv_in_channel_effective_num
            bottleneck_expand_channel_effective_num = self.depth_conv.conv.max_in_channels
            flop1 = 0
        # flops*(c_in_e / c_in)*(c_out_e / c_out)
        if bottleneck_expand_channel_effective_num is None:
            depth_conv_in_channel_effective_num = in_channel_effective_num
        else:
            depth_conv_in_channel_effective_num = bottleneck_expand_channel_effective_num
        if self.use_se:
            x_se = self.depth_conv(x)
            x_se = x_se.mean(3, keepdim=True).mean(2, keepdim=True)
            flop1 += count_conv_flop(self.depth_conv.se.fc.reduce, x_se) * \
                     (depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * (
                             depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * 2
        flop2 = count_conv_flop(self.depth_conv.conv.conv, x) * \
                (depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels) * (
                        depth_conv_in_channel_effective_num / self.depth_conv.conv.max_in_channels)
        x = self.depth_conv(x)

        # flops*(c_in_e / c_in)*(c_out_e / c_out)
        point_linear_in_channel_effective_num = depth_conv_in_channel_effective_num
        # stage_head should have own out_channels
        if self.stage_head:
            point_linear_out_channel_probs = F.softmax(self.alpha_out, dim=0)
            point_linear_out_channel_effective_num = sum(
                probs * mask_out_channel_num_i for probs, mask_out_channel_num_i in
                zip(point_linear_out_channel_probs,
                    self.sub_out_channel_list))
        else:
            point_linear_out_channel_effective_num = in_channel_effective_num
        flop3 = count_conv_flop(self.point_linear.conv.conv, x) * (
                point_linear_in_channel_effective_num / self.point_linear.conv.max_in_channels) * (
                        point_linear_out_channel_effective_num / self.point_linear.conv.max_out_channels)

        x = self.point_linear(x)
        out_channel_effective_num = point_linear_out_channel_effective_num
        return flop1 + flop2 + flop3, x, out_channel_effective_num

    @staticmethod
    def is_zero_layer():
        return False

    def get_soft_gumbel_out(self):
        return self.soft_gumbel_out

    def entropy(self, eps=1e-8):
        entropy = 0
        if self.inverted_bottleneck:
            probs = F.softmax(self.alpha_expand.data, dim=0)  # softmax to probability
            log_probs = torch.log(probs + eps)
            entropy += - torch.sum(torch.mul(probs, log_probs))
        if self.stage_head:
            probs = F.softmax(self.alpha_out.data, dim=0)
            log_probs = torch.log(probs + eps)
            entropy += - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def set_soft_gumbel_out(self, soft_gumbel_out):
        self.soft_gumbel_out = soft_gumbel_out


class NormalMBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 expand_ratio=6, mid_channels=None, use_se=False, act_func='relu6',
                 expand_ratio_step=None, out_channels_step=None, stage_head=None,
                 stage_tail=None):
        super(NormalMBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.use_se = use_se
        self.act_func = act_func
        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEModule(feature_dim))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return '%dx%d_MBConv%d_se' % (self.kernel_size, self.kernel_size, self.expand_ratio)
        else:
            return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': NormalMBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'use_se': self.use_se,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return NormalMBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        if self.use_se:
            x_se_in = self.depth_conv(x)
            x_se_in = x_se_in.mean(3, keepdim=True).mean(2, keepdim=True)
            flops_se_1 = count_conv_flop(self.depth_conv.se.fc.reduce, x_se_in)
            x_s = self.depth_conv.se.fc.reduce(x_se_in)
            flops_se_2 = count_conv_flop(self.depth_conv.se.fc.expand, x_s)
            flop2 = flops_se_1 + flops_se_2 + flop2
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = self.channel // self.reduction  # make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(OrderedDict([
            # S
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            # E
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        # dim=[0,1,2,3]
        # feature map in each channel converts to avg value.
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        # * means point to point product
        return x * y


class DynamicSE(SEModule):

    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            # dim=[0,1,2,3]
            # feature map in each channel converts to avg value.
            y = x.mean(3, keepdim=True).mean(2, keepdim=True)
            y = self.fc(y)
            # * means point to point product
            return x * y
        else:
            y = x.mean(3, keepdim=True).mean(2, keepdim=True)
            reduce_conv = self.fc.reduce
            mask_reduce_filter = torch.squeeze(self.mask, 1)
            reduce_filter = reduce_conv.weight * mask_reduce_filter
            reduce_bias = reduce_conv.bias
            # reduce
            y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
            # relu
            y = self.fc.relu(y)
            # expand
            y = self.fc.expand(y)
            # hard sigmoid
            y = self.fc.h_sigmoid(y)
            return x * y

    def wider_in_channel(self, wider_in_channel, random_init=False, noise=True):
        self.fc.reduce = wider_in_channel_conv(conv=self.fc.reduce, wider_in_channel=wider_in_channel, noise=noise,
                                               random_init=random_init)
        num_mid = wider_in_channel // self.reduction
        self.fc.expand = wider_in_channel_conv(conv=self.fc.expand, wider_in_channel=num_mid, noise=noise,
                                               random_init=random_init)

        self.fc.reduce = wider_out_channel_conv(conv=self.fc.reduce, wider_out_channel=num_mid, noise=noise,
                                                random_init=random_init)
        self.fc.expand = wider_out_channel_conv(conv=self.fc.expand, wider_out_channel=wider_in_channel, noise=noise,
                                                random_init=random_init)


class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride
        self.tau = 1.0
        self.soft_gumbel_out = None

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.)

    def net2deepernet(self):
        return None

    def wider_out_channel(self, wider_out_channel, random_init=False):
        return None

    def wider_in_channel(self, wider_in_channel, random_init=False):
        return None

    @property
    def module_str(self):
        return 'Zero'

    # @property
    def config(self, in_channel=None):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x, in_channel_effective_num=None):
        return 0, self.forward(x), in_channel_effective_num

    @staticmethod
    def is_zero_layer():
        return True

    def get_soft_gumbel_out(self):
        return None

    def set_soft_gumbel_out(self, soft_gumbel_out):
        return None

    def set_probs(self):
        return None


if __name__ == '__main__':
    a = torch.randn([4, 3, 1, 1])
    b = torch.zeros([3, 1, 1, 1])
    b[0][0][0] = 1
    b = torch.squeeze(b, 1)
    print(b)
    print(a * b)
