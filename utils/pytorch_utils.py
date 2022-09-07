# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch
import torch as th
import torch.nn as nn
import os
import shutil
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from utils.__init__ import get_same_padding
from collections import OrderedDict


def identify_fc(fc=None, noise=True):
    fc_w_std = fc.weight.data.std()
    fc_b_std = fc.bias.data.std()
    fc.weight.data.zero_()
    fc.bias.data.zero_()
    if noise:
        noise_w = np.random.normal(scale=1e-6 * fc_w_std, size=list(fc.weight.size()))
        fc.weight.data += th.FloatTensor(noise_w).type_as(fc.weight.data)
        noise_b = np.random.normal(scale=1e-6 * fc_b_std, size=list(fc.bias.size()))
        fc.bias.data += th.FloatTensor(noise_b).type_as(fc.bias.data)
    return fc


def identify_depth_conv(depth_conv=None, noise=True):
    assert depth_conv is not None
    m = depth_conv.conv.conv
    # depth_conv.conv is conv_2d
    assert m is not None and m.weight.dim() == 4
    # first use
    m2 = th.nn.Conv2d(in_channels=m.in_channels, out_channels=m.in_channels, kernel_size=m.kernel_size,
                      groups=m.in_channels, padding=m.padding, bias=False)
    m.weight.data.zero_()
    m2.weight.data.zero_()

    # central_location is kernel_size[0] // 2, 5//2 =2, or 3//2=1
    # https://github.com/keras-team/keras/blob/master/examples/mnist_net2net.py
    # https://github.com/soumith/net2net.torch/blob/master/init.lua
    c = m.kernel_size[0] // 2
    restore = False
    # origin code: m.out_channels
    for i in range(0, m2.out_channels):
        assert m2.weight.data.size(1) == 1, 'it should be depth_conv'
        m2.weight.data.narrow(0, i, 1).narrow(1, 0, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)

    if noise:
        noise = np.random.normal(scale=1e-6 * m2.weight.data.std(),
                                 size=list(m2.weight.size()))
        m2.weight.data += th.FloatTensor(noise).type_as(m2.weight.data)

    if restore:
        m2.weight.data = m2.weight.data.view(m2.weight.size(0),
                                             m2.in_channels,
                                             m2.kernel_size[0],
                                             m2.kernel_size[0])

    bnorm = depth_conv.bn
    bnorm.weight.data.fill_(1)
    bnorm.bias.data.fill_(1e-6)
    bnorm.running_mean.fill_(0)
    bnorm.running_var.fill_(1)

    for i in range(0, m2.out_channels):
        # m have n out_channels. m2 have n*expand_ratio out_channels.
        m.weight.data[i] = m2.weight.data[i]
    # m.out_channels/m.in_channels == expand_ratio, keep identity
    # add or not{?}
    # m.weight.data.div_(m.out_channels / m.in_channels)
    # m.weight.data.div_(m.in_channels / m.out_channels)

    return m, bnorm


def identify_point_linear(point_linear=None, noise=True):
    assert point_linear is not None
    m = point_linear.conv.conv
    # point_linear.conv is conv_2d
    assert m is not None and m.weight.dim() == 4
    # first use
    m2 = th.nn.Conv2d(in_channels=m.in_channels, out_channels=m.in_channels, kernel_size=m.kernel_size,
                      padding=m.padding, bias=False)
    m.weight.data.zero_()
    m2.weight.data.zero_()

    # central_location is kernel_size[0] // 2, 5//2 =2, or 3//2=1
    # https://github.com/keras-team/keras/blob/master/examples/mnist_net2net.py
    # https://github.com/soumith/net2net.torch/blob/master/init.lua
    c = m.kernel_size[0] // 2
    restore = False
    # origin code: m.out_channels
    for i in range(0, m2.out_channels):
        if m2.weight.data.size(1) == 1:
            # if m2 is depth-wise convolution, m2.weight.data.size(1) is 1, second dim is each filter's slice number:1.
            # depthwise-conv.narrow(0, i, 1).narrow(1, 0, 1) means i_th kernel's 0_th slice matrix
            # depthwise-conv only has one slice matrix in each kernel
            m2.weight.data.narrow(0, i, 1).narrow(1, 0, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)
        else:
            # if m2 is point_conv convolution:
            # narrow(0, i, 1) means i_th kernel
            # narrow(1, i, 1) means i_th kernel's i_th slice matrix.
            # narrow(2, c, 1).narrow(3, c, 1) means a slice matrix center
            # m2.weight.data.shape={Size: 4} torch.size([10, 10, 5, 5])
            # data = tensor:narrow(dim, index, size)
            # index to index+size-1 element in dim
            m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)

    if noise:
        noise = np.random.normal(scale=1e-6 * m2.weight.data.std(),
                                 size=list(m2.weight.size()))
        m2.weight.data += th.FloatTensor(noise).type_as(m2.weight.data)

    if restore:
        m2.weight.data = m2.weight.data.view(m2.weight.size(0),
                                             m2.in_channels,
                                             m2.kernel_size[0],
                                             m2.kernel_size[0])

    bnorm = point_linear.bn
    bnorm.weight.data.fill_(1)
    bnorm.bias.data.fill_(1e-6)
    bnorm.running_mean.fill_(0)
    bnorm.running_var.fill_(1)

    for i in range(0, m2.out_channels):
        # m have n out_channels. m2 have n*expand_ratio out_channels.
        m.weight.data[i % m.out_channels] += m2.weight.data[i]
    # m.out_channels/m.in_channels == expand_ratio, keep identity

    return m, bnorm


def identify_inverted_bottleneck(inverted_bottleneck=None, noise=True):
    assert inverted_bottleneck is not None
    m = inverted_bottleneck.conv.conv
    # inverted_bottleneck.conv is conv_2d
    assert m is not None and m.weight.dim() == 4
    # first use
    m2 = th.nn.Conv2d(in_channels=m.in_channels, out_channels=m.in_channels, kernel_size=m.kernel_size,
                      padding=m.padding, bias=False)
    m2.weight.data.zero_()

    # central_location is kernel_size[0] // 2, 5//2 =2, or 3//2=1
    # https://github.com/keras-team/keras/blob/master/examples/mnist_net2net.py
    # https://github.com/soumith/net2net.torch/blob/master/init.lua
    c = m.kernel_size[0] // 2
    restore = False
    # origin code: m.out_channels
    for i in range(0, m2.out_channels):
        if m2.weight.data.size(1) == 1:
            # if m2 is depth-wise convolution, m2.weight.data.size(1) is 1, second dim is each filter's slice number:1.
            # depthwise-conv.narrow(0, i, 1).narrow(1, 0, 1) means i_th kernel's 0_th slice matrix
            # depthwise-conv only has one slice matrix in each kernel
            m2.weight.data.narrow(0, i, 1).narrow(1, 0, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)
        else:
            # if m2 is point_conv convolution:
            # narrow(0, i, 1) means i_th kernel
            # narrow(1, i, 1) means i_th kernel's i_th slice matrix.
            # narrow(2, c, 1).narrow(3, c, 1) means a slice matrix center
            # m2.weight.data.shape={Size: 4} torch.size([10, 10, 5, 5])
            # data = tensor:narrow(dim, index, size)
            # index to index+size-1 element in dim
            m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)

    if noise:
        noise = np.random.normal(scale=1e-6 * m2.weight.data.std(),
                                 size=list(m2.weight.size()))
        m2.weight.data += th.FloatTensor(noise).type_as(m2.weight.data)

    if restore:
        m2.weight.data = m2.weight.data.view(m2.weight.size(0),
                                             m2.in_channels,
                                             m2.kernel_size[0],
                                             m2.kernel_size[0])

    bnorm = inverted_bottleneck.bn
    bnorm.weight.data.fill_(1)
    bnorm.bias.data.fill_(1e-6)
    bnorm.running_mean.fill_(0)
    bnorm.running_var.fill_(1)

    for i in range(0, m.out_channels):
        m.weight.data[i] = m2.weight.data[i % m2.out_channels]
    # m.out_channels/m.in_channels == expand_ratio, keep identity
    m.weight.data.div_(m.out_channels / m.in_channels)
    return m, bnorm


def deeper(m, nonlin, bnorm_flag=False, weight_norm=True, noise=True):
    """
    Deeper operator adding a new layer on topf of the given layer.
    Args:
        m (module) - module to add a new layer onto.
        nonlin (module) - non-linearity to be used for the new layer.
        bnorm_flag (bool, False) - whether add a batch normalization btw.
        weight_norm (bool, True) - if True, normalize weights of m before
            adding a new layer.
        noise (bool, True) - if True, add noise to the new layer weights.
    """

    # deeper Convolution

    assert m.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"
    assert m.weight.dim() == 4, 'm should be conv_2d'
    # conv_2d
    # pad_w = pad_h
    pad_h = int((m.kernel_size[0] - 1) / 2)
    # new Conv named m2, also the channel_in and channel_out equal to origin channel.
    m2 = th.nn.Conv2d(in_channels=m.out_channels, out_channels=m.out_channels, kernel_size=m.kernel_size,
                      padding=pad_h)
    # new conv m2 is initialized as all zero.
    m2.weight.data.zero_()
    # central_location is kernel_size[0] // 2, 5//2 =2, or 3//2=1
    # https://github.com/keras-team/keras/blob/master/examples/mnist_net2net.py
    # https://github.com/soumith/net2net.torch/blob/master/init.lua
    c = m.kernel_size[0] // 2
    restore = False

    # div norm on m.weight
    if weight_norm:
        for i in range(m.out_channels):
            weight = m.weight.data
            norm = weight.select(0, i).norm()
            weight.div_(norm)
            m.weight.data = weight

    # initialize deeper module as identity matrix
    # m.out_channels)=10
    # m.out_channels means in_channels and in_channels of m2.weight.data.
    # Also means kernels' number in m2.weight.data.
    assert m.weight.dim() == 4, 'm should be conv_2d'
    for i in range(0, m.out_channels):
        if m2.weight.data.size(1) == 1:
            # if m2 is depth-wise convolution, m2.weight.data.size(1) is 1, second dim is each filter's slice number:1.
            # depthwise-conv.narrow(0, i, 1).narrow(1, 0, 1) means i_th kernel's 0_th slice matrix
            # depthwise-conv only has one slice matrix in each kernel
            m2.weight.data.narrow(0, i, 1).narrow(1, 0, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)
        else:
            # if m2 is normal convolution:
            # narrow(0, i, 1) means i_th kernel
            # narrow(1, i, 1) means i_th kernel's i_th slice matrix.
            # narrow(2, c, 1).narrow(3, c, 1) means a slice matrix center
            # m2.weight.data.shape={Size: 4} torch.size([10, 10, 5, 5])
            # data = tensor:narrow(dim, index, size)
            # index to index+size-1 element in dim
            m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)

    if noise:
        noise = np.random.normal(scale=5e-2 * m2.weight.data.std(),
                                 size=list(m2.weight.size()))
        m2.weight.data += th.FloatTensor(noise).type_as(m2.weight.data)

    if restore:
        m2.weight.data = m2.weight.data.view(m2.weight.size(0),
                                             m2.in_channels,
                                             m2.kernel_size[0],
                                             m2.kernel_size[0])

    m2.bias.data.zero_()

    if bnorm_flag:
        if m.weight.dim() == 4:
            bnorm = th.nn.BatchNorm2d(m2.out_channels)
        bnorm.weight.data.fill_(1)
        bnorm.bias.data.fill_(0)
        bnorm.running_mean.fill_(0)
        bnorm.running_var.fill_(1)

    # s: build a new nn.Sequential() module as ConvBlock
    s = th.nn.Sequential()
    # add old module
    s.add_module('conv', m)
    if bnorm_flag:
        s.add_module('bnorm', bnorm)
    if nonlin is not None:
        s.add_module('nonlin', nonlin())
    s.add_module('conv_new', m2)

    return s


def wider_out_channel_point_linear(point_linear, wider_out_channel, noise=True, weight_norm=True, random_init=False):
    w1 = point_linear.conv.conv.weight.data
    nw1 = point_linear.conv.conv.weight.data.clone()
    old_width = w1.size(0)
    new_width = wider_out_channel
    bnorm = point_linear.bn
    assert nw1.dim() == 4
    assert new_width >= w1.size(0), "New size should be larger than old out_channels"
    if new_width == w1.size(0):
        return point_linear
    nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))

    # expand point_linear.bn and load in old bn parameters
    if bnorm is not None:
        nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
        nrunning_var = bnorm.running_var.clone().resize_(new_width)
        if bnorm.affine:
            nweight = bnorm.weight.data.clone().resize_(new_width)
            nbias = bnorm.bias.data.clone().resize_(new_width)

        nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
        nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
        if bnorm.affine:
            nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
            nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

    # load in old weight parameters
    nw1.narrow(0, 0, old_width).copy_(w1)

    # TEST:normalize weights. norm is 2 norm. After normalization, the norm==1
    if weight_norm:
        for i in range(old_width):
            norm = w1.select(0, i).norm()
            w1.select(0, i).div_(norm)

    # select weights randomly to copy and load in new channel kernel
    tracking = dict()
    for i in range(old_width, new_width):
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)
        if random_init:
            # random padding new kernel
            n = point_linear.conv.conv.kernel_size[0] * point_linear.conv.conv.kernel_size[
                1] * point_linear.conv.conv.out_channels
            nw1.select(0, i).normal_(0, np.sqrt(2. / n))
        else:
            nw1.select(0, i).copy_(w1.select(0, idx).clone())
        if bnorm is not None:
            nrunning_mean[i] = bnorm.running_mean[idx]
            nrunning_var[i] = bnorm.running_var[idx]
            if bnorm.affine:
                nweight[i] = bnorm.weight.data[idx]
                nbias[i] = bnorm.bias.data[idx]
    bnorm.num_features = new_width
    # for idx, d in tracking.items():
    #     # tracking is a dict()
    #     # list d[:]=tracking[idx]
    #     for item in d:
    #         # d[:] is the list of old width and new width idx/i.
    #         # len(d) is the number of channel choosed times.
    #         # corresponding to paper, only second layer needs div_(choosed_times) when widering
    #         nw1[item].div_(len(d))
    #         if bnorm is not None:
    #             nrunning_mean[item].div_(len(d))
    #             nrunning_var[item].div_(len(d))
    #             if bnorm.affine:
    #                 nweight[item].div_(len(d))
    #                 nbias[item].div_(len(d))
    point_linear.conv.conv.out_channels = new_width
    if noise:
        noise = np.random.normal(scale=1e-3 * nw1.std(),
                                 size=list(nw1.data.size()))
        # print('list(nw1.data.size())', list(nw1.data.size()))
        nw1 += th.FloatTensor(noise).type_as(nw1)
    point_linear.conv.conv.weight.data = nw1
    if bnorm is not None:
        bnorm.running_var = nrunning_var
        bnorm.running_mean = nrunning_mean
        if bnorm.affine:
            bnorm.weight.data = nweight
            bnorm.bias.data = nbias
        point_linear.bn = bnorm
    return point_linear


def wider_depth_conv(depth_conv, wider_channel, noise=True, weight_norm=True, random_init=False):
    w1 = depth_conv.conv.conv.weight.data
    nw1 = depth_conv.conv.conv.weight.data.clone()
    old_width = w1.size(0)
    new_width = wider_channel
    bnorm = depth_conv.bn
    assert nw1.dim() == 4
    assert new_width >= w1.size(0), "New size should be larger than old out_channels"
    if new_width == w1.size(0):
        return depth_conv
    nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
    # reset groups of depth_conv
    if bnorm is not None:
        nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
        nrunning_var = bnorm.running_var.clone().resize_(new_width)
        if bnorm.affine:
            nweight = bnorm.weight.data.clone().resize_(new_width)
            nbias = bnorm.bias.data.clone().resize_(new_width)

        nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
        nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
        if bnorm.affine:
            nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
            nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

    # load in old weight parameters
    nw1.narrow(0, 0, old_width).copy_(w1)

    # Think about depth_conv weight_norm. TEST:normalize weights. norm is 2 norm. After normalization, the norm==1
    if weight_norm:
        for i in range(old_width):
            norm = w1.select(0, i).norm()
            w1.select(0, i).div_(norm)

    # select weights randomly to copy and load in new channel kernel
    tracking = dict()
    for i in range(old_width, new_width):
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)
        if random_init:
            # random padding new kernel
            n = depth_conv.conv.conv.kernel_size[0] * depth_conv.conv.conv.kernel_size[
                1] * depth_conv.conv.conv.out_channels
            nw1.select(0, i).normal_(0, np.sqrt(2. / n))
        else:
            nw1.select(0, i).copy_(w1.select(0, idx).clone())
        if bnorm is not None:
            nrunning_mean[i] = bnorm.running_mean[idx]
            nrunning_var[i] = bnorm.running_var[idx]
            if bnorm.affine:
                nweight[i] = bnorm.weight.data[idx]
                nbias[i] = bnorm.bias.data[idx]
            bnorm.num_features = new_width

    # initialize as divide weight
    # for idx, d in tracking.items():
    #     # tracking is a dict()
    #     # list d[:]=tracking[idx]
    #     for item in d:
    #         # d[:] is the list of old width and new width idx/i.
    #         # len(d) is the number of channel choosed times.
    #         # corresponding to paper, only second layer needs div_(choosed_times) when widering
    #         nw1[item].div_(len(d))
    #         if bnorm is not None:
    #             nrunning_mean[item].div_(len(d))
    #             nrunning_var[item].div_(len(d))
    #             if bnorm.affine:
    #                 nweight[item].div_(len(d))
    #                 nbias[item].div_(len(d))

    depth_conv.conv.conv.out_channels = new_width
    depth_conv.conv.conv.groups = new_width

    if noise:
        noise = np.random.normal(scale=1e-3 * nw1.std(),
                                 size=list(nw1.data.size()))
        # print('list(nw1.data.size())', list(nw1.data.size()))
        nw1 += th.FloatTensor(noise).type_as(nw1)
    depth_conv.conv.conv.weight.data = nw1
    if bnorm is not None:
        bnorm.running_var = nrunning_var
        bnorm.running_mean = nrunning_mean
        if bnorm.affine:
            bnorm.weight.data = nweight
            bnorm.bias.data = nbias
        depth_conv.bn = bnorm
    return depth_conv


def wider_in_channel_inverted_bottleneck(inverted_bottleneck, wider_in_channel, noise=True, random_init=False):
    w2 = inverted_bottleneck.conv.conv.weight.data
    nw2 = w2.clone()
    new_width = wider_in_channel
    old_width = w2.size(1)
    if new_width == old_width:
        return inverted_bottleneck
    if nw2.size(1) == 1:
        # nw2 is depthwise convolution and it kernel_number == new_width.
        nw2.resize_(new_width, 1, nw2.size(2), nw2.size(3))
    else:
        # nw2 is normal convolution
        nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))

    # w1 and nw1 do not use transpose(0,1) denotes: w1 and nw1 is kernel_numbers_level increase kernel_numbers instead kernel_size.
    # w2 and nw2 use transpose(0,1) denotes: w2 -> nw2 is kernel_first_dim_level expand kernel_size instead kernel number.
    w2 = w2.transpose(0, 1)
    nw2 = nw2.transpose(0, 1)

    # inherit old_net stat_dict into old_channel. it is along the w2.transpose(0, 1) dim=0 --> kernel's first dim.
    # old_width is old kernel number, new_width is new kernel number.
    if nw2.size(0) == 1:
        nw2.narrow(0, 0, 1).copy_(w2)
    else:
        nw2.narrow(0, 0, old_width).copy_(w2)

    # select weights randomly
    tracking = dict()
    for i in range(old_width, new_width):
        # random choose idx from old_width
        # i is in new_width
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)

        # TEST:random init for new units
        if random_init:
            # random padding new kernel
            n2 = inverted_bottleneck.conv.conv.kernel_size[0] * inverted_bottleneck.conv.conv.kernel_size[
                1] * inverted_bottleneck.conv.conv.out_channels
            nw2.select(0, i).normal_(0, np.sqrt(2. / n2))
        else:
            # use net2net_wider
            # select i(as index) in dim=0
            # dim=0, i is from new_width, idx is from old_width.
            nw2.select(0, i).copy_(w2.select(0, idx).clone())

    # initialize as divide weight
    if not random_init:
        for idx, d in tracking.items():
            # tracking is a dict()
            # list d[:]=tracking[idx]
            for item in d:
                # d[:] is the list of old width and new width idx/i.
                # len(d) is the number of channel choosed times.
                # corresponding to paper, only second layer needs div_(choosed_times) when widering
                nw2[item].div_(len(d))

    if noise:
        noise = np.random.normal(scale=2e-4 * nw2.std(),
                                 size=list(nw2.data.size()))
        nw2 += th.FloatTensor(noise).type_as(nw2)
    # swap by dim0=0, dim1=1
    # https://discuss.pytorch.org/t/swap-axes-in-pytorch/970
    w2.transpose_(0, 1)
    nw2.transpose_(0, 1)

    inverted_bottleneck.conv.conv.in_channels = new_width
    inverted_bottleneck.conv.conv.weight.data = nw2
    return inverted_bottleneck


def wider_in_channel_conv(conv, wider_in_channel, noise=True, random_init=False):
    w2 = conv.weight.data
    nw2 = w2.clone()
    new_width = wider_in_channel
    old_width = w2.size(1)
    if new_width == old_width:
        return conv
    if nw2.size(1) == 1:
        # nw2 is depthwise convolution and it kernel_number == new_width.
        nw2.resize_(new_width, 1, nw2.size(2), nw2.size(3))
    else:
        # nw2 is normal convolution
        nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))

    # w1 and nw1 do not use transpose(0,1) denotes: w1 and nw1 is kernel_numbers_level increase kernel_numbers instead kernel_size.
    # w2 and nw2 use transpose(0,1) denotes: w2 -> nw2 is kernel_first_dim_level expand kernel_size instead kernel number.
    w2 = w2.transpose(0, 1)
    nw2 = nw2.transpose(0, 1)

    # inherit old_net stat_dict into old_channel. it is along the w2.transpose(0, 1) dim=0 --> kernel's first dim.
    # old_width is old kernel number, new_width is new kernel number.
    if nw2.size(0) == 1:
        nw2.narrow(0, 0, 1).copy_(w2)
    else:
        nw2.narrow(0, 0, old_width).copy_(w2)

    # select weights randomly
    tracking = dict()
    for i in range(old_width, new_width):
        # random choose idx from old_width
        # i is in new_width
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)

        # TEST:random init for new units
        if random_init:
            # random padding new kernel
            n2 = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
            nw2.select(0, i).normal_(0, np.sqrt(2. / n2))
        else:
            # use net2net_wider
            # select i(as index) in dim=0
            # dim=0, i is from new_width, idx is from old_width.
            nw2.select(0, i).copy_(w2.select(0, idx).clone())

    # initialize as divide weight
    if not random_init:
        for idx, d in tracking.items():
            # tracking is a dict()
            # list d[:]=tracking[idx]
            for item in d:
                # d[:] is the list of old width and new width idx/i.
                # len(d) is the number of channel choosed times.
                # corresponding to paper, only second layer needs div_(choosed_times) when widering
                nw2[item].div_(len(d))

    if noise:
        noise = np.random.normal(scale=2e-3 * nw2.std(),
                                 size=list(nw2.data.size()))
        nw2 += th.FloatTensor(noise).type_as(nw2)
    # swap by dim0=0, dim1=1
    # https://discuss.pytorch.org/t/swap-axes-in-pytorch/970
    w2.transpose_(0, 1)
    nw2.transpose_(0, 1)

    conv.in_channels = new_width
    conv.weight.data = nw2
    return conv


def wider_out_channel_conv(conv, wider_out_channel, noise=True, random_init=False, weight_norm=True):
    w1 = conv.weight.data
    b1 = conv.bias.data
    nw1 = conv.weight.data.clone()
    old_width = w1.size(0)
    new_width = wider_out_channel
    assert nw1.dim() == 4
    assert new_width >= w1.size(0), "New size should be larger than old out_channels"
    if new_width == w1.size(0):
        return conv
    nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))

    # expand point_linear.bn and load in old bn parameters

    # load in old weight parameters
    nw1.narrow(0, 0, old_width).copy_(w1)

    # TEST:normalize weights. norm is 2 norm. After normalization, the norm==1
    if weight_norm:
        for i in range(old_width):
            norm = w1.select(0, i).norm()
            w1.select(0, i).div_(norm)
    if b1 is not None:
        nb1 = conv.bias.data.clone()
        nb1.resize_(new_width)
    nb1.narrow(0, 0, old_width).copy_(b1)

    # select weights randomly to copy and load in new channel kernel
    tracking = dict()
    for i in range(old_width, new_width):
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)
        if random_init:
            # random padding new kernel
            n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
            nw1.select(0, i).normal_(0, np.sqrt(2. / n))
        else:
            nw1.select(0, i).copy_(w1.select(0, idx).clone())
        nb1[i] = b1[idx]

    conv.out_channels = new_width
    if noise:
        noise = np.random.normal(scale=1e-3 * nw1.std(),
                                 size=list(nw1.data.size()))
        # print('list(nw1.data.size())', list(nw1.data.size()))
        nw1 += th.FloatTensor(noise).type_as(nw1)
    conv.weight.data = nw1
    conv.bias.data = nb1
    return conv


def wider(m1, m2, new_width, bnorm=None, out_size=None, noise=True,
          random_init=True, weight_norm=True):
    """
    Convert m1 layer to its wider version by adapthing next weight layer and
    possible batch norm layer in btw.
    Args:
        m1 - module to be wider
        m2 - follwing module to be adapted to m1
        new_width - new width for m1.
        bn (optional) - batch norm layer, if there is btw m1 and m2
        out_size (list, optional) - necessary for m1 == conv3d and m2 == linear. It
            is 3rd dim size of the output feature map of m1. Used to compute
            the matching Linear layer size
        noise (bool, True) - add a slight noise to break symmetry btw weights.
        random_init (optional, True) - if True, new weights are initialized
            randomly.
        weight_norm (optional, True) - If True, weights are normalized before
            transfering.
    """

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1) // factor ** 2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None, \
                    "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1) // factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            # w2.size(0) denotes kernel number/in_channels.
            # w2.size(1) denotes channel number in each group
            assert w1.size(0) == w2.size(1) or w1.size(0) == w2.size(0) * w2.size(
                1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger than old out_channels"

        old_width = w1.size(0)
        nw1 = m1.weight.data.clone()
        # nw2 = w2 = m2.weight.data with shape[10,1,5,5]
        nw2 = w2.clone()

        if nw1.dim() == 4:
            # deal with 2D kernel
            # out_channels, in_channels // groups, *kernel_size: [10,1,5,5]
            if nw1.size(1) == 1:
                # nw1 is depthwise convolution and it kernel_number == new_width.
                nw1.resize_(new_width, 1, nw1.size(2), nw1.size(3))
            else:
                # nw1 is normal convolution
                nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
            if nw2.size(1) == 1:
                # nw2 is depthwise convolution and it kernel_number == new_width.
                nw2.resize_(new_width, 1, nw2.size(2), nw2.size(3))
            else:
                # nw2 is normal convolution
                nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        else:
            nw1.resize_(new_width, nw1.size(1))
            nw2.resize_(nw2.size(0), new_width)

        # depthwise.bias registers as None.
        if b1 is not None:
            nb1 = m1.bias.data.clone()
            nb1.resize_(new_width)

        # for bn layer
        if bnorm is not None:
            nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
            nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if bnorm.affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

        # w1 and nw1 do not use transpose(0,1) denotes: w1 and nw1 is kernel_numbers_level increase kernel_numbers instead kernel_size.
        # w2 and nw2 use transpose(0,1) denotes: w2 -> nw2 is kernel_first_dim_level expand kernel_size instead kernel number.
        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        # inherit old_net stat_dict into old_channel. it is along the w2.transpose(0, 1) dim=0 --> kernel's first dim.
        # old_width is old kernel number, new_width is new kernel number.
        nw1.narrow(0, 0, old_width).copy_(w1)
        if nw2.size(0) == 1:
            nw2.narrow(0, 0, 1).copy_(w2)
        else:
            nw2.narrow(0, 0, old_width).copy_(w2)
        nb1.narrow(0, 0, old_width).copy_(b1)

        # BN layer is optional but when there is btw m1 and m2, it must be Ture
        if bnorm is not None:
            nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if bnorm.affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        # TEST:normalize weights
        if weight_norm:
            for i in range(old_width):
                norm = w1.select(0, i).norm()
                w1.select(0, i).div_(norm)

        # select weights randomly
        tracking = dict()
        for i in range(old_width, new_width):
            # random choose idx from old_width
            # i is in new_width
            idx = np.random.randint(0, old_width)
            try:
                tracking[idx].append(i)
            except:
                tracking[idx] = [idx]
                tracking[idx].append(i)

            # TEST:random init for new units
            if random_init:
                # random padding new kernel
                n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                if m2.weight.dim() == 4:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                elif m2.weight.dim() == 5:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                elif m2.weight.dim() == 2:
                    n2 = m2.out_features * m2.in_features
                nw1.select(0, i).normal_(0, np.sqrt(2. / n))
                nw2.select(0, i).normal_(0, np.sqrt(2. / n2))
            else:
                # use net2net_wider
                # select i(as index) in dim=0
                # dim=0, i is from new_width, idx is from old_width.
                nw1.select(0, i).copy_(w1.select(0, idx).clone())
                nw2.select(0, i).copy_(w2.select(0, idx).clone())
            nb1[i] = b1[idx]

            if bnorm is not None:
                # BN layer is optional but when there is btw m1 and m2, it must be Ture
                nrunning_mean[i] = bnorm.running_mean[idx]
                nrunning_var[i] = bnorm.running_var[idx]
                if bnorm.affine:
                    nweight[i] = bnorm.weight.data[idx]
                    nbias[i] = bnorm.bias.data[idx]
        bnorm.num_features = new_width

        # initialize as divide weight
        if not random_init:
            for idx, d in tracking.items():
                # tracking is a dict()
                # list d[:]=tracking[idx]
                for item in d:
                    # d[:] is the list of old width and new width idx/i.
                    # len(d) is the number of channel choosed times.
                    # corresponding to paper, only second layer needs div_(choosed_times) when widering
                    nw2[item].div_(len(d))

        # swap by dim0=0, dim1=1
        # https://discuss.pytorch.org/t/swap-axes-in-pytorch/970
        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width

        if noise:
            noise = np.random.normal(scale=5e-2 * nw1.std(),
                                     size=list(nw1.data.size()))
            print('list(nw1.data.size())', list(nw1.data.size()))
            nw1 += th.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            # when m2='Linear',
            if w1.dim() == 4:
                # factor is important to work for Conv2d-Linear expand channel.
                # m2 is Linear. And m2_Linear should expand weight and in_features channel.
                # nw2.view(m2.weight.size(0), new_width * factor ** 2) is a matrix
                # with shape [m2.weight.size(0), new_width * factor ** 2]
                m2.weight.data = nw2.view(m2.weight.size(0), new_width * factor ** 2)
                m2.in_features = new_width * factor ** 2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width * factor)
                m2.in_features = new_width * factor
        else:
            # when m2='Conv'
            m2.weight.data = nw2

        m1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias
        return m1, m2, bnorm


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    # type: # (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        # >>> logits = torch.randn(20, 32)
        # >>> # Sample soft categorical using reparametrization trick:
        # >>> F.gumbel_softmax(logits, tau=1, hard=False)
        # >>> # Sample hard categorical using "Straight-through" trick:
        # >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).cuda().exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        # Reparametrization trick.
        ret = y_soft
        return ret


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data_1, self.next_data_2 = next(self.loader)
        except StopIteration:
            self.next_data_1 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data_1 = self.next_data_1.cuda(non_blocking=True)
            self.next_data_2 = self.next_data_2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_1, data_2 = self.next_data_1, self.next_data_2
        self.preload()
        return data_1, data_2


class dual_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data_1, self.next_data_2, self.next_data_3, self.next_data_4 = next(self.loader)
        except StopIteration:
            self.next_data_1 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data_1 = self.next_data_1.cuda(non_blocking=True)
            self.next_data_2 = self.next_data_2.cuda(non_blocking=True)
            self.next_data_3 = self.next_data_3.cuda(non_blocking=True)
            self.next_data_4 = self.next_data_4.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_1, data_2, data_3, data_4 = self.next_data_1, self.next_data_2, self.next_data_3, self.next_data_4
        self.preload()
        return data_1, data_2, data_3, data_4


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


def build_activation(act_func='relu6', inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


# noinspection PyUnresolvedReferences
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class console_log(object):
    def __init__(self, logs_path='./'):
        self.logs_path = logs_path
        if not os.path.exists(self.logs_path):
            os.mkdir(self.logs_path)

    def write_log(self, log_str=None, should_print=True, prefix='console_log', end='\n'):
        with open(os.path.join(self.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)


def check_dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_out_channel_settings(period='b0'):
    if period == 'b0':
        blocks_out_channels_step = [
            # stride 2, blocks 3.
            [16, 28, 4],
            [16, 28, 4],
            [16, 28, 4],
            # stride 2, blocks 3.
            [16, 40, 8],
            [16, 40, 8],
            [16, 40, 8],
            # stride 2, blocks 5.
            [48, 96, 8],
            [48, 96, 8],
            [48, 96, 8],
            [48, 96, 8],
            [48, 96, 8],
            # stride 1, blocks 5.
            [72, 128, 8],
            [72, 128, 8],
            [72, 128, 8],
            [72, 128, 8],
            [72, 128, 8],
            # stride 2, blocks 6.
            [112, 184, 8],
            [112, 184, 8],
            [112, 184, 8],
            [112, 184, 8],
            [112, 184, 8],
            [112, 184, 8],
            # stride 1, blocks 3.
            [112, 184, 8],
            [112, 184, 8],
            [112, 184, 8],
        ]
    elif period == 'b1':
        blocks_out_channels_step = [
            # stride 2, blocks 3.
            [16, 32, 4],
            [16, 32, 4],
            [16, 32, 4],
            # stride 2, blocks 3.
            [16, 48, 8],
            [16, 48, 8],
            [16, 48, 8],
            # stride 2, blocks 5.
            [48, 104, 8],
            [48, 104, 8],
            [48, 104, 8],
            [48, 104, 8],
            [48, 104, 8],
            # stride 1, blocks 5.
            [72, 136, 8],
            [72, 136, 8],
            [72, 136, 8],
            [72, 136, 8],
            [72, 136, 8],
            # stride 2, blocks 6.
            [112, 224, 8],
            [112, 224, 8],
            [112, 224, 8],
            [112, 224, 8],
            [112, 224, 8],
            [112, 224, 8],
            # stride 1, blocks 3.
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
        ]
    elif period == 'b2':
        blocks_out_channels_step = [
            # stride 2, blocks 3.
            [16, 36, 4],
            [16, 36, 4],
            [16, 36, 4],
            # stride 2, blocks 3.
            [16, 56, 8],
            [16, 56, 8],
            [16, 56, 8],
            # stride 2, blocks 5.
            [48, 112, 8],
            [48, 112, 8],
            [48, 112, 8],
            [48, 112, 8],
            [48, 112, 8],
            # stride 1, blocks 5.
            [72, 144, 8],
            [72, 144, 8],
            [72, 144, 8],
            [72, 144, 8],
            [72, 144, 8],
            # stride 2, blocks 6.
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            # stride 1, blocks 3.
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
        ]
    elif period == 'b3':
        blocks_out_channels_step = [
            # stride 2, blocks 3.
            [16, 36, 4],
            [16, 36, 4],
            [16, 36, 4],
            # stride 2, blocks 3.
            [16, 64, 8],
            [16, 64, 8],
            [16, 64, 8],
            # stride 2, blocks 5.
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            # stride 1, blocks 5.
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            # stride 2, blocks 6.
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            # stride 1, blocks 3.
            [112, 336, 8],
            [112, 336, 8],
            [112, 336, 8],
        ]
    elif period == 'b4':
        blocks_out_channels_step = [
            # stride 2, blocks 3.
            [16, 40, 4],
            [16, 40, 4],
            [16, 40, 4],
            # stride 2, blocks 3.
            [16, 64, 8],
            [16, 64, 8],
            [16, 64, 8],
            # stride 2, blocks 5.
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            [48, 128, 8],
            # stride 1, blocks 5.
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            [72, 160, 8],
            # stride 2, blocks 6.
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            [112, 256, 8],
            # stride 1, blocks 3.
            [112, 336, 8],
            [112, 336, 8],
            [112, 336, 8],
        ]

    else:
        raise NotImplementedError
    return blocks_out_channels_step


def get_expand_rate_settings(period='b0'):
    if period == 'b0' or period == 'b1':
        blocks_expand_ratio_step = [
            # stride 2, blocks 3.
            [0.75, 3.75, 0.5],
            [0.75, 3.75, 0.5],
            [0.75, 3.75, 0.5],
            # stride 2, blocks 3.
            [0.75, 3.75, 0.5],
            [0.75, 3.75, 0.5],
            [0.75, 3.75, 0.5],
            # stride 2, blocks 5.
            [0.75, 4.25, 0.5],
            [0.75, 4.25, 0.5],
            [0.75, 4.25, 0.5],
            [0.75, 4.25, 0.5],
            [0.75, 4.25, 0.5],
            # stride 1, blocks 5.
            [0.75, 4.5, 0.75],
            [0.75, 4.5, 0.75],
            [0.75, 4.5, 0.75],
            [0.75, 4.5, 0.75],
            [0.75, 4.5, 0.75],
            # stride 2, blocks 6.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            # stride 1, blocks 3.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
        ]
    elif period == 'b2':
        blocks_expand_ratio_step = [
            # stride 2, blocks 3.
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            # stride 2, blocks 3.
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            # stride 2, blocks 5.
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            # stride 1, blocks 5.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            # stride 2, blocks 6.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            # stride 1, blocks 3.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
        ]
    elif period == 'b3':
        blocks_expand_ratio_step = [
            # stride 2, blocks 3.
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            [0.75, 4.75, 0.5],
            # stride 2, blocks 3.
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            # stride 2, blocks 5.
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            [0.75, 5.25, 0.5],
            # stride 1, blocks 5.
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            [0.75, 5.25, 0.75],
            # stride 2, blocks 6.
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            # stride 1, blocks 3.
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
        ]
    else:
        blocks_expand_ratio_step = [
            # stride 2, blocks 3.
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            # stride 2, blocks 3.
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            # stride 2, blocks 5.
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            [0.75, 5.75, 0.5],
            # stride 1, blocks 5.
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            # stride 2, blocks 6.
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            # stride 1, blocks 3.
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
            [0.75, 6, 0.75],
        ]
    return blocks_expand_ratio_step


if __name__ == '__main__':
    # test wider
    # from modules.layers import *

    a = get_out_channel_settings()
    print(a[3][1])
    # input2 = torch.cat((input,input,input), dim=1, out=None)
    # out_channels = 3
    # feature_dim = 9
    # wider_out_channel = 9
    # point_linear = nn.Sequential(OrderedDict([
    #     ('conv',
    #      DynamicPointConv2d(max_in_channels=feature_dim, max_out_channels=out_channels, kernel_size=1, stride=1)
    #      ),
    #     ('bn', nn.BatchNorm2d(out_channels)),
    # ]))
    # print('before:', point_linear)
    # point_linear.eval()
    # output1 = point_linear(input2)
    # point_linear = wider_out_channel_point_linear(point_linear, wider_out_channel, noise=False, weight_norm=True, random_init=False)
    # print('after:', point_linear)
    # point_linear.eval()
    # output2 = point_linear(input2)
    # print(th.abs((output1.sum() - output2.sum()).sum()).item())
    # 68.74
    # =====================================
    # input = torch.rand(32, 40, 32, 32, out=None)
    # input2 = torch.cat((input, input, input, input), dim=1, out=None)
    # print('input: ', input.size(), 'input2: ', input2.size())
    # in_channels = 40
    # feature_dim = 160
    # se = DynamicSE(in_channels)
    # print('before:', se)
    # output1 = se(input)
    # se.wider_in_channel(wider_in_channel=feature_dim, noise=True, random_init=False)
    # print('after:', se)
    # se.eval()
    # output2 = se(input2)
    # print(th.abs((output1.sum() - output2.sum()).sum()).item())
    # wider:983695.25
    # random_init:1057184.00
    # =====================================
    # input = torch.rand(32, 3, 32, 32, out=None)
    # input2 = torch.cat((input,input,input), dim=1, out=None)
    # print('input: ',input.size(), 'input2: ',input2.size())
    # in_channels = 3
    # feature_dim = 9
    # inverted_bottleneck = nn.Sequential(OrderedDict([
    #     ('conv',
    #      DynamicPointConv2d(max_in_channels=in_channels, max_out_channels=feature_dim, kernel_size=1,
    #                         stride=1)),
    #     ('bn', nn.BatchNorm2d(feature_dim)),
    #     # SwitchableBatchNorm2d(feature_dim, switchable_channel_list=self.sub_expand_channel_list)),
    #     ('act', build_activation('relu6', inplace=True)),
    # ]))
    # print('before:', inverted_bottleneck)
    # inverted_bottleneck.eval()
    # output1 = inverted_bottleneck(input)
    # inverted_bottleneck = wider_in_channel_inverted_bottleneck(inverted_bottleneck, wider_in_channel=9, noise=True, random_init=False)
    # print('after:', inverted_bottleneck)
    # inverted_bottleneck.eval()
    # output2 = inverted_bottleneck(input2)
    # print(th.abs((output1.sum() - output2.sum()).sum()).item())
    # random_init:15544.58
    # net2widernet:390.58
    # =====================================
    # input = torch.rand(5, out=None).cuda()
    # output, index = gumbel_softmax(input, tau=1,hard=True,dim=0)
    # print(input)
    # print(output)
    # print(index.data.item())
    # test inverted_bottleneck
    # input = torch.rand(32, 3, 16, 16, out=None)
    # inverted_bottleneck = nn.Sequential(OrderedDict([
    #     ('conv',
    #      nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0,
    #                bias=False)),
    #     ('bn', nn.BatchNorm2d(9)),
    #     ('act', nn.ReLU6(inplace=True)),
    # ]))
    # print('before:', inverted_bottleneck)
    # inverted_bottleneck.eval()
    # output1 = inverted_bottleneck(input)
    # inverted_bottleneck.conv, inverted_bottleneck.bn = \
    #     identify_inverted_bottleneck(inverted_bottleneck=inverted_bottleneck, noise=True)
    # print('after:', inverted_bottleneck)
    # inverted_bottleneck.eval()
    # output2 = inverted_bottleneck(input)
    # # 3390.0048828125
    # print(th.abs((input.sum() - output1.sum()).sum()).item())
    # # 0.1271484375
    # print(th.abs((input.sum() - output2.sum()).sum()).item())
    #
    # # test point_linear
    # input = torch.rand(32, 9, 16, 16, out=None)
    # point_linear = nn.Sequential(OrderedDict([
    #     ('conv', nn.Conv2d(9, 3, 1, 1, 0, bias=False)),
    #     ('bn', nn.BatchNorm2d(3)),
    # ]))
    # print('before:', point_linear)
    # point_linear.eval()
    # output1 = point_linear(input)
    # point_linear.conv, point_linear.bn = identify_point_linear(point_linear=point_linear, noise=False)
    # print('after:', point_linear)
    # point_linear.eval()
    # output2 = point_linear(input)
    # # 38745.33203125
    # print(th.abs((input.sum() - output1.sum()).sum()).item())
    # # 0.1484375
    # print(th.abs((input.sum() - output2.sum()).sum()).item())
    #
    # # test depth_conv
    # input = torch.rand(32, 9, 16, 16, out=None)
    # kernel_size = 3
    # pad = get_same_padding(kernel_size)
    # depth_conv = nn.Sequential(OrderedDict([
    #     ('conv', nn.Conv2d(9, 9, kernel_size, 1, pad, groups=9, bias=False)),
    #     ('bn', nn.BatchNorm2d(9)),
    #     ('act', nn.ReLU6(inplace=True)),
    # ]))
    # print('before:', depth_conv)
    # depth_conv.eval()
    # output1 = depth_conv(input)
    # depth_conv.conv, depth_conv.bn = identify_depth_conv(depth_conv=depth_conv, noise=True)
    # print('after:', depth_conv)
    # depth_conv.eval()
    # output2 = depth_conv(input)
    # # 38745.33203125
    # print(th.abs((input.sum() - output1.sum()).sum()).item())
    # # 0.1484375
    # print(th.abs((input.sum() - output2.sum()).sum()).item())
    #
    # # test fc
    # input = torch.rand(32, 9, 16, 16, out=None)
    # fc = nn.Conv2d(9, 9, 1, 1, 0, bias=True)
    # print('before:', fc)
    # fc.eval()
    # output1 = fc(input)
    # fc = identify_fc(fc=fc, noise=True)
    # print('after:', fc)
    # fc.eval()
    # output2 = fc(input)
    # # 38745.33203125
    # print(th.abs((input.sum() - output1.sum()).sum()).item())
    # # 0.1484375
    # print(th.abs((input.sum() - output2.sum()).sum()).item())
    # print(th.abs(output1.sum()).item())
    # print(th.abs(output2.sum()).item())
