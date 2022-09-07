import torch.nn as nn
import torch.nn.functional as F
from utils import get_same_padding


class SwitchableBatchNorm2d(nn.Module):

    # set switchable BatchNorm2d
    def __init__(self, max_feature_dim, switchable_channel_list=None):
        super(SwitchableBatchNorm2d, self).__init__()
        if switchable_channel_list is None:
            # only have one single channel number
            self.switchable_channel_list = None
            self.max_feature_dim = max_feature_dim
            self.bn = nn.BatchNorm2d(self.max_feature_dim)
        else:
            self.num_features_list = switchable_channel_list
            bns = []
            for i in range(len(switchable_channel_list)):
                bns.append(nn.BatchNorm2d(max_feature_dim))
            self.bn = nn.ModuleList(bns)
        self.active_index = len(switchable_channel_list)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim:
            return bn(x)
        else:
            exponential_average_factor = 0.0
            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )

    def set_active(self, idx=None):
        self.active_index = idx

    def forward(self, x):
        if self.num_features_list is None:
            feature_dim = x.size(1)
            y = self.bn_forward(x, self.bn, feature_dim)
            return y
        else:
            y = self.bn[self.active_index](x)
            return y


class DynamicSeparableConv2d(nn.Module):

    def __init__(self, max_in_channels, kernel_size, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, self.kernel_size, self.stride, padding=self.padding,
            groups=self.max_in_channels, bias=False,
        )
        self.mask = None

    def forward(self, x):
        in_channel = x.size(1)
        if self.mask is None:
            filters = self.conv.weight
        else:
            # assert len(self.conv.weight) == len(self.mask)
            filters = self.conv.weight * self.mask

        y = F.conv2d(
            x, filters, None, self.stride, self.padding, self.dilation, groups=in_channel
        )
        return y


class DynamicPointConv2d(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, padding=self.padding, stride=self.stride,
            bias=False,
        )
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            filters = self.conv.weight
        else:
            filters = self.conv.weight * self.mask

        y = F.conv2d(x, filters, None, self.stride, self.padding, self.dilation, 1)
        return y
