# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import numpy as np
import warnings
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import copy
from modules.layers import *
from modules.layers import MBInvertedConvLayer
from utils.pytorch_utils import gumbel_softmax


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order, expand_ratio_step,
                        out_channels_step, stage_head=True, stage_tail=True):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Identity': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head, stage_tail,
        : IdentityLayer(
            in_C, out_C, ops_order=ops_order),
        'Zero': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head, stage_tail,
        : ZeroLayer(
            stride=S),
    }
    # add MBConv layers

    name2ops.update({
        # in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, expand_ratio_step=None, out_channels_step=None
        '3x3_MBConv1': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                              stage_tail: MBInvertedConvLayer_1x1(in_C, out_C, 3, S,
                                                                  1,
                                                                  expand_ratio_step=expand_ratio_step,
                                                                  out_channels_step=out_channels_step,
                                                                  stage_head=stage_head,
                                                                  stage_tail=stage_tail),
        '3x3_MBConv6': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                              stage_tail: MBInvertedConvLayer(in_C, out_C, 3, S, 6,
                                                              expand_ratio_step=expand_ratio_step,
                                                              out_channels_step=out_channels_step,
                                                              stage_head=stage_head,
                                                              stage_tail=stage_tail),
        #######################################################################################
        '5x5_MBConv6': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                              stage_tail: MBInvertedConvLayer(in_C, out_C, 5, S, 6,
                                                              expand_ratio_step=expand_ratio_step,
                                                              out_channels_step=out_channels_step,
                                                              stage_head=stage_head,
                                                              stage_tail=stage_tail),
        #######################################################################################
        '3x3_MBConv6_se': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                 stage_tail: MBInvertedConvLayer(in_C, out_C, 3, S,
                                                                 6, use_se=True,
                                                                 expand_ratio_step=expand_ratio_step,
                                                                 out_channels_step=out_channels_step,
                                                                 stage_head=stage_head,
                                                                 stage_tail=stage_tail,
                                                                 ),
        #######################################################################################
        '5x5_MBConv6_se': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                 stage_tail: MBInvertedConvLayer(in_C, out_C, 5, S,
                                                                 6, use_se=True,
                                                                 expand_ratio_step=expand_ratio_step,
                                                                 out_channels_step=out_channels_step,
                                                                 stage_head=stage_head,
                                                                 stage_tail=stage_tail,
                                                                 ),
        ###################h_swish####################################################################
        '3x3_MBConv1_h_swish': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                      stage_tail: MBInvertedConvLayer(in_C, out_C,
                                                                      3, S, 1,
                                                                      act_func='h_swish',
                                                                      expand_ratio_step=expand_ratio_step,
                                                                      out_channels_step=out_channels_step,
                                                                      stage_head=stage_head,
                                                                      stage_tail=stage_tail),
        '3x3_MBConv6_h_swish': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                      stage_tail: MBInvertedConvLayer(in_C, out_C,
                                                                      3, S, 6,
                                                                      act_func='h_swish',
                                                                      expand_ratio_step=expand_ratio_step,
                                                                      out_channels_step=out_channels_step,
                                                                      stage_head=stage_head,
                                                                      stage_tail=stage_tail),
        #######################################################################################
        '5x5_MBConv6_h_swish': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                      stage_tail: MBInvertedConvLayer(in_C, out_C,
                                                                      5, S, 6,
                                                                      act_func='h_swish',
                                                                      expand_ratio_step=expand_ratio_step,
                                                                      out_channels_step=out_channels_step,
                                                                      stage_head=stage_head,
                                                                      stage_tail=stage_tail),
        #######################################################################################
        '3x3_MBConv6_se_h_swish': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                         stage_tail: MBInvertedConvLayer(in_C,
                                                                         out_C, 3,
                                                                         S, 6,
                                                                         use_se=True,
                                                                         act_func='h_swish',
                                                                         expand_ratio_step=expand_ratio_step,
                                                                         out_channels_step=out_channels_step,
                                                                         stage_head=stage_head,
                                                                         stage_tail=stage_tail,
                                                                         ),
        #######################################################################################
        '5x5_MBConv6_se_h_swish': lambda in_C, out_C, S, expand_ratio_step, out_channels_step, stage_head,
                                         stage_tail: MBInvertedConvLayer(in_C,
                                                                         out_C, 5,
                                                                         S, 6,
                                                                         use_se=True,
                                                                         act_func='h_swish',
                                                                         expand_ratio_step=expand_ratio_step,
                                                                         out_channels_step=out_channels_step,
                                                                         stage_head=stage_head,
                                                                         stage_tail=stage_tail,
                                                                         ),
    })
    return [
        name2ops[name](in_channels, out_channels, stride, expand_ratio_step, out_channels_step, stage_head, stage_tail,
                       )
        for name in
        candidate_ops
    ]


class MixedEdge(MyModule):
    MODE = None  # gumbel, full, two, None, full_v2

    def __init__(self, candidate_ops, stage_head=False, freeze_old_block_tau=False):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        nn.init.uniform_(self.AP_path_alpha, -0.25, 0.25)
        self.freeze_old_block_tau = freeze_old_block_tau
        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None
        self.tau = 5.0
        self.tau_soft = 5.0
        self.one_hot = None
        self.one_hot_1 = None
        self.one_hot_2 = None
        self.stage_head = stage_head
        if self.stage_head:
            assert self.candidate_ops[
                       0].sub_out_channel_list is not None, 'self.candidate_ops[0] should have output channel mask'
            self.alpha_out = Parameter(torch.Tensor(len(self.candidate_ops[0].sub_out_channel_list)))
            nn.init.uniform_(self.alpha_out, -0.25, 0.25)
            self.soft_gumbel_out = None

    def net2deepernet(self):
        # self.candidate_ops is nn.ModuleList([MBInvertedConvLayer,...,MBInvertedConvLayer,zero])
        for op in self.candidate_ops:
            op.net2deepernet()
        return None

    def wider_out_channel(self, wider_out_channel, random_init=False):
        for op in self.candidate_ops:
            op.wider_out_channel(wider_out_channel, random_init=random_init)

    def wider_in_channel(self, wider_in_channel, random_init=False):
        for op in self.candidate_ops:
            op.wider_in_channel(wider_in_channel, random_init=random_init)

    def wider_expand_rate(self, wider_expand_rate, random_init=False):
        for op in self.candidate_ops:
            op.wider_expand_rate(wider_expand_rate, random_init=random_init)

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops.data
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        for op in self.candidate_ops:
            if isinstance(op, MBInvertedConvLayer):
                entropy += op.entropy(eps=eps)
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def forward(self, x):
        one_hot = self.one_hot.cuda(x.device)
        output = sum(w * op(x) if w > 0 else w for w, op in zip(one_hot, self.candidate_ops))
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    def sample_flops(self, x, in_channel_effective_num=None):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops, out_channel_effective_num, op_x = 0, 0, None
        for w, op in zip(self.one_hot, self.candidate_ops):
            if w > 0:
                op_flops, op_x, op_out_channel_effective_num = op.get_flops(x,
                                                                            in_channel_effective_num=in_channel_effective_num)
                flops = torch.add(flops, op_flops)
                out_channel_effective_num = torch.add(out_channel_effective_num, op_out_channel_effective_num)
            else:
                flops = torch.add(flops, w)
                out_channel_effective_num = torch.add(out_channel_effective_num, w)
        out_channel_effective_num = torch.div(out_channel_effective_num, len(self.active_index))
        return flops, op_x, out_channel_effective_num

    def set_probs(self):
        self.log_prob = None
        # binarize according to probs
        probs = self.probs_over_ops
        if MixedEdge.MODE == 'gumbel_1_path':
            self.one_hot, active_index_1 = gumbel_softmax(self.AP_path_alpha, tau=self.tau, hard=True, dim=0)
            self.active_index = [active_index_1.data.item()]
            self.current_prob_over_ops = probs
            for op in self.candidate_ops:
                op.set_probs()
        elif MixedEdge.MODE == 'gumbel_2_path':
            self.one_hot_1, active_index_1 = gumbel_softmax(self.AP_path_alpha, tau=self.tau, hard=True, dim=0)
            self.one_hot_2, active_index_2 = gumbel_softmax(self.AP_path_alpha, tau=self.tau, hard=True, dim=0)
            while active_index_2 == active_index_1:
                self.one_hot_2, active_index_2 = gumbel_softmax(self.AP_path_alpha, tau=1.0, hard=True, dim=0)
            self.active_index = [active_index_1.data.item(), active_index_2.data.item()]
            self.one_hot = torch.div((self.one_hot_1 + self.one_hot_2), 2.0)
            self.current_prob_over_ops = probs
            for op in self.candidate_ops:
                op.set_probs()
        else:
            self.one_hot, active_index_1 = gumbel_softmax(self.AP_path_alpha, tau=0.01, hard=True, dim=0)
            self.current_prob_over_ops = probs
            for op in self.candidate_ops:
                op.set_probs()
            sample = active_index_1.data.item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + [_i for _i in range(sample + 1, self.n_choices)]
            self.log_prob = torch.log(probs[sample])

        if self.stage_head:
            # set this layer all ops have the same soft_gumbel_out
            self.soft_gumbel_out = gumbel_softmax(self.alpha_out, tau=self.tau_soft,
                                                  hard=False, dim=0)
            for op in self.candidate_ops:
                op.set_soft_gumbel_out(soft_gumbel_out=self.soft_gumbel_out)
                op.alpha_out = self.alpha_out

        # avoid over-regularization
        # https://github.com/microsoft/nni/blob/ \
        # f1ce1648b24d2668c2eb8fa02b158a7b6da80ea4/ \
        # src/sdk/pynni/nni/nas/pytorch/proxylessnas/mutator.py#L229 \
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_tau(self, tau, tau_soft=None):
        if self.freeze_old_block_tau:
            self.tau = 0.1
        else:
            self.tau = tau
        self.tau_soft = tau_soft
        for op in self.candidate_ops:
            try:
                op.set_tau(tau=tau, tau_soft=tau_soft)
            except AttributeError:
                continue

    def get_soft_gumbel_out(self):
        assert self.stage_head, 'it should be stage_head'
        return self.soft_gumbel_out

    def set_soft_gumbel_out(self, soft_gumbel_out):
        for op in self.candidate_ops:
            op.set_soft_gumbel_out(soft_gumbel_out)


if __name__ == '__main__':
    # a = torch.zeros(3, 2, 2, requires_grad=False)
    # b = torch.zeros(3, 3, 1, 1, requires_grad=False)
    # a = a + 2
    # for i in range(len(b)):
    #     print('i', i)
    #     b[i][:2][:][:] = 1
    # print(b)
    # print(a)
    # print(b[0] * a)
    # print(a * b[0])
    def f():
        return 0, 1, 2, 3


    print(f()[3])
