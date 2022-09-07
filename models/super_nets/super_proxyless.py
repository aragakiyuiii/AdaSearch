# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy

from modules.mix_op import *
from models.normal_nets.proxyless_nets import *


class SuperProxylessNASNets(ProxylessNASNets):

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, period='b0',
                 this_period_n_cell_stages=None, fix_depth=False, blocks_expand_ratio_step=None,
                 blocks_out_channels_step=None, freeze_old_block_tau=False,
                 old_period_n_cell_stages=None):

        # assume each block has expand_ratio_step(0.75,3.25,0.5) and out_channels_step(16,40,8)
        assert sum(n_cell_stages) == len(blocks_expand_ratio_step)
        assert sum(n_cell_stages) == len(blocks_out_channels_step)

        self._redundant_modules = None
        self._unused_modules = None
        self.period = period
        self.this_period_n_cell_stages = this_period_n_cell_stages
        self.blocks_expand_ratio_step = blocks_expand_ratio_step
        self.blocks_out_channels_step = blocks_out_channels_step
        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first normal conv layer, output_channels is 32
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # first block with expand_ratio==1, output_channels is 12, or 16
        # if 3x3_MBConv1 out_channel is searchable, the super_net flops count also needs to write new code.
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'],
            input_channel, first_cell_width, 1, 'weight_bn_act', expand_ratio_step=[1.0, 1.0, 0],
            out_channels_step=[12, 16, 4],
        ), )
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        blocks = [first_block]
        block_th = 0
        for stage_order, (width, n_cell, s) in enumerate(zip(width_stages, n_cell_stages, stride_stages)):
            for i in range(n_cell):
                if i < self.this_period_n_cell_stages[stage_order]:
                    expand_ratio_step, out_channels_step = blocks_expand_ratio_step[block_th], blocks_out_channels_step[
                        block_th]
                    if i == 0:
                        stride = s
                        stage_head = True
                    else:
                        stride = 1
                        stage_head = False
                    # conv
                    if fix_depth:
                        # don't use zero operation in searchable blocks. So the depth is fixed.
                        modified_conv_candidates = conv_candidates
                    else:
                        if stride == 1 and input_channel == width:
                            # use zero operation in searchable blocks. So the depth is searchable.
                            modified_conv_candidates = conv_candidates + ['Zero']
                        else:
                            modified_conv_candidates = conv_candidates
                    if i + 1 == self.this_period_n_cell_stages[stage_order]:
                        stage_tail = True
                    else:
                        stage_tail = False
                    # out_channels_step[1] is out_channels which sets in imagenet_100_arch_search
                    if freeze_old_block_tau:
                        if i < old_period_n_cell_stages[stage_order]:
                            conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                                modified_conv_candidates, input_channel, out_channels_step[1], stride, 'weight_bn_act',
                                expand_ratio_step=expand_ratio_step, out_channels_step=out_channels_step,
                                stage_head=stage_head,
                                stage_tail=stage_tail,
                            ), stage_head=stage_head, freeze_old_block_tau=freeze_old_block_tau)
                        else:
                            conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                                modified_conv_candidates, input_channel, out_channels_step[1], stride, 'weight_bn_act',
                                expand_ratio_step=expand_ratio_step, out_channels_step=out_channels_step,
                                stage_head=stage_head,
                                stage_tail=stage_tail,
                            ), stage_head=stage_head)
                    else:
                        conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                            modified_conv_candidates, input_channel, out_channels_step[1], stride, 'weight_bn_act',
                            expand_ratio_step=expand_ratio_step, out_channels_step=out_channels_step,
                            stage_head=stage_head,
                            stage_tail=stage_tail,
                        ), stage_head=stage_head)

                    # 2.shortcut
                    if stride == 1 and input_channel == width and i != 0:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                    inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                    blocks.append(inverted_residual_block)
                    input_channel = out_channels_step[1]
                else:
                    expand_ratio_step, out_channels_step = blocks_expand_ratio_step[block_th], blocks_out_channels_step[
                        block_th]
                    shortcut = IdentityLayer(input_channel, input_channel)
                    blocks.append(shortcut)
                    input_channel = out_channels_step[1]

                # index to find block_th's expand_ratio_step and block_th's out_channels_step
                block_th += 1
        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        super(SuperProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier, True,
                                                    n_cell_stages, self.this_period_n_cell_stages)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def net2deepernet(self, deeper_n_cell_stages=None, width_stages=None, stride_stages=None, conv_candidates=None,
                      fix_depth=False, n_cell_stages=None):
        # width_stages=None, stride_stages can use old net config
        print(self.this_period_n_cell_stages, deeper_n_cell_stages)
        assert len(self.this_period_n_cell_stages) == len(deeper_n_cell_stages), 'they should have same stage number'
        searchable_block_i = 1
        block_th = 0  # locate the args.width
        input_channel = None
        for stage_order, (width, next_period_n_cell, n_cell, s) in enumerate(
                zip(width_stages, deeper_n_cell_stages, n_cell_stages, stride_stages)):
            # for stage_order, n_cell in enumerate(deeper_n_cell_stages)
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if i < self.this_period_n_cell_stages[stage_order]:
                    expand_ratio_step, out_channels_step = self.blocks_expand_ratio_step[block_th], \
                                                           self.blocks_out_channels_step[
                                                               block_th]
                    input_channel = out_channels_step[1]
                    # existing old blocks
                    # block index++
                elif i < next_period_n_cell:
                    expand_ratio_step, out_channels_step = self.blocks_expand_ratio_step[block_th], \
                                                           self.blocks_out_channels_step[
                                                               block_th]
                    # add deeper block
                    stride = 1
                    stage_head = False
                    if fix_depth:
                        # don't use zero operation in searchable blocks. So the depth is fixed.
                        modified_conv_candidates = conv_candidates
                    else:
                        modified_conv_candidates = conv_candidates + ['Zero']
                    conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                        modified_conv_candidates, input_channel, out_channels_step[1], stride, 'weight_bn_act',
                        expand_ratio_step=expand_ratio_step, out_channels_step=out_channels_step, stage_head=stage_head,
                    ), stage_head=stage_head)
                    shortcut = IdentityLayer(input_channel, input_channel)
                    inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                    inverted_residual_block.net2deepernet()
                    self.blocks[searchable_block_i] = inverted_residual_block
                    input_channel = out_channels_step[1]
                    # block index++
                searchable_block_i += 1
                # index to find block_th's expand_ratio_step and block_th's out_channels_step
                block_th += 1

    def net2widernet(self, wider_out_channel, random_init=False, wider_expand_rate=None):
        # searchable_block_i = 1
        args_th = 0  # locate the args.width
        for i, _ in enumerate(self.blocks):
            print('i=%d, block_i_out_channel=%d' % (i, wider_out_channel[args_th][1]))
            if i == 0:
                # blocks[0] keep out_channel 16 and expand_ratio is 1.
                continue
            if i != len(self.blocks) - 1:
                self.blocks[i].wider_out_channel(wider_out_channel[args_th][1], random_init=random_init)
                self.blocks[i + 1].wider_in_channel(wider_out_channel[args_th][1], random_init=random_init)
            else:
                self.blocks[i].wider_out_channel(wider_out_channel[args_th][1], random_init=random_init)
                self.feature_mix_layer.wider_in_channel(wider_out_channel[args_th][1], random_init=random_init)
            self.blocks[i].wider_expand_rate(wider_expand_rate[args_th][1], random_init=random_init)
            args_th += 1
        return None

    # @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            # if 'AP_path_alpha' in name or 'alpha_expand' in name or 'alpha_out' in name:
            #     yield param
            if 'AP_path_alpha' in name:
                yield param
            elif 'alpha_expand' in name:
                yield param
            elif 'alpha_out' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'alpha_expand' not in name and 'alpha_out' not in name and 'mask' not in name:
                yield param

    def net_name_parameters(self):
        for name, param in self.named_parameters():
            print(name)

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_probs(self):
        self.blocks[0].mobile_inverted_conv.set_probs()
        for m in self.redundant_modules:
            try:
                m.set_probs()
            except AttributeError as e:
                print(type(m), ' do not support reset_probs ', e)

    def set_mix_op_tau(self, tau=10, tau_soft=10):
        for m in self.redundant_modules:
            try:
                m.set_tau(tau=tau, tau_soft=tau_soft)
            except AttributeError:
                continue

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop

        block_th = 0
        # block_0
        mb_conv = self.blocks[block_th].mobile_inverted_conv
        if not isinstance(mb_conv, MixedEdge):
            delta_flop, _, in_channel_effective = self.blocks[block_th].mobile_inverted_conv.get_flops(x)
            expected_flops = expected_flops + delta_flop
        else:
            delta_flop, x_0, in_channel_effective = self.blocks[block_th].get_flops(x)
        x = self.blocks[block_th](x)
        # first block is fixed width
        block_th += 1
        for stage_order, n_cell in enumerate(self.n_cell_stages):  # sum(n_cell_stages)=24
            # in each new stage, expected_out_channel init as None
            expected_out_channel = None
            for i in range(n_cell):
                if i < self.this_period_n_cell_stages[stage_order]:
                    # count shortcut_flops
                    if self.blocks[block_th].shortcut is None:
                        shortcut_flop = 0
                    else:
                        shortcut_flop, _ = self.blocks[block_th].shortcut.get_flops(x)
                    expected_flops = expected_flops + shortcut_flop
                    # count op candidates flops
                    mb_conv = self.blocks[block_th].mobile_inverted_conv
                    assert isinstance(mb_conv, MixedEdge), 'it should be MixedEdge'
                    probs_over_ops = mb_conv.current_prob_over_ops
                    x_op = None
                    # 逐个计算每一个ops的期望值，weighted sum ops_flops
                    for j, op in enumerate(mb_conv.candidate_ops):
                        if op is None or op.is_zero_layer():
                            continue
                        op_flops, x_op, out_channel_effective_j = op.get_flops(x,
                                                                               in_channel_effective_num=in_channel_effective)
                        # op_i_flops * prob_i
                        expected_flops = expected_flops + op_flops * probs_over_ops[j]
                        if i == 0:
                            # set expected_out_channel in new stage_head.
                            if expected_out_channel is None:
                                expected_out_channel = 0
                            expected_out_channel = expected_out_channel + out_channel_effective_j * probs_over_ops[j]
                    in_channel_effective = expected_out_channel
                    x = x_op
                block_th += 1

        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def sample_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop
        block_th = 0
        # block_0
        mb_conv = self.blocks[block_th].mobile_inverted_conv
        if not isinstance(mb_conv, MixedEdge):
            delta_flop, _, in_channel_effective = self.blocks[block_th].mobile_inverted_conv.get_flops(x)
            expected_flops = expected_flops + delta_flop
        else:
            delta_flop, x_0, in_channel_effective = self.blocks[block_th].get_flops(x)
        x = self.blocks[block_th](x)
        # first block is fixed width
        block_th += 1
        for stage_order, n_cell in enumerate(self.n_cell_stages):  # sum(n_cell_stages)=24
            for i in range(n_cell):
                if i < self.this_period_n_cell_stages[stage_order]:
                    # count ops flops
                    mb_conv = self.blocks[block_th].mobile_inverted_conv
                    assert isinstance(mb_conv, MixedEdge), 'it should be MixedEdge'
                    # after: Direct count MixedEdge.flops
                    sample_flops, x, in_channel_effective = mb_conv.sample_flops(x, in_channel_effective)
                    expected_flops += sample_flops
                block_th += 1

        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        # init a new ProxylessNASNets(normal net)
        return ProxylessNASNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)

# if __name__ == '__main__':
#     width_stages = [24, 48, 88, 128, 192, 336]
