# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from modules.layers import *
import json


def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    def net2deepernet(self):
        # self.mobile_inverted_conv is MixedEdge()
        self.mobile_inverted_conv.net2deepernet()
        return None

    def wider_out_channel(self, wider_out_channel, random_init=False):
        self.mobile_inverted_conv.wider_out_channel(wider_out_channel, random_init=random_init)

    def wider_in_channel(self, wider_in_channel, random_init=False):
        self.mobile_inverted_conv.wider_in_channel(wider_in_channel, random_init=random_init)

    def wider_expand_rate(self, wider_expand_rate, random_init=False):
        self.mobile_inverted_conv.wider_expand_rate(wider_expand_rate, random_init=random_init)


    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    # @property
    def config(self, in_channel=None):
        # print('self.mobile_inverted_conv:',self.mobile_inverted_conv)
        # print('self.mobile_inverted_conv.config:', self.mobile_inverted_conv.config(in_channel=in_channel))
        mobile_inverted_conv_config, out_channel = self.mobile_inverted_conv.config(in_channel=in_channel)
        if self.shortcut is not None:
            shortcut_config = self.shortcut.config()
        else:
            shortcut_config = None
        return {
                   'name': MobileInvertedResidualBlock.__name__,
                   'mobile_inverted_conv': mobile_inverted_conv_config,
                   'shortcut': shortcut_config,
               }, out_channel

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

    def get_soft_gumbel_out(self):
        # self.mobile_inverted_conv is MixedEdge(candidate_ops=build_candidate_ops(modified_conv_candidates,stage_head=stage_head, stage_tail=stage_tail))
        # self.shortcut is IdentityLayer(input_channel, input_channel) or None.
        return self.mobile_inverted_conv.get_soft_gumbel_out()

    def set_soft_gumbel_out(self, soft_gumbel_out):
        self.mobile_inverted_conv.set_soft_gumbel_out(soft_gumbel_out)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier, for_super_proxyless_nas_nets=False,
                 n_cell_stages=None, this_period_n_cell_stages=None):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier
        self.for_super_proxyless_nas_nets = for_super_proxyless_nas_nets
        self.n_cell_stages = n_cell_stages
        self.this_period_n_cell_stages = this_period_n_cell_stages

    def forward(self, x):
        x = self.first_conv(x)
        if self.for_super_proxyless_nas_nets:
            # forward super-proxyless-nas net
            block_th = 0
            x = self.blocks[block_th](x)
            block_th += 1
            for stage_order, n_cell in enumerate(self.n_cell_stages):  # sum(n_cell_stages)=24
                for i in range(n_cell):
                    if i < self.this_period_n_cell_stages[stage_order]:
                        x = self.blocks[block_th](x)
                        if i == 0:  # and block_th!=0 or i==0 and block_th==1
                            soft_gumbel_out = self.blocks[block_th].get_soft_gumbel_out()
                            for j in range(block_th + 1, block_th + self.this_period_n_cell_stages[stage_order]):
                                self.blocks[j].set_soft_gumbel_out(soft_gumbel_out)
                    block_th += 1
        else:
            # forward nromal-proxyless-nas net
            for block in self.blocks:
                x = block(x)
        try:
            x = self.feature_mix_layer(x)
        except Exception as e:
            print('x.size():', x.size(), 'e:', e)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    # @property
    def config(self):
        block_config = []
        # let super_net set in_channel for first searchable blocks.
        in_channel = 32
        i = 0
        for block in self.blocks:
            # if it is IdentityLayer, don't append its config
            if isinstance(block, IdentityLayer):
                i += 1
                continue
            if i == 0:
                config, in_channel = block.config()
            else:
                if isinstance(block.mobile_inverted_conv, ZeroLayer):
                    continue
                config, in_channel = block.config(in_channel)
            block_config.append(config)
            i += 1
        feature_mix_layer_config = self.feature_mix_layer.config(in_channel=in_channel)
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config(),
            'blocks': block_config,
            'feature_mix_layer': feature_mix_layer_config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            if block_config['name'] == 'MobileInvertedResidualBlock':
                blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    @staticmethod
    def build_normal_net_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            if block_config['name'] == 'MobileInvertedResidualBlock':
                # build Normal MobileInvertedResidualBlock
                blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            if isinstance(block, IdentityLayer):
                continue
            if isinstance(block.mobile_inverted_conv, ZeroLayer):
                continue
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x
