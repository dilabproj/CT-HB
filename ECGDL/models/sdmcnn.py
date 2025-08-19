from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models.model import BasicConv1d


def pad_zero(out, size):
    return F.pad(out, (0, size - out.size(-1)))


def pad_repeat(out, size):
    out_rep = out.repeat(1, 1, size // out.size(-1))
    return F.pad(out_rep, (0, size - out_rep.size(-1)))


class ShiftDilated(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shift_size: int,
                 **kwargs):
        super(ShiftDilated, self).__init__()
        self.shift_size = shift_size
        # TODO: maybe better implementation
        kwargs['dilation'] = shift_size
        kwargs['stride'] = shift_size
        self.conv = BasicConv1d(in_channels, out_channels, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        # original implementation
        return pad_zero(torch.cat([self.conv(x[:, :, i:]) for i in range(self.shift_size)], 2), x.size(-1))


class RepeatDilated(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shift_size: int,
                 **kwargs):
        super(RepeatDilated, self).__init__()
        self.shift_size = shift_size
        # TODO: maybe better implementation
        kwargs['dilation'] = shift_size
        kwargs['stride'] = shift_size
        self.conv = BasicConv1d(in_channels, out_channels, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return pad_repeat(self.conv(x), x.size(-1))


class DilatedLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shift_size: int,
                 **kwargs):
        super(DilatedLayer, self).__init__()
        self.shift_size = shift_size
        kwargs['dilation'] = shift_size
        kwargs['stride'] = shift_size
        self.conv = BasicConv1d(in_channels, out_channels, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return pad_zero(self.conv(x), x.size(-1))


class SDMBlock(nn.Module):
    DILATED_MAPPING = {
        'shifted': ShiftDilated,
        'repeated': RepeatDilated,
        'None': DilatedLayer,
    }

    def __init__(self,
                 input_channel: int,
                 filter_per_layer: int,
                 kernel_sz: List[int],
                 layer_type: str = 'default',
                 repeat_type: str = 'shifted',
                 pool_type: str = 'no',
                 glu: bool = True):
        super(SDMBlock, self).__init__()

        self.kernel_sz = kernel_sz
        self.layer_type = layer_type
        self.pool_type = pool_type
        self.use_glu = glu
        # use nn.ModuleList to register all modules in the list
        if layer_type in ['default', 'shift']:
            self.shift_dilateds = nn.ModuleList([
                SDMBlock.DILATED_MAPPING[repeat_type](
                    input_channel, filter_per_layer, shift_size=k, kernel_size=3
                )
                for k in kernel_sz
            ])
        if layer_type in ['default', 'cnn']:
            self.cnns = nn.ModuleList([
                BasicConv1d(input_channel, filter_per_layer, kernel_size=k, padding=(k - 1) // 2)
                for k in kernel_sz
            ])

        # output_channel = filter_per_layer / 2 (glu) * len(kernel_sz) * 2 (shift_dilateds + cnns)
        layer_size = {
            'default': 2,
            'shift': 1,
            'cnn': 1,
        }[layer_type]
        self.out_channel = filter_per_layer * len(kernel_sz) * layer_size
        if self.use_glu:
            self.out_channel = self.out_channel // 2

    def forward(self, x):  # pylint: disable=arguments-differ
        outputs = []
        # shift dilateds
        if self.layer_type in ['default', 'shift']:
            outputs += [
                F.glu(layer(x), 1) if self.use_glu else layer(x)  # glu layer splited feature dimension
                for layer in self.shift_dilateds
            ]
        if self.layer_type in ['default', 'cnn']:
            outputs += [
                F.glu(layer(x), 1) if self.use_glu else layer(x)  # glu layer splited feature dimension
                for layer in self.cnns
            ]

        out = torch.cat(outputs, 1)  # concat outputs along feature dimension

        if self.pool_type == 'avg_pool':
            out = F.avg_pool1d(out, 2)
        elif self.pool_type == 'max_pool':
            out = F.max_pool1d(out, 2)

        return out


class SDMCNN(nn.Module):
    # Sample model_structure:
    #     'block_args': [
    #         {'filter_per_layer': 64, 'kernel_sz': [5, 7], 'pool_type': 'no', 'glu': True},
    #         {'filter_per_layer': 32, 'kernel_sz': [3, 5], 'pool_type': 'no', 'glu': True},
    #         {'filter_per_layer': 16, 'kernel_sz': [5, 7], 'pool_type': 'no', 'glu': True},
    #     ],
    #     'dropout_ratio': 0.0,

    def __init__(self,  # pylint: disable=unused-argument
                 n_class: int,
                 n_variate: int,
                 block_args: List[Dict[str, Any]],
                 dropout_ratio: float = 0.0,
                 **kwargs):
        super(SDMCNN, self).__init__()
        # assert len(block_args) == 3, block_args
        blocks = []
        for layer_args in block_args:
            layer = SDMBlock(n_variate, **layer_args)
            n_variate = layer.out_channel
            blocks.append(layer)
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(n_variate, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        out = self.blocks(x)
        out = F.avg_pool1d(out, out.size(-1))  # GAP
        out = self.dropout(out)
        out = self.fc(out.squeeze(-1))
        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
