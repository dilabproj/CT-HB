from typing import Dict, List, Any

import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models.model import BasicConv1d


class JAMAResidualBlock(nn.Module):

    def __init__(self,
                 last_layer_filter: int,
                 layer_attrs: List[Any],
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(JAMAResidualBlock, self).__init__()

        self.last_layer_filter = last_layer_filter
        convs: List[nn.Module] = []
        for idx, layer_attr in enumerate(layer_attrs):
            layer_filter, kernel_size, dropout_flag = layer_attr
            padding_size = kernel_size // 2
            dropout = dropout_ratio if dropout_flag else 0.0

            if idx == len(layer_attrs) - 1:
                convs.append(
                    nn.Sequential(
                        nn.Conv1d(
                            self.last_layer_filter, layer_filter,
                            kernel_size=kernel_size, stride=2, padding=padding_size),
                        nn.BatchNorm1d(layer_filter),
                        nn.Dropout(dropout)
                    )
                )
            else:
                convs.append(
                    BasicConv1d(
                        self.last_layer_filter, layer_filter, dropout_ratio=dropout,
                        kernel_size=kernel_size, padding=padding_size))
            self.last_layer_filter = layer_filter

        self.convs = nn.Sequential(*convs)
        self.short = nn.MaxPool1d(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        short = self.short(x)
        convs = self.convs(x)
        if short.size()[-1] != convs.size()[-1]:
            short = F.pad(short, (0, 1))
        out = convs + short

        return F.relu(out)


class JamaModel(nn.Module):
    # Sample model_structure:
    # { 'feature_layer': [(64, 25, False, False), (32, 25, False, True), (32, 25, True, True)],
    #   'residual': [[(32, 25, True), (32, 25, True)],
    #                [(32, 25, True), (32, 25, True)],
    #                [(32, 25, True), (32, 25, True)],
    #                [(32, 25, True), (32, 25, False)]
    #               ]
    # }

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 dropout_ratio: float = 0.5,
                 **kwargs):  # pylint: disable=unused-argument
        super(JamaModel, self).__init__()

        self.last_layer_filter = n_variate

        convs = []
        stride2_cnt = 0
        for layer_attr in model_structure['feature_layer']:
            layer_filter, kernel_size, stride_flag, dropout_flag = layer_attr
            padding_size = kernel_size // 2
            stride = 2 if stride_flag else 1
            dropout = dropout_ratio if dropout_flag else 0.0

            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, dropout_ratio=dropout,
                kernel_size=kernel_size, stride=stride, padding=padding_size)
            convs.append(layer)
            self.last_layer_filter = layer_filter
        self.convs = nn.Sequential(*convs)

        residual_blocks = []
        for layer_attrs in model_structure['residual']:
            residual_blocks.append(JAMAResidualBlock(self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        stride2_cnt = sum([1 if l[2] else 0 for l in model_structure['feature_layer']])
        stride2_times = len(model_structure['residual']) + stride2_cnt
        length = (5000 // (2 ** stride2_times) + 1) if stride2_times > 3 else (5000 // (2 ** stride2_times))
        self.fc = nn.Linear(self.last_layer_filter * length, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.convs(x)
        out = self.residual_blocks(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
