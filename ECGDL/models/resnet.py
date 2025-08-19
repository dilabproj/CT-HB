import math
from typing import List, Any
import torch.nn as nn
import torch.nn.functional as F


class ResNetResidualBlock(nn.Module):

    def __init__(self,
                 last_layer_filter: int,
                 layer_attrs: List[Any],
                 dropout_ratio: float,
                 **kwargs):  # pylint: disable=unused-argument
        super(ResNetResidualBlock, self).__init__()

        self.last_layer_filter = last_layer_filter
        self.dropout_ratio = dropout_ratio

        convs: List[nn.Module] = []
        for idx, layer_attr in enumerate(layer_attrs):
            layer_filter, kernel_size = layer_attr

            # Determine padding size
            padding_size = math.floor((kernel_size - 1) / 2)
            layer = [
                nn.Dropout(self.dropout_ratio),
                nn.Conv1d(self.last_layer_filter, layer_filter, kernel_size, padding=padding_size)
            ]
            if kernel_size % 2 == 0:
                layer.append(nn.ConstantPad1d((0, 1), 0))

            # Determine layer structure
            convs += [
                nn.Sequential(*layer),
                nn.BatchNorm1d(layer_filter)
            ]
            if idx != len(layer_attrs) - 1:
                convs.append(nn.ReLU())

            self.last_layer_filter = layer_filter

        self.convs = nn.Sequential(*convs)
        self.short = nn.Sequential(
            nn.Conv1d(last_layer_filter, self.last_layer_filter, 1),
            nn.BatchNorm1d(self.last_layer_filter),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        short = self.short(x)
        convs = self.convs(x)
        out = convs + short

        return F.relu(out)


class ResNetModel(nn.Module):
    # Sample model_structure:
    # [[(64, 8), (64, 5), (64, 3)],
    # [(64 * 2, 8), (64 * 2, 5), (64 * 2, 3)],
    # [(64 * 2, 8), (64 * 2, 5), (64 * 2, 3)]]

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_structure: List[List[Any]],
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(ResNetModel, self).__init__()

        self.last_layer_filter = n_variate

        convs = []
        for layer_attrs in model_structure:
            layer = ResNetResidualBlock(self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio)
            convs.append(layer)
            # Take out the last layer filter size
            self.last_layer_filter = layer_attrs[-1][0]
        self.convs = nn.Sequential(*convs)

        self.linear = nn.Linear(self.last_layer_filter, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.convs(x)
        out = F.avg_pool1d(out, out.size(-1)).squeeze(2)
        out = self.linear(out)
        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
