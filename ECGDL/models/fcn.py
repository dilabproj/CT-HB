from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models.model import BasicConv1d


class FCNModel(nn.Module):
    # Sample model_structure: [(128, 16, False), (256, 7, False), (128, 3, False)]

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_structure: List[Tuple[int, int, bool]],
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(FCNModel, self).__init__()
        convs: List[nn.Module] = []
        self.last_layer_filter = n_variate
        for layer_attr in model_structure:
            layer_filter, kernel_size, Maxpool_flag = layer_attr
            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, kernel_size=kernel_size, dropout_ratio=dropout_ratio)
            convs.append(layer)
            if Maxpool_flag:
                mp_layer = nn.MaxPool1d(2)
                convs.append(mp_layer)

            self.last_layer_filter = layer_filter
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(self.last_layer_filter, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.convs(x)
        out = F.avg_pool1d(out, out.size(-1))  # Global Average Pooling
        out = self.fc(out.view(-1, self.last_layer_filter))
        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
