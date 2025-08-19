from typing import Dict, List, Any, Optional, Type

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models import StandfordModel

# Initiate Logger
logger = logging.getLogger(__name__)


class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv(x)
        return F.relu(x, inplace=True)


class SSSTM(nn.Module):
    # Sample config:
    # config.model_args = {
    #     "model_structure":  { 'shared_layers': [
    #                                   ('conv', 32, 32,
    #                                   ('conv', 32, 32),
    #                                   ('maxpool', 8, 2),
    #                                   ('conv', 64, 32),
    #                                   ('conv', 64, 32),
    #                                   ('maxpool',8, 1),
    #                                   ('conv', 128, 32),
    #                                   ('conv', 128, 32),
    #                             ],
    #                             'task_specific_layers':[
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                                 [(128, 128, 2)],
    #                             ]
    #                         },
    # }
    def __init__(self,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 signal_len: Optional[int] = None,
                 model_name: Type[torch.nn.Module] = None,
                 dropout_ratio: float = 0.6,
                 **kwargs):  # pylint: disable=unused-argument
        super(SSSTM, self).__init__()

        self.last_layer_filter = n_variate

        if model_name is None:
            shared_layers: List[nn.Module] = []
            for layer_attrs in model_structure['shared_layers']:
                layer_name, attr1, attr2 = layer_attrs
                if layer_name == 'conv':
                    layer_filter, kernel_size = attr1, attr2
                    padding_size = kernel_size // 2
                    shared_layers.append(BasicConv1d(
                        self.last_layer_filter, layer_filter, kernel_size=kernel_size, padding=padding_size))
                elif layer_name == 'maxpool':
                    kernel_size, stride_size = attr1, attr2
                    shared_layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size))
                self.last_layer_filter = layer_filter
            self.convs = nn.Sequential(*shared_layers)
        elif model_name is StandfordModel:
            assert signal_len is not None, "Using StandfordModel should specify signal length!"
            model = StandfordModel(
                n_class=2, signal_len=signal_len, n_variate=n_variate, model_structure=model_structure)
            self.convs = nn.Sequential(*list(model.children())[:-1])
            self.last_layer_filter = model_structure['residual'][-1][-1][0]
        else:
            logger.warning("Unidentified model name!")

        # Task specific layers
        self.task_specific_layers = nn.ModuleDict({})
        last_layer_filter_first = self.last_layer_filter
        for idx, layer_attrs in enumerate(model_structure['task_specific_layers']):
            task_specific_layer: List[nn.Module] = []
            self.last_layer_filter = last_layer_filter_first
            for layer_filter in layer_attrs:
                task_specific_layer.append(nn.Linear(self.last_layer_filter, layer_filter))
                task_specific_layer.append(nn.Dropout(0.6))
                self.last_layer_filter = layer_filter
            task_specific_layer.append(nn.Sigmoid())
            self.task_specific_layers[f'task{idx}'] = nn.Sequential(*task_specific_layer)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.convs(x)
        # Global max pooling
        out, _ = torch.max(out, 2)

        output_logits, output_prob = [], []
        for key in self.task_specific_layers:
            output = self.task_specific_layers[key](out)
            output_logits.append(output)
            output_prob.append(F.softmax(output, dim=1))
        return output_prob, output_logits
