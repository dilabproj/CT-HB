from typing import Dict, List, Any
import torch
import torch.nn as nn


class CDAE(nn.Module):
    def __init__(self,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 signal_len: int,
                 dropout_ratio: float = 0.2,
                 **kwargs):  # pylint: disable=unused-argument
        super(CDAE, self).__init__()
        self.last_layer_filter = n_variate

        encoder: List[nn.Module] = []
        for layer_filter in model_structure['encoder']:
            encoder.append(nn.Conv1d(self.last_layer_filter, layer_filter, 5, padding=2))
            encoder.append(nn.ReLU(inplace=True))
            encoder.append(nn.MaxPool1d(2))
            self.last_layer_filter = layer_filter
        self.encoder = nn.Sequential(*encoder)

        decoder: List[nn.Module] = []
        for layer_filter in model_structure['decoder']:
            decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
            decoder.append(nn.Conv1d(self.last_layer_filter, layer_filter, 5, padding=2))
            decoder.append(nn.ReLU(inplace=True))
            self.last_layer_filter = layer_filter
        decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if signal_len == 300:
            decoder.append(nn.Conv1d(self.last_layer_filter, n_variate, 5, padding=4))
        else:
            decoder.append(nn.Conv1d(self.last_layer_filter, n_variate, 5, padding=2))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)
