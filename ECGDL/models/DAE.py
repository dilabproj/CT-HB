from typing import Dict, List, Any
import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 signal_len: int,
                 dropout_ratio: float = 0.2,
                 **kwargs):  # pylint: disable=unused-argument
        super(DAE, self).__init__()
        self.last_layer_filter = n_variate * signal_len
        self.n_variate = n_variate
        self.signal_len = signal_len

        encoder: List[nn.Module] = []
        for layer_filter in model_structure['encoder']:
            encoder.append(nn.Linear(self.last_layer_filter, layer_filter))
            encoder.append(nn.ReLU(inplace=True))
            self.last_layer_filter = layer_filter
        self.encoder = nn.Sequential(*encoder)

        decoder: List[nn.Module] = []
        for layer_filter in model_structure['decoder']:
            decoder.append(nn.Linear(self.last_layer_filter, layer_filter))
            if layer_filter != 1:
                decoder.append(nn.ReLU(inplace=True))
            self.last_layer_filter = layer_filter
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x.view(x.size(0), self.n_variate, self.signal_len)
