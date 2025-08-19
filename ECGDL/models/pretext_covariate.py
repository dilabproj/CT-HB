from typing import Dict, List, Any, Type, Union

import logging
import torch
import torch.nn as nn

from ECGDL.models import StandfordModel
from ECGDL.models.causal_cnn import CausalCNNEncoder

# Initiate Logger
logger = logging.getLogger(__name__)


class PIRL(nn.Module):
    # Sample config:
    # config.model_args = {
    #         "model_name": StandfordModel,
    #         "model_structure": {
    #             'feature_layer': [(64, kernel_sz)],
    #             'residual': [
    #                 [(64, kernel_sz), (64, kernel_sz)],
    #                 [(64, kernel_sz), (64, kernel_sz)],
    #                 [(64, kernel_sz), (64, kernel_sz)],
    #                 [(64, kernel_sz), (64, kernel_sz)],
    #                 [(128, kernel_sz), (128, kernel_sz)],
    #                 [(128, kernel_sz), (128, kernel_sz)],
    #                 [(128, kernel_sz), (128, kernel_sz)],
    #                 [(128, kernel_sz), (128, kernel_sz)],
    #                 [(256, kernel_sz), (256, kernel_sz)],
    #                 [(256, kernel_sz), (256, kernel_sz)],
    #                 [(256, kernel_sz), (256, kernel_sz)],
    #                 [(256, kernel_sz), (256, kernel_sz)],
    #                 [(512, kernel_sz), (512, kernel_sz)],
    #                 [(512, kernel_sz), (512, kernel_sz)],
    #                 [(512, kernel_sz), (512, kernel_sz)],
    #                 [(512, kernel_sz), (512, kernel_sz)],
    #             ],
    #             'hb_specific_layers': [
    #                 [512, 128],
    #                 [512, 128],
    #                 [512, 128],
    #             ]
    #         },
    #         "dropout_ratio": 0.3,
    #         "signal_len": 5000,
    #     }
    def __init__(self,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 model_name: Type[torch.nn.Module],
                 signal_len: int = 300,
                 dropout_ratio: float = 0.6,
                 **kwargs):  # pylint: disable=unused-argument
        super(PIRL, self).__init__()

        self.last_layer_filter = n_variate
        self.model_name = model_name

        if model_name is StandfordModel:
            model = StandfordModel(
                n_class=2, signal_len=signal_len, n_variate=n_variate, model_structure=model_structure)
            self.convs: Union[nn.Sequential, CausalCNNEncoder] = nn.Sequential(*list(model.children())[:-2])
            self.last_layer_filter = model_structure['residual'][-1][-1][0]
        elif model_name is CausalCNNEncoder:
            self.convs = CausalCNNEncoder(in_channels=n_variate, **model_structure)
            self.last_layer_filter = 320
        else:
            logger.warning("No matched model structure!")

        # Task specific layers
        self.hb_specific_layers = nn.ModuleDict({})
        last_layer_filter_first = self.last_layer_filter
        for idx, layer_attrs in enumerate(model_structure['hb_specific_layers']):
            hb_specific_layers = []
            self.last_layer_filter = last_layer_filter_first
            for layer_filter in layer_attrs:
                hb_specific_layers.extend([
                    nn.Linear(self.last_layer_filter, layer_filter)
                ])
                self.last_layer_filter = layer_filter
            self.hb_specific_layers[f'hb{idx}'] = nn.Sequential(*hb_specific_layers)

    def forward(self, anchor, positive, negative):  # pylint: disable=arguments-differ
        anchor_out = self.convs(anchor)
        positive_out = self.convs(positive)
        negative_out = self.convs(negative)

        # TODO: maybe can change to maxpooling (tested no significant improvment)
        if self.model_name is not CausalCNNEncoder:
            # Global max pooling
            anchor_out, _ = torch.max(anchor_out, 2)
            positive_out, _ = torch.max(positive_out, 2)
            negative_out, _ = torch.max(negative_out, 2)

        anchor_output_logits = self.hb_specific_layers['hb0'](anchor_out)
        anchor_output_logits = nn.functional.normalize(anchor_output_logits, p=2, dim=1)
        if len(self.hb_specific_layers) == 2:
            positive_output_logits = self.hb_specific_layers['hb1'](positive_out)
        else:
            positive_output_logits = self.hb_specific_layers['hb0'](positive_out)
        positive_output_logits = nn.functional.normalize(positive_output_logits, p=2, dim=1)
        negative_output_logits = self.hb_specific_layers['hb0'](negative_out)
        negative_output_logits = nn.functional.normalize(negative_output_logits, p=2, dim=1)

        return anchor_output_logits, positive_output_logits, negative_output_logits
