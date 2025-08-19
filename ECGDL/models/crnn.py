from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models.model import BasicConv1d


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, hidden_size]
        output = output.view(T, b, -1)

        return output


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):  # pylint: disable=arguments-differ
        # [b, T, hidden_size] -> [b, T, 1]
        energy = self.projection(encoder_outputs)
        attn_weights = F.softmax(energy.squeeze(-1), dim=1)
        # [b, T, hidden_size] * [b, T, 1] -> [b, hidden_size]
        outputs = torch.bmm(attn_weights.unsqueeze(-1).transpose(2, 1), encoder_outputs).squeeze(1)
        return outputs, attn_weights


class CRNN(nn.Module):
    # Sample model: {
    #     "model_structure": {
    #         'cnn': [(128, 10), (256, 7), (128, 3)],
    #         'rnn': (128, 1, True)
    #     },
    #     'add_attention_layer': False,
    #     'dropout_ratio': 0.0
    # }

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 add_attention_layer: bool = True,
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(CRNN, self).__init__()

        convs = []
        self.last_layer_filter = n_variate
        for layer_attr in model_structure['cnn']:
            layer_filter, kernel_size = layer_attr
            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, kernel_size=kernel_size, dropout_ratio=dropout_ratio)
            convs.append(layer)
            self.last_layer_filter = layer_filter
        self.cnn = nn.Sequential(*convs)

        layer_output, num_layer, bidirection_flag = model_structure['rnn']
        self.rnn = nn.LSTM(
            self.last_layer_filter, layer_output, num_layer, bidirectional=bidirection_flag)
        self.last_layer_filter = layer_output * 2 if bidirection_flag else layer_output

        self.add_attention_layer = add_attention_layer
        if self.add_attention_layer:
            self.attention = SelfAttention(self.last_layer_filter)
        self.fc = nn.Linear(self.last_layer_filter, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.cnn(x)
        out = out.permute(2, 0, 1)  # [T, b, hidden_size]

        # RNN features
        out, _ = self.rnn(out)  # [T, b, hidden_size]

        # Transpose output dimension
        out = out.transpose(0, 1)

        # Check if Attention
        if self.add_attention_layer:
            out, _ = self.attention(out)  # [b, T, hidden_size]
            out = self.fc(out)
        else:
            out = self.fc(out[:, -1, :])

        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits


class RCRNN(nn.Module):
    # Sample model: {
    #     "model_structure": {
    #         'rnn1': (128, 1, True),
    #         'cnn': [(256, 10), (256, 7), (128, 3)],
    #         'rnn2': (128, 1, True)
    #     },
    #     'add_attention_layer': True,
    #     'dropout_ratio': 0.2
    # }

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 add_attention_layer: bool = True,
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(RCRNN, self).__init__()

        self.last_layer_filter = n_variate
        layer_output, num_layer, bidirection_flag = model_structure['rnn1']
        self.rnn1 = nn.LSTM(
            self.last_layer_filter, layer_output, num_layer, bidirectional=bidirection_flag)
        self.last_layer_filter = layer_output * 2 if bidirection_flag else layer_output

        convs = []
        for layer_attr in model_structure['cnn']:
            layer_filter, kernel_size = layer_attr
            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, kernel_size=kernel_size, dropout_ratio=dropout_ratio)
            convs.append(layer)
            self.last_layer_filter = layer_filter
        self.cnn = nn.Sequential(*convs)

        layer_output, num_layer, bidirection_flag = model_structure['rnn2']
        self.rnn2 = nn.LSTM(
            self.last_layer_filter, layer_output, num_layer, bidirectional=bidirection_flag)
        self.last_layer_filter = layer_output * 2 if bidirection_flag else layer_output

        self.add_attention_layer = add_attention_layer
        if self.add_attention_layer:
            self.attention = SelfAttention(self.last_layer_filter)
        self.fc = nn.Linear(self.last_layer_filter, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = x.permute(2, 0, 1)  # [T, b, hidden_size]

        out, _ = self.rnn1(out)

        # Transpose output dimension
        out = out.permute(1, 0, 2)  # [b, T, hidden_size]

        out = self.cnn(out)
        out = out.permute(1, 0, 2)  # [T, b, hidden_size]

        # RNN features
        out, _ = self.rnn2(out)  # [T, b, hidden_size]

        # Transpose output dimension
        out = out.transpose(0, 1)

        # Check if Attention
        if self.add_attention_layer:
            out, _ = self.attention(out)  # [b, T, hidden_size]
            out = self.fc(out)
        else:
            out = self.fc(out[:, -1, :])

        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
