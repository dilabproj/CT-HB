import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_ratio: float = 0.0, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.dropout_ratio = dropout_ratio
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
