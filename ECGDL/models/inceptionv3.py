import torch
import torch.nn as nn
import torch.nn.functional as F

from ECGDL.models.model import BasicConv1d


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)

        self.branch1x5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch1x5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)

        self.branch1x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch1x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch1x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch1x5 = self.branch1x5_1(x)
        branch1x5 = self.branch1x5_2(branch1x5)

        branch1x3dbl = self.branch1x3dbl_1(x)
        branch1x3dbl = self.branch1x3dbl_2(branch1x3dbl)
        branch1x3dbl = self.branch1x3dbl_3(branch1x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch1x5, branch1x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)

        self.branch1x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch1x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch1x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x3 = self.branch1x3(x)

        branch1x3dbl = self.branch1x3dbl_1(x)
        branch1x3dbl = self.branch1x3dbl_2(branch1x3dbl)
        branch1x3dbl = self.branch1x3dbl_3(branch1x3dbl)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

        outputs = [branch1x3, branch1x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_1x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 192, kernel_size=1)

        c7 = channels_1x7
        self.branch1x7_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch1x7_3 = BasicConv1d(c7, 192, kernel_size=7, padding=3)

        self.branch1x7dbl_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch1x7dbl_3 = BasicConv1d(c7, c7, kernel_size=7, padding=3)
        self.branch1x7dbl_5 = BasicConv1d(c7, 192, kernel_size=7, padding=3)

        self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch1x7 = self.branch1x7_1(x)
        branch1x7 = self.branch1x7_3(branch1x7)

        branch1x7dbl = self.branch1x7dbl_1(x)
        branch1x7dbl = self.branch1x7dbl_3(branch1x7dbl)
        branch1x7dbl = self.branch1x7dbl_5(branch1x7dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch1x7, branch1x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch1x3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch1x3_2 = BasicConv1d(192, 320, kernel_size=3, stride=2)

        self.branch1x7x3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch1x7x3_2 = BasicConv1d(192, 192, kernel_size=7, padding=3)
        self.branch1x7x3_4 = BasicConv1d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x3 = self.branch1x3_1(x)
        branch1x3 = self.branch1x3_2(branch1x3)

        branch1x7x3 = self.branch1x7x3_1(x)
        branch1x7x3 = self.branch1x7x3_2(branch1x7x3)
        branch1x7x3 = self.branch1x7x3_4(branch1x7x3)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
        outputs = [branch1x3, branch1x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 320, kernel_size=1)

        self.branch1x3_1 = BasicConv1d(in_channels, 384, kernel_size=1)
        self.branch1x3_2a = BasicConv1d(384, 384, kernel_size=3, padding=1)
        self.branch1x3_2b = BasicConv1d(384, 384, kernel_size=3, padding=1)

        self.branch1x3dbl_1 = BasicConv1d(in_channels, 448, kernel_size=1)
        self.branch1x3dbl_2 = BasicConv1d(448, 384, kernel_size=3, padding=1)
        self.branch1x3dbl_3a = BasicConv1d(384, 384, kernel_size=3, padding=1)
        self.branch1x3dbl_3b = BasicConv1d(384, 384, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

    def forward(self, x):  # pylint: disable=arguments-differ
        branch1x1 = self.branch1x1(x)

        branch1x3 = self.branch1x3_1(x)
        branch1x3 = [
            self.branch1x3_2a(branch1x3),
            self.branch1x3_2b(branch1x3),
        ]
        branch1x3 = torch.cat(branch1x3, 1)

        branch1x3dbl = self.branch1x3dbl_1(x)
        branch1x3dbl = self.branch1x3dbl_2(branch1x3dbl)
        branch1x3dbl = [
            self.branch1x3dbl_3a(branch1x3dbl),
            self.branch1x3dbl_3b(branch1x3dbl),
        ]
        branch1x3dbl = torch.cat(branch1x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch1x3, branch1x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, n_class):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv1d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv1d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, n_class)
        self.fc.stddev = 0.001

    def forward(self, x):  # pylint: disable=arguments-differ
        # N x 768 x 17 x 17
        x = F.avg_pool1d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class Inceptionv3Model(nn.Module):
    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 aux_logits: bool = False,
                 **kwargs):  # pylint: disable=unused-argument
        super(Inceptionv3Model, self).__init__()
        self.aux_logits = aux_logits
        self.Conv1d_1a_1x10 = BasicConv1d(12, 96, kernel_size=10, stride=2)
        self.Conv1d_2a_1x10 = BasicConv1d(96, 96, kernel_size=10, stride=2)
        self.Conv1d_2b_1x10 = BasicConv1d(96, 192, kernel_size=10, padding=1)
        self.Conv1d_3b_1x1 = BasicConv1d(192, 384, kernel_size=1)
        self.Conv1d_4a_1x10 = BasicConv1d(384, 768, kernel_size=10)
        self.Mixed_5b = InceptionA(768, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_1x7=128)
        self.Mixed_6c = InceptionC(768, channels_1x7=160)
        self.Mixed_6d = InceptionC(768, channels_1x7=160)
        self.Mixed_6e = InceptionC(768, channels_1x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, n_class)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, n_class)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                import scipy.stats as stats  # pylint: disable=import-outside-toplevel
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)  # type: ignore
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # pylint: disable=arguments-differ
        # N x 12 x 5000
        x = self.Conv1d_1a_1x10(x)
        # N x (12 x 8) x 2496
        x = self.Conv1d_2a_1x10(x)
        # N x (12 x 8) x 1244
        x = self.Conv1d_2b_1x10(x)
        # N x (12 x 16) x 1237
        x = F.max_pool1d(x, kernel_size=10, stride=2)
        # N x (12 x 16) x 614
        x = self.Conv1d_3b_1x1(x)
        # N x (12 x 32) x 605
        x = self.Conv1d_4a_1x10(x)
        # N x (12 x 64) x 596
        x = F.max_pool1d(x, kernel_size=10, stride=2)
        # N x (12 x 64) x 300
        x = self.Mixed_5b(x)
        # N x (12 x 64) x 1 x 300
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (n_class)

        output_logits = x
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits
