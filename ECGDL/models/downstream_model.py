from typing import Type, Dict, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ECGDL.models.self_supervised_signal_transform import SSSTM
from ECGDL.models.CDAE import CDAE
from ECGDL.models.DAE import DAE
from ECGDL.models.VAE import VAE
from ECGDL.models.pretext_covariate import PIRL
from ECGDL.models.causal_cnn import CausalCNNEncoder


class DownstreamModel(nn.Module):
    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 model_name: Type[torch.nn.Module],
                 model_path: str,
                 model_agrs: Dict[str, Any],
                 device: int,
                 freeze: bool,
                 linear: bool = False,
                 dropout_ratio: float = 0.6,
                 dataparallel: bool = False,
                 **kwargs):  # pylint: disable=unused-argument
        super(DownstreamModel, self).__init__()
        self.device = device
        state_dict = torch.load(model_path, map_location=device)['state_dict']
        if dataparallel:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # remove `module.`
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model_name, self.model_agrs, self.linear = model_name, model_agrs, linear

        if self.model_name is SSSTM:
            model = SSSTM(n_variate=n_variate, **model_agrs)
            model.load_state_dict(state_dict)
            self.pretrain_module = nn.Sequential(*list(model.convs.children()))
            feature_len = 320
        elif self.model_name is CDAE:
            model = CDAE(n_variate=n_variate, **model_agrs)  # type: ignore
            model.load_state_dict(state_dict)
            self.pretrain_module = nn.Sequential(*list(model.encoder.children()))  # type: ignore
            feature_len = 140
        elif self.model_name is DAE:
            model = DAE(n_variate=n_variate, **model_agrs)  # type: ignore
            model.load_state_dict(state_dict)
            self.pretrain_module = nn.Sequential(*list(model.encoder.children()))  # type: ignore
            feature_len = 328
        elif self.model_name is VAE:
            model = VAE(n_variate=n_variate, **model_agrs)  # type: ignore
            model.load_state_dict(state_dict)
            self.pretrain_module = model.encoder  # type: ignore
            self.context_to_mu = model.context_to_mu
            self.context_to_logvar = model.context_to_logvar
            feature_len = 10
        elif self.model_name is PIRL:
            model = PIRL(n_variate=n_variate, **model_agrs)  # type: ignore
            model.load_state_dict(state_dict)
            self.pretrain_module = nn.Sequential(*list(model.convs.children()))
            feature_len = 320
        elif self.model_name is CausalCNNEncoder:
            model = CausalCNNEncoder(in_channels=n_variate, **model_agrs)  # type: ignore
            model.load_state_dict(state_dict)
            self.pretrain_module = nn.Sequential(*list(model.children()))
            feature_len = 320

        # Specify whether freezing weights of model
        if freeze:
            if self.model_name is VAE:
                for param in self.context_to_mu.parameters():
                    param.requires_grad = False
                for param in self.context_to_logvar.parameters():
                    param.requires_grad = False
            for param in self.pretrain_module.parameters():
                param.requires_grad = False

        for idx, child in enumerate(self.pretrain_module.modules()):
            if idx < 87:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(feature_len),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(feature_len, n_class),
        )

    def reparameterize(self, mu, logvar):
        sd = torch.exp(logvar * 0.5)
        eps = Variable(torch.randn(sd.size())).to(self.device) # Sample from standard normal
        z = eps.mul(sd).add_(mu)
        return z

    def bottleneck(self, h):
        mu, logvar = F.softplus(self.context_to_mu(h)), F.softplus(self.context_to_logvar(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):  # pylint: disable=arguments-differ
        if self.model_name is DAE:
            x = x.view(x.size(0), -1)
        if self.model_name is VAE:
            _, (_, final_state) = self.pretrain_module(x.view(x.size(0), 280, 1))
            final_state = final_state.view(1, 2, x.size(0), 128)
            final_state = final_state[-1]
            h_1, h_2 = final_state[0], final_state[1]
            final_state = torch.cat([h_1, h_2], 1)
            out, _, _ = self.bottleneck(final_state)
        else:
            out = self.pretrain_module(x)

        if self.model_name is CDAE:
            out = out.view(out.size(0), -1)
        elif self.model_name is SSSTM:
            out, _ = torch.max(out, 2)

        if self.linear:
            return None, out
        else:
            out = self.fc_layer(out)
            output_logits = out
            out = F.softmax(output_logits, dim=1)
            output_prob = out
            return output_prob, output_logits
