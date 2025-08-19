from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self,
                 n_variate: int,
                 model_structure: Dict[str, List[Any]],
                 dropout_ratio: float = 0.2,
                 **kwargs):  # pylint: disable=unused-argument
        super(VAE, self).__init__()
        self.latent_dim = 10
        self.encoder = nn.LSTM(n_variate, 128, batch_first=True, bidirectional=True)

        self.context_to_mu = nn.Linear(256, self.latent_dim)
        self.context_to_logvar = nn.Linear(256, self.latent_dim)

        self.decoder = nn.LSTM(self.latent_dim, n_variate, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(280 * 2, 280)

    def reparameterize(self, mu, logvar):
        sd = torch.exp(logvar * 0.5)
        eps = Variable(torch.randn(sd.size())).cuda(7) # Sample from standard normal
        z = eps.mul(sd).add_(mu)
        return z

    def bottleneck(self, h):
        mu, logvar = F.softplus(self.context_to_mu(h)), F.softplus(self.context_to_logvar(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):  # pylint: disable=arguments-differ
        _, (_, final_state) = self.encoder(x.view(x.size(0), 280, 1))
        final_state = final_state.view(1, 2, x.size(0), 128)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = torch.cat([h_1, h_2], 1)
        z, mu, logvar = self.bottleneck(final_state)
        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        z = torch.cat([z] * 280, 1).view(x.size(0), 280, self.latent_dim)
        out, _ = self.decoder(z)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out)
        return torch.sigmoid(out), kld
