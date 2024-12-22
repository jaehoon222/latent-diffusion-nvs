import torch
import torch.nn as nn
import numpy as np


class MultiEmbedder(nn.Module):
    def __init__(self, keys, n_positions, n_channels, n_embed, bias=False):
        super().__init__()
        self.keys = keys
        self.n_positions = n_positions
        self.n_channels = n_channels
        self.n_embed = n_embed
        self.fc = nn.Linear(self.n_channels, self.n_embed, bias=bias)

    def forward(self, **kwargs):
        values = [kwargs[k] for k in self.keys]
        inputs = list()
        for k in self.keys:
            # print(kwargs["t"])
            entry = kwargs[k].reshape(kwargs[k].shape[0], -1, self.n_channels)
            inputs.append(entry)
        x = torch.cat(inputs, dim=1)
        assert not torch.isnan(x).any()
        # max_value = torch.max(x)

        # print("Максимальное значение в тензоре:", max_value.item())
        assert x.shape[1] == self.n_positions, x.shape
        x = self.fc(x)
        assert not torch.isnan(x).any()
        return x
