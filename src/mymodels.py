import torch
import torch.nn as nn


class Vanilla_Dense(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(Vanilla_Dense, self).__init__()
        self.first_hidden = nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.FullyConnectedOutput = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out

