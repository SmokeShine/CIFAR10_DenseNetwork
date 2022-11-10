import torch
import torch.nn as nn


class Vanilla_Dense(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(Vanilla_Dense, self).__init__()
        self.num_outputs = num_outputs

        self.dense1 = nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        dense1 = self.dense1(input)
        dense1 = self.relu(dense1)

        dense2 = self.dense2(dense1)
        dense2 = self.relu(dense2)

        dense3 = self.dense3(dense2)
        output = self.sigmoid(dense3)

        return output
