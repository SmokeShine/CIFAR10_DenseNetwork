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


class Vanilla_Dense3(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(Vanilla_Dense3, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/method/alexnet
class AlexNet(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/method/resnet
class ResNet(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/method/vgg-16
class VGGNet(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/method/inception-v4
class Inception(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/method/xception
class Xception(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out


# https://paperswithcode.com/paper/shufflenet-an-extremely-efficient
class ShuffleNet(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        self.first_hidden = nn.Linear(
            in_features=input_dim, out_features=first_hidden_size
        )
        self.sigmoid = nn.Sigmoid()
        self.second_hidden = nn.Linear(
            in_features=first_hidden_size, out_features=second_hidden_size
        )
        self.FullyConnectedOutput = nn.Linear(
            in_features=second_hidden_size, out_features=num_classes
        )

    def forward(self, x):
        out = None
        x = x.reshape(len(x), -1)
        x = self.first_hidden(x)
        x = self.sigmoid(x)
        x = self.second_hidden(x)
        x = self.sigmoid(x)
        out = self.FullyConnectedOutput(x)
        return out
