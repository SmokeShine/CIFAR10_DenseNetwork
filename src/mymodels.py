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
        # cross entropy requires logits and not probabilities
        return out


# https://paperswithcode.com/method/alexnet
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(AlexNet, self).__init__()
        # https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-22_at_6.35.45_PM.png

        ################ Notes ################
        # Thus, compared to standard feedforward neural networks with similarly-sized layers, CNNs have
        # much fewer connections and parameters and so they are easier to train, while their theoretically-best
        # performance is likely to be only slightly worse.

        # It contains eight learned layers —
        # five convolutional and three fully-connected

        # Deep convolutional neural networks with ReLUs train several times faster than their
        # equivalents with tanh units.

        #  Current GPUs
        # are particularly well-suited to cross-GPU parallelization, as they are able to read from and write to
        # one another’s memory directly, without going through host machine memory

        # If we set s < z, we obtain overlapping pooling

        # The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3
        # with a stride of 4 pixels

        # The second convolutional layer takes as input the (response-normalized
        # and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48.

        # The third, fourth, and fifth convolutional layers are connected to one another without any intervening
        # pooling or normalization layers

        # The third convolutional layer has 384 kernels of size 3 × 3 ×
        # 256 connected to the (normalized, pooled) outputs of the second convolutional layer

        # The fourth
        # convolutional layer has 384 kernels of size 3 × 3 × 192

        # the fifth convolutional layer has 256
        # kernels of size 3 × 3 × 192

        # The fully-connected layers have 4096 neurons each.

        # The first form of data augmentation consists of generating image translations and horizontal reflections.
        # Without this scheme, our network suffers from substantial overfitting, which would have
        # forced us to use much smaller networks

        # The second form of data augmentation consists of altering the intensities of the RGB channels in
        # training images.abs(This scheme approximately captures an important property of natural images,
        # namely, that object identity is invariant to changes in the intensity and color of the illumination)

        # Combining the predictions of many different models is a very successful way to reduce test errors
        # [1, 3], but it appears to be too expensive for big neural networks that already take several days
        # to train
        # The neurons which are
        # “dropped out” in this way do not contribute to the forward pass and do not participate in backpropagation. So every time an input is presented, the neural network samples a different architecture,
        # but all these architectures share weights

        # This technique reduces complex co-adaptations of neurons,
        # since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to
        # learn more robust features that are useful in conjunction with many different random subsets of the
        # other neurons

        # We use dropout in the first two fully-connected layers
        # Dropout roughly doubles the number of iterations required to converge.

        # We trained our models using stochastic gradient descent
        # with a batch size of 128 examples, momentum of 0.9, and
        # weight decay of 0.0005.

        # We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01.

        # We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
        # as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates
        # the early stages of learning by providing the ReLUs with positive inputs
        ################ End of Notes ################
        # stride of 4
        self.conv2d_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2
        )

        self.conv2d_2 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=5, padding=2
        )

        self.conv2d_3 = nn.Conv2d(
            in_channels=192, out_channels=384, kernel_size=3, padding=1
        )

        self.conv2d_4 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1
        )

        self.conv2d_5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        # max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 2048 fully connected 1
        self.first_fully_connected = nn.Linear(in_features=384, out_features=4096)
        # 2048 fully connected 2
        self.second_fully_connected = nn.Linear(in_features=4096, out_features=4096)
        # output layer
        self.FullyConnectedOutput = nn.Linear(
            in_features=4096, out_features=num_classes
        )
        # dropout
        self.dropout = nn.Dropout()
        # relu
        self.relu = nn.ReLU()

        self.load_initialization_weights()

    def load_initialization_weights(self):
        self.conv2d_1.weight.data.normal_(mean=0.0, std=1.0)
        self.conv2d_2.weight.data.normal_(mean=0.0, std=1.0)
        self.conv2d_3.weight.data.normal_(mean=0.0, std=1.0)
        self.first_fully_connected.weight.data.normal_(mean=0.0, std=1.0)
        self.second_fully_connected.weight.data.normal_(mean=0.0, std=1.0)
        self.FullyConnectedOutput.weight.data.normal_(mean=0.0, std=1.0)

        self.conv2d_1.bias.data = torch.tensor(1.0)
        self.conv2d_2.bias.data = torch.tensor(1.0)
        self.conv2d_3.bias.data = torch.tensor(1.0)
        self.first_fully_connected.bias.data = torch.tensor(1.0)
        self.second_fully_connected.bias.data = torch.tensor(1.0)
        self.FullyConnectedOutput.bias.data = torch.tensor(1.0)

    def forward(self, x):
        out = None
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2d_3(x)
        x = self.relu(x)

        # Flatten
        x = x.reshape(len(x), -1)

        x = self.first_fully_connected(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.second_fully_connected(x)
        x = self.dropout(x)
        x = self.relu(x)

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
