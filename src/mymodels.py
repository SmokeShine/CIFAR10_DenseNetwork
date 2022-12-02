import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        # https://production-media.paperswithcode.com/methods/Screen_Shot_2020-09-25_at_10.26.40_AM_SAB79fQ.png

        ################ Notes ################
        # Deep networks naturally integrate low/mid/high-level features [50] and
        # classifiers in an end-to-end multi-layer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth)

        # notorious problem of vanishing/exploding gradients largely addressed by normalized initialization and intermediate normalization layers

        # When deeper networks are able to start converging,
        # a degradation problem has been exposed: with the network depth increasing,
        # accuracy gets saturated (which might be unsurprising) and then degrades rapidly.

        # such degradation is not caused by overfitting,
        # and adding more layers to a suitably deep model leads to higher training error,

        # we explicitly let these layers fit a residual mapping.
        # We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.

        # the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers

        # The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers.

        # shallow = shallow + identity.. but performance degrades for RHS (deep network), implying identity is difficult to approximate from dense layers

        #  The function F(x,{Wi})
        #  represents the residual mapping to be learned

        # We can fairly compare plain/residual networks that simultaneously have the same number of parameters, depth, width, and computational cost
        # No additional parameter and element wise multiplication only

        # applicable to convolutional layers and dense.

        # fewer filters and lower complexity than VGG nets
        # 34-layer baseline has 3.6 billion FLOPs (multiply-adds), which is only 18% of VGG-19 (19.6 billion FLOPs).

        # When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options:
        # (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter;
        # (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1 ×  1 convolutions)
        # For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

        # scale augmentation, horizontal flip, crop, standard color augmentation

        # batch normalization (BN) right after each convolution and before activation

        # We use SGD with a mini-batch size of 256.
        # The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60×10^4 iterations

        # We use a weight decay of 0.0001 and a momentum of 0.9. We do not use dropout

        # average the scores at multiple scales (images are resized such that the shorter side is in  {224,256,384,480,640}).

        # We evaluate both top-1 and top-5 error rates.

        # optimization difficulty is unlikely to be caused by vanishing gradients. -  because batch normalization was used which ensure there is non zero variance in forward and verify manually the backward gradients

        # ResNet eases the optimization by providing faster convergence at the early stage

        # projection shortcuts are not essential for addressing the degradation problem
        # if The dimensions of  x and  F are not equal, then applying linear projection to match dimension

        # If the identity shortcut in Fig. 5 (right) is replaced with projection, one can show that the time complexity and model size are doubled,
        # as the shortcut is connected to the two high-dimensional ends. So identity shortcuts lead to more efficient models for the bottleneck designs.

        # We combine six models of different depth to form an ensemble - comparison with state of the art

        # CIFAR10
        # The network inputs are 32×32 images, with the per-pixel mean subtracted.
        # The first layer is 3×3 convolutions.
        # Then we use a stack of 6n layers with 3×3 convolutions on the
        # feature maps of sizes {32,16,8} respectively, with 2n layers for each feature map size.
        # The numbers of filters are {16,32,64} respectively
        # stride of 2
        # global average pooling
        # 10-way fully-connected layer, and softmax.
        # totally 6n+2 stacked weighted layers
        # 3n shortcuts - connected to the pairs of 3×3 layers
        # weight decay of 0.0001
        # momentum of 0.9
        # weight initialization ???? https://www.arxiv-vanity.com/papers/1502.01852/
        # https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ac8a913c051976a3f41f20df7d6126e57.html

        # Batch normalization
        # no drop out

        # Augmentation - 4 pixels are padded on each side
        # 32×32 random crop
        # horizontal flip

        # we use 0.01 to warm up the training until the training error is below 80% (about 400 iterations), and then go back to 0.1 and continue training

        # ResNets have generally smaller responses than their plain counterparts. - emulating identity function

        # impose regularization via deep and thin architectures by design - so no dropout or maxout as number of nodes are less

        # We adopt Faster R-CNN [32] as the detection method. Here we are interested in the improvements of replacing VGG-16 [41] with ResNet-101.

        ################ End of Notes ################
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    # GT
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = None
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, out.size()[3])
        x = out.view(x.size(0), -1)
        out = self.linear(x)
        return out


# https://paperswithcode.com/method/vgg-16
class VGGNet(nn.Module):
    def __init__(self, input_dim, first_hidden_size, second_hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        # https://production-media.paperswithcode.com/methods/Screen_Shot_2020-09-25_at_10.26.40_AM_SAB79fQ.png

        ################ Notes ################

        # The convolutional layers mostly have 3 x 3 filters
        # and follow two simple design rules:
        # (i) for the same output feature map size, the layers have the same number of filters;
        # (ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.

        ################ End of Notes ################
        super(VGGNet, self).__init__()
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
        ################ Notes ################

        ################ End of Notes ################
        super(Inception, self).__init__()
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
        ################ Notes ################

        ################ End of Notes ################
        super(Xception, self).__init__()
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
        ################ Notes ################

        ################ End of Notes ################
        super(ShuffleNet, self).__init__()
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
