"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
import copy
import numpy as np
import pandas as pd
import torch as th
from torch import nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: Non-Linear Projection Head (z) and representation (h).

    Encoder is used for contrastive learning. It gets transformed images, and returns projection head as well as
    representation.
    """
    def __init__(self, options):
        super(CNNEncoder, self).__init__()
        # Container to hold layers of the architecture in order
        self.layers = nn.ModuleList()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Get the dimensions of all layers
        dims = options["conv_dims"]
        # Input image size. Example: 28 for a 28x28 image.
        img_size = self.options["img_size"]
        # Get dimensions for convolution layers in the following format: [i, o, k, s, p, d]
        # i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
        convolution_layers = dims[:-1]
        # Final output dimension of encoder i.e. dimension of projection head
        output_dim = dims[-1]
        # Go through convolutional layers
        for layer_dims in convolution_layers:
            i, o, k, s, p, d = layer_dims
            self.layers.append(nn.Conv2d(i, o, k, stride=s, padding=p, dilation=d))
            # BatchNorm if True
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm2d(o))
            # Add activation
            self.layers.append(nn.LeakyReLU(inplace=False))
            # Dropout if True
            if options["isDropout"]:
                self.layers.append(nn.Dropout2d(options["dropout_rate"]))
        # Do global average pooling over spatial dimensions to make Encoder agnostic to input image size
        self.global_ave_pool = global_ave_pool
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(o, output_dim)
        # Last linear layer for final projection
        self.linear_layer2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # batch size, height, width, channel of the input
        bs, h, w, ch = x.size()
        # Forward pass on convolutional layers
        for layer in self.layers:
            x = layer(x)
        # Global average pooling over spatial dimensions. This is also used as learned representation.
        h = self.global_ave_pool(x)
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = F.relu(self.linear_layer1(h))
        # Apply final linear layer
        z = self.linear_layer2(z)
        return z, h


class Classifier(nn.Module):
    def __init__(self, options):
        super(Classifier, self).__init__()

        self.options = copy.deepcopy(options)
        # Add hidden layers
        self.l1 = nn.Linear(self.options["dims"][-1], 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, 1)
        self.probs = nn.Sigmoid()

    def forward(self, h):
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        logits = self.logits(h)
        probs = self.probs(logits)
        return probs


class Flatten(nn.Module):
    "Flattens tensor to 2D: (batch_size, feature dim)"
    def forward(self, x):
        return x.view(x.shape[0], -1)

def global_ave_pool(x):
    """Global Average pooling of convolutional layers over the spatioal dimensions.
    Results in 2D tensor with dimension: (batch_size, number of channels) """
    return th.mean(x, dim=[2, 3])

def compute_image_size(args):
    """Computes resulting image size after a convolutional layer
    i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
    old_size = size of input image,
    new_size= size of output image.
    """
    old_size, i, o, k, s, p, d = args
    new_size = int((old_size+2*p-d*(k-1)-1)//s) + 1
    return new_size

