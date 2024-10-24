import torch
import torch.nn as nn
import torch.nn.functional as F


class DeMixing(nn.Module):
    """
    Spatial demixing Layer.

    Parameters
    ----------
    size : int
        Number of output features.
    nonlin : callable
        Activation function.
    axis : int
        Axis to apply demixing.
    specs : dict
        Dictionary of layer specifications.
    """

    def __init__(
        self,
        size,
        nonlin=nn.Identity(),
        axis=-1,
        regularize="l1_lambda",
        bias_trainable=True,
        bias_const=0,
        unitnorm=False,
    ):
        super(DeMixing, self).__init__()
        self.size = size
        self.nonlin = nonlin
        self.axis = axis
        self.bias_trainable = bias_trainable
        self.bias_const = bias_const

        self.w = nn.Parameter(torch.empty((size,)), requires_grad=True)
        self.b_in = nn.Parameter(torch.empty((size,)), requires_grad=self.bias_trainable)

        nn.init.kaiming_uniform_(self.w, nonlinearity="relu")
        nn.init.constant_(self.b_in, self.bias_const)

        self.constraint = nn.functional.normalize if unitnorm else None

        self.reg = None
        if regularize == "l1_lambda":
            self.reg = nn.L1Loss()
        elif regularize == "l2_scope":
            self.reg = nn.MSELoss()

    def forward(self, x):
        """
        Applies demixing to input tensor.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Demixed tensor.
        """
        demix = torch.tensordot(x, self.w, dims=([self.axis], [0]))
        demix = self.nonlin(demix + self.b_in)
        if self.constraint:
            demix = self.constraint(demix, p=2, dim=1)
        return demix


class Flatten(nn.Module):
    """
    Flatten layer to connect feature extraction and classification parts of a network.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    forward(x)
        Flattens the input tensor.
    """

    def forward(self, x):
        """Flattens the input tensor."""
        return x.view(x.size(0), -1)


class DepthwiseConv2d(nn.Module):
    """
    Depthwise separable convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    kernel_size : int or tuple
        Size of the convolutional kernel.
    depthwise_multiplier : int, optional
        Multiplier for depthwise convolution. Defaults to 1.

    Attributes
    ----------
    depthwise : nn.Conv2d
        Depthwise convolutional layer.

    Methods
    -------
    forward(x)
        Applies depthwise convolution to the input tensor.
    """

    def __init__(self, in_channels, kernel_size, depthwise_multiplier=1, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depthwise_multiplier,
            kernel_size,
            groups=in_channels,
            **kwargs,
        )

    def forward(self, x):
        """Applies depthwise convolution to the input tensor."""
        return self.depthwise(x)


class SeparableConv2d(nn.Module):
    """
    Separable convolutional layer (depthwise + pointwise convolution).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolutional kernel.

    Attributes
    ----------
    depthwise : DepthwiseConv2d
        Depthwise convolutional layer.
    pointwise : nn.Conv2d
        Pointwise convolutional layer.

    Methods
    -------
    forward(x)
        Applies separable convolution to the input tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        """Applies separable convolution to the input tensor."""
        return self.pointwise(self.depthwise(x))
