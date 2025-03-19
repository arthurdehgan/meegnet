import os
from collections import defaultdict, OrderedDict
import logging
from typing import Tuple
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from meegnet.layer import Flatten, DepthwiseConv2d, SeparableConv2d, Conv2dWithConstraint

LOG = logging.getLogger("meegnet")


def create_net(
    net_option: str, input_size: int, n_outputs: int, net_params: dict = None
) -> nn.Module:
    """
    Creates a neural network based on the specified option.

    Parameters
    ----------
    net_option : str
        Network architecture option ("MLP", "meegnet", "custom", "VGG", "EEGNet", "vanPutNet").
    input_size : int
        Input size of the network.
    n_outputs : int
        Number of output neurons.
    net_params : dict, optional
        Network parameters (required for "MLP" and "custom" options).

    Returns
    -------
    nn.Module
        Created neural network.

    Raises
    ------
    AttributeError
        If net_option is invalid.
    """

    if net_params is None:
        net_params = {
            "linear": 100,
            "hlayers": 2,
            "dropout": 0.5,
        }

    net_options = {
        "mlp": lambda: MLP(
            input_size=input_size,
            n_outputs=n_outputs,
            hparams={
                "mlp_width": net_params["linear"],
                "mlp_depth": net_params["hlayers"],
                "mlp_dropout": net_params["dropout"],
            },
        ),
        "meegnet": lambda: MEEGNet(input_size, n_outputs),
        "custom": lambda: FullNet(
            input_size,
            n_outputs,
            net_params["hlayers"],
            net_params["filters"],
            net_params["nchan"],
            net_params["linear"],
            net_params["dropout"],
            net_params["batchnorm"],
            net_params["maxpool"],
        ),
        "vgg": lambda: VGG16(input_size, n_outputs),
        "eegnet": lambda: EEGNet(input_size, n_outputs),
        # "eegnet": lambda: EEGNetv4(input_size, n_outputs),
        "vanputnet": lambda: VanPutNet(input_size, n_outputs),
    }

    if net_option.lower() not in net_options:
        raise AttributeError(f"Invalid network option: {net_option}")

    if net_option.lower() in ["mlp", "custom"] and net_params is None:
        raise ValueError("net_params is required for MLP and custom networks")

    return net_options[net_option.lower()]()


class CustomNet(nn.Module):
    """
    Base class for custom neural networks.

    Parameters
    ----------
    input_size : tuple
        Input shape of the network.
    n_outputs : int
        Number of output neurons.

    Attributes
    ----------
    input_size : tuple
        Input shape of the network.
    n_outputs : int
        Number of output neurons.

    Methods
    -------
    forward(x)
        Defines the forward pass through the network.
    get_output_shape(layers)
        Computes the output shape of a sequence of layers.
    feature_extraction(x)
        Defines the feature extraction part of the network (must be implemented).
    classif(x)
        Defines the classification part of the network (must be implemented).
    """

    def __init__(self, input_size: tuple, n_outputs: int) -> None:
        """
        Initializes the CustomNet.

        Args:
        input_size (tuple): Input shape of the network.
        n_outputs (int): Number of output neurons.
        """
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the network.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        feats = self.feature_extraction(x)
        outs = self.classif(feats)
        return outs

    def get_output_shape(self, layers: nn.Sequential = None) -> tuple:
        if layers is None:
            layers = self.feature_extraction
        return layers(torch.zeros((1, *self.input_size))).shape

    def get_lin_size(self, layers: nn.Sequential) -> int:
        """
        Computes the output size of a sequence of layers.

        Args:
        layers (nn.Sequential): Sequence of layers.

        Returns:
        int: Output size of the sequence.
        """
        return self.get_output_shape(layers)[-1]


class EEGNetv4(CustomNet):
    def __init__(
        self,
        input_size: tuple,
        n_outputs: int,
        final_conv_length: str | int = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool1_stride_size: int = 4,
        pool2_kernel_size: int = 8,
        pool2_stride_size: int = 8,
        conv_spatial_max_norm: float = 1,
        activation: nn.Module = nn.ELU,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        drop_prob: float = 0.25,
    ) -> None:
        super().__init__(input_size, n_outputs)

        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2 if F2 is not None else self.F1 * D
        self.kernel_length = kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.pool1_kernel_size = pool1_kernel_size
        self.pool1_stride_size = pool1_stride_size
        self.pool2_kernel_size = pool2_kernel_size
        self.pool2_stride_size = pool2_stride_size
        self.conv_spatial_max_norm = conv_spatial_max_norm
        self.activation = activation
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_eps = batch_norm_eps
        self.drop_prob = drop_prob

        pool_class = {"mean": nn.AvgPool2d, "max": nn.MaxPool2d}

        self.feature_extraction = nn.Sequential(
            # Rearrange("batch ch t -> batch 1 ch t"),
            nn.Conv2d(
                input_size[0],
                F1,
                (1, kernel_length),
                stride=1,
                bias=False,
                padding=(0, kernel_length // 2),
            ),
            nn.BatchNorm2d(
                self.F1,
                momentum=batch_norm_momentum,
                affine=batch_norm_affine,
                eps=batch_norm_eps,
            ),
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (input_size[0], 1),
                max_norm=conv_spatial_max_norm,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
            nn.BatchNorm2d(
                self.F1 * self.D,
                momentum=batch_norm_momentum,
                affine=batch_norm_affine,
                eps=batch_norm_eps,
            ),
            activation(),
            pool_class[pool_mode](
                kernel_size=(1, pool1_kernel_size),
                stride=(1, pool1_stride_size),
            ),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, depthwise_kernel_length),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, depthwise_kernel_length // 2),
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
            nn.BatchNorm2d(
                self.F2,
                momentum=batch_norm_momentum,
                affine=batch_norm_affine,
                eps=batch_norm_eps,
            ),
            activation(),
            pool_class[pool_mode](
                kernel_size=(1, pool2_kernel_size),
                stride=(1, pool2_stride_size),
            ),
            nn.Dropout(p=drop_prob),
        )

        output_shape = self.get_output_shape()
        n_out_virtual_chans = output_shape[2]

        if self.final_conv_length == "auto":
            n_out_time = output_shape[3]
            self.final_conv_length = n_out_time

        self.classif = nn.Sequential(
            nn.Conv2d(
                self.F2,
                n_outputs,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
            # Rearrange("batch x y z -> batch x z y"),
        )


class EEGNet(CustomNet):
    """
    EEGNet architecture implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape (S, C, T) where S is the number of sensors.
    n_outputs : int
        Number of output classes.
    filter_size : int, optional
        Filter size. Defaults to 64.
    n_filters : int, optional
        Number of filters. Defaults to 16.
    dropout : float, optional
        Dropout rate. Defaults to 0.5.
    dropout_option : str, optional
        Dropout type ("SpatialDropout2D" or "Dropout"). Defaults to "Dropout".
    depthwise_multiplier : int, optional
        Depthwise multiplier. Defaults to 2.

    Attributes
    ----------
    feature_extraction : nn.Sequential
        Feature extraction layers.
    classif : nn.Sequential
        Classification layers.
    """

    def __init__(
        self,
        input_size: tuple,
        n_outputs: int,
        filter_size: int = 64,
        n_filters: int = 16,
        dropout: float = 0.5,
        dropout_option: str = "Dropout",
        depthwise_multiplier: int = 2,
    ) -> None:
        super().__init__(input_size, n_outputs)

        if dropout_option not in ["SpatialDropout2D", "Dropout"]:
            raise ValueError(
                "dropout_option must be one of SpatialDropout2D "
                "or Dropout, passed as a string."
            )
        dropout_type = nn.Dropout2d if dropout_option == "SpatialDropout2D" else nn.Dropout

        n_channels = input_size[1]
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                input_size[0],
                n_filters,
                (1, filter_size),
                padding=(1, filter_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(n_filters),
            DepthwiseConv2d(
                n_filters,
                (n_channels, 1),
                depthwise_multiplier=depthwise_multiplier,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(n_filters * depthwise_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropout_type(dropout),
            SeparableConv2d(
                n_filters * depthwise_multiplier,
                n_filters * depthwise_multiplier,
                (1, 16),
                padding=(1, 8),
                bias=False,
            ),
            nn.BatchNorm2d(n_filters * depthwise_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropout_type(dropout),
            Flatten(),
        )

        lin_size = self.get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(nn.Linear(lin_size, n_outputs))


# This implementation is rather common and found on various blogs/github repos
class VGG16(CustomNet):
    """
    VGG16 architecture implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape (C, H, W) or (S, C, T) for M/EEG data. With S the number of sensors,
        C the number of channels and T the number of time samples
    n_outputs : int
        Number of output classes.
    """

    def __init__(self, input_size: tuple, n_outputs: int) -> None:
        super().__init__(input_size, n_outputs)

        in_channels = input_size[0]
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
        )

        lin_size = self.get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(
            nn.Linear(lin_size, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, n_outputs),
        )


class MLP(CustomNet):
    """
    Multi-Layer Perceptron (MLP) implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape.
    n_outputs : int
        Number of output neurons.
    hparams : dict
        Hyperparameters:
            - mlp_width (int): Number of neurons in each hidden layer.
            - mlp_depth (int): Number of hidden layers.
            - mlp_dropout (float): Dropout rate.
    """

    def __init__(self, input_size: tuple, n_outputs: int, hparams: dict) -> None:
        super().__init__(input_size, n_outputs)
        n_inputs = np.prod(input_size)

        self.feature_extraction = nn.Sequential(
            Flatten(),
            nn.Linear(n_inputs, hparams["mlp_width"]),
            nn.Dropout(hparams["mlp_dropout"]),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hparams["mlp_width"], hparams["mlp_width"]),
                    nn.Dropout(hparams["mlp_dropout"]),
                    nn.ReLU(),
                )
                for _ in range(hparams["mlp_depth"] - 2)
            ],
        )

        self.classif = nn.Linear(hparams["mlp_width"], n_outputs)


class MEEGNet(CustomNet):
    """
    MEEGNet architecture implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape (S, C, T) where S is the number of sensors.
    n_outputs : int
        Number of output classes.
    n_linear : int, optional
        Number of neurons in the first linear layer. Defaults to 2000.
    dropout : float, optional
        Dropout rate. Defaults to 0.5.
    """

    def __init__(
        self,
        input_size: tuple,
        n_outputs: int,
        n_linear: int = 2000,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(input_size, n_outputs)

        maxpool = nn.MaxPool2d(kernel_size=(1, 20), stride=1)

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_size[0], 100, (input_size[1], 1)),
            nn.ReLU(),
            nn.Conv2d(100, 200, (1, 9)),
            maxpool,
            nn.ReLU(),
            nn.Conv2d(200, 200, (1, 9)),
            maxpool,
            nn.ReLU(),
            nn.Conv2d(200, 100, (1, 9)),
            maxpool,
            nn.ReLU(),
            Flatten(),
            nn.Dropout(dropout),
        )

        lin_size = self.get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(
            nn.Linear(lin_size, n_linear // 2),
            nn.Linear(n_linear // 2, n_outputs),
        )


class FullNet(CustomNet):
    """
    FullNet architecture implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape (S, C, T) where S is the number of sensors.
    n_outputs : int
        Number of output classes.
    hlayers : int, optional
        Number of hidden layers. Defaults to 2.
    filter_size : int, optional
        Filter size. Defaults to 7.
    nchan : int, optional
        Number of channels. Defaults to 5.
    n_linear : int, optional
        Number of neurons in the first linear layer. Defaults to 150.
    dropout : float, optional
        Dropout rate. Defaults to 0.25.
    batchnorm : bool, optional
        Whether to use batch normalization. Defaults to False.
    maxpool : int, optional
        Max pooling size. Defaults to 0.
    """

    def __init__(
        self,
        input_size: tuple,
        n_outputs: int,
        hlayers: int = 2,
        filter_size: int = 7,
        nchan: int = 5,
        n_linear: int = 150,
        dropout: float = 0.25,
        batchnorm: bool = False,
        maxpool: int = 0,
    ) -> None:
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_size[0], nchan, (input_size[1], 1)),
            nn.ReLU(),
            *self._build_conv_layers(hlayers, filter_size, nchan, batchnorm, maxpool),
            Flatten(),
            nn.Dropout(dropout),
        )

        lin_size = self.get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(
            nn.Linear(lin_size, n_linear // 2),
            nn.Linear(n_linear // 2, n_outputs),
        )

    def _build_conv_layers(self, hlayers, filter_size, nchan, batchnorm, maxpool):
        layers = []
        prev = nchan

        # Encoder
        for i in range(hlayers // 2):
            nex = prev * 2
            layers += [
                nn.Conv2d(prev, nex, (1, filter_size)),
                nn.ReLU(),
                *([nn.BatchNorm2d(nex)] if batchnorm else []),
                *([nn.MaxPool2d((1, maxpool), 1)] if maxpool != 0 else []),
            ]
            prev = nex

        # Middle layer (if odd number of hidden layers)
        if hlayers % 2 != 0:
            layers += [
                nn.Conv2d(prev, prev, (1, filter_size)),
                nn.ReLU(),
                *([nn.BatchNorm2d(prev)] if batchnorm else []),
                *([nn.MaxPool2d((1, maxpool), 1)] if maxpool != 0 else []),
            ]

        # Decoder
        for i in range(hlayers // 2):
            nex = prev // 2
            layers += [
                nn.Conv2d(prev, nex, (1, filter_size)),
                nn.ReLU(),
                *([nn.BatchNorm2d(nex)] if batchnorm else []),
                *([nn.MaxPool2d((1, maxpool), 1)] if maxpool != 0 else []),
            ]
            prev = nex

        return layers


class VanPutNet(CustomNet):
    """
    VanPutNet architecture implementation.

    Parameters
    ----------
    input_size : tuple
        Input shape (C, H, W) or (S, C, T) for M/EEG data.
    n_output : int
        Number of output classes.
    dropout : float, optional
        Dropout rate. Defaults to 0.25.
    """

    def __init__(self, input_size: tuple, n_output: int, dropout: float = 0.25) -> None:
        super().__init__(input_size, n_output)

        in_channels = input_size[0]
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, 100, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(100, 100, 3),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(100, 300, (2, 3)),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(300, 300, (1, 7)),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(300, 100, (1, 3)),
            nn.Conv2d(100, 100, (1, 3)),
            Flatten(),
        )

        lin_size = self.get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(
            nn.Linear(lin_size, n_output),
            nn.Softmax(dim=1),
        )


class AutoEncoder(CustomNet):
    def __init__(
        self,
        input_size,
    ):
        CustomNet.__init__(self, input_size)

        lin_size = input_size[0] * input_size[1] * input_size[2]

        self.encoder = nn.Sequential(
            nn.Linear(lin_size, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, lin_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class Model:
    def __init__(
        self,
        name: str,
        net_option: str,
        input_size: tuple,
        n_outputs: int,
        save_path: str = None,
        learning_rate: float = 0.00001,
        optimizer: callable = optim.Adam,
        criterion: callable = nn.CrossEntropyLoss(),
        n_folds: int = 5,
        device: str = "cuda",
        net_params: dict = None,
    ) -> None:
        """
        Initialize the Model class.

        Parameters
        ----------
        name : str
            Model name.
        net_option : str
            Network architecture option.
        input_size : tuple
            Input shape (C, H, W) or (S, C, T) for EEG data.
        n_outputs : int
            Number of output classes.
        save_path : str, optional
            Model save path. Defaults to None.
        learning_rate : float, optional
            Learning rate. Defaults to 0.00001.
        optimizer : callable, optional
            Optimizer function. Defaults to Adam.
        criterion : callable, optional
            Loss function. Defaults to CrossEntropyLoss.
        n_folds : int, optional
            Number of folds for cross-validation. Defaults to 5.
        device : str, optional
            Device to use (cuda or cpu). Defaults to "cuda".
        net_params : dict, optional
            Network architecture parameters. Defaults to None.
        """
        assert isinstance(name, str), "Name must be a string"
        assert isinstance(input_size, tuple), "Input size must be a tuple"
        assert len(input_size) == 3, "Input size must have 3 dimensions"

        self.name = name
        self.input_size = input_size  # TODO here put assertions on the shape
        self.net = create_net(net_option, input_size, n_outputs, net_params)
        self.n_outputs = n_outputs
        self.n_folds = n_folds
        self.criterion = criterion
        self.save_path = save_path
        self.lr = learning_rate
        self.optimizer = optimizer(self.net.parameters(), lr=learning_rate)
        self.tracker = TrainingTracker(self.save_path, self.name)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif device == "cuda":
            LOG.warning("Warning: gpu device requested but unavailable. Setting device to CPU")
            self.device = "cpu"
        else:
            self.device = "cpu"
        self.net.to(self.device)

    def train(
        self,
        dataset,
        batch_size: int = 128,
        patience: int = 10,
        max_epoch: int = None,
        model_path: str = None,
        early_stop: str = "loss",
        num_workers: int = 4,
        continue_training: bool = False,
        verbose: int = 3,
    ) -> None:
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        dataset
            Dataset to train on.
        batch_size : int, optional
            Batch size for training. Defaults to 128.
        patience : int, optional
            Patience for early stopping. Defaults to 10.
        max_epoch : int, optional
            Maximum number of epochs to train. Defaults to None.
        model_path : str, optional
            Path to save the model. Defaults to None.
        num_workers : int, optional
            Number of workers for data loading. Defaults to 4.
        continue_training : bool, optional
            Wether to pick up training from last checkpoint. By default False. Use when
            training stopped abbruptly because of an error, or to re-train or fine-tune
            the model.
        verbose : int, optional
            The verbosity level. By default 3.
            Values under 3 will display less detailed progress during training.
            Values under 2 will not display any update on performance epoch per epoch during training.

        Notes
        -----
        This method trains the model using the provided dataset and hyperparameters.
        It uses early stopping based on the validation loss and saves the model
        periodically.
        """
        assert len(dataset.data) > 0, "Dataset is empty."
        # Check dataset compatibility
        assert (
            dataset.data[0].shape == self.input_size
        ), "Dataset sample size must match network input size."
        assert early_stop in (
            "loss",
            "accuracy",
        ), f"{early_stop} is not a valid early_stop option."

        # Setting model_path
        self.tracker.set_model_path(model_path)

        # Set training mode and batch size
        self.net.train()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create data loaders
        LOG.info("Creating DataLoaders...")
        train_index, valid_index, _ = dataset.split_data()
        trainloader = DataLoader(
            dataset.torchDataset(train_index),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        validloader = DataLoader(
            dataset.torchDataset(valid_index),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        # Log training configuration
        LOG.info("Starting Training with:")
        LOG.info(f"Batch size: {batch_size}")
        LOG.info(f"Learning rate: {self.lr}")
        LOG.info(f"Patience: {patience}")
        if max_epoch is not None:
            LOG.info(f"Maximum Epoch: {max_epoch}")

        epoch = 0
        if continue_training:
            epoch = self.tracker.best["epoch"] + 1 
            self.tracker.patience_state = 0

        while self.tracker.patience_state < patience and (
            max_epoch is None or epoch < max_epoch
        ):
            self.train_epoch(epoch, trainloader, verbose=verbose)
            train_loss, train_acc = self.evaluate(trainloader)
            valid_loss, valid_acc = self.evaluate(validloader)
            self.tracker.update(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                self.net,
                self.optimizer,
                early_stop=early_stop,
            )
            if verbose >= 2:
                LOG.info(f"Epoch: {epoch}")
                LOG.info(f" [LOSS] TRAIN {train_loss:.4f} / VALID {valid_loss:.4f}")
                LOG.info(f" [ACC] TRAIN {100*train_acc:.2f}% / VALID {100*valid_acc:.2f}%")
            epoch += 1

    def fit(self, *args, **kwargs):
        return self.train(args, *kwargs)

    def train_batch(self, batch):
        self.optimizer.zero_grad()
        if len(batch) > 2:
            X, y, groups = batch
        else:
            X, y = batch
        y = y.view(-1)
        targets = Variable(y.type(torch.LongTensor)).to(self.device)
        X = X.view(-1, *self.input_size).to(self.device)
        loss = self.criterion(self.net.forward(X), targets)
        loss = loss
        loss.backward()
        self.optimizer.step()
        return loss

    def display_progress(self, i, epoch, loss, n_batches, verbose=3):
        progress = f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {loss:.5f}"
        if n_batches > 10:
            if i % (n_batches // 10) == 0 and verbose > 2:
                LOG.info(progress)
        elif verbose > 2:
            LOG.info(progress)

    def train_epoch(self, epoch, trainloader, verbose):
        # Train loop for a single epoch
        n_batches = len(trainloader)
        for i, batch in enumerate(trainloader):
            loss = self.train_batch(batch)
            self.display_progress(i, epoch, loss, n_batches, verbose=verbose)

    def evaluate(self, dataloader):
        with torch.no_grad():
            losses = 0
            accuracy = 0
            counter = 0
            for batch in dataloader:
                if len(batch) > 2:
                    X, y, _ = batch
                else:
                    X, y = batch
                y = y.view(-1)
                targets = Variable(y.type(torch.LongTensor)).to(self.device)
                X = X.view(-1, *self.input_size).to(self.device)
                out = self.net.forward(X)
                loss = self.criterion(out, targets)
                acc = self.compute_accuracy(out, targets)
                n = y.size(0)
                losses += loss.detach().sum().data.cpu().numpy() * n
                accuracy += acc.sum().data.cpu().numpy() * n
                counter += n
            floss = losses / float(counter)
            faccuracy = accuracy / float(counter)
            return floss, faccuracy

    def test(self, dataset):
        _, _, test_index = dataset.split_data()
        test_loader = DataLoader(
            dataset.torchDataset(test_index),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        test_loss, test_acc = self.evaluate(test_loader)
        LOG.info(f" [LOSS] TEST {test_loss}")
        LOG.info(f" [ACC] TEST {test_acc}")
        return test_loss, test_acc

    def n_parameters(self, model: nn.Module = None):
        if model is None:
            model = self.net
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def _get_from_hub(self, repo=None):
        if repo is None:
            repo = "lamaroufle/meegnet"
        model_name = "_".join(self.name.split("_")[:-2])
        input_size_string = "_".join(map(str, self.input_size))
        filename = f"{model_name}_{input_size_string}_{self.n_outputs}"
        model_path = hf_hub_download(repo_id="lamaroufle/meegnet", filename=filename + ".pt")
        hf_hub_download(repo_id="lamaroufle/meegnet", filename=filename + ".mat")
        return model_path

    def from_pretrained(self, repo=None):
        model_path = self._get_from_hub(repo)
        self.load(model_path)

    def _load_net(self, model_path: str = None) -> Tuple:
        """Load network state and optimizer state from file."""
        if model_path is None:
            model_path = self.tracker.model_path
        if os.path.exists(model_path):
            LOG.info("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            net_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            # net_state = checkpoint["state_dict"]()
            # optimizer_state = checkpoint["optimizer"].state_dict()
        mat_path = model_path[:-2] + "mat"
        if os.path.exists(mat_path):
            self.tracker.load(mat_path)
        else:
            LOG.warning(f"Error while loading checkpoint from {model_path}")
        return net_state, optimizer_state

    def load(self, model_path=None):
        """Load model from file."""
        try:
            net_state, optimizer_state = self._load_net(model_path)
            if net_state[list(net_state.keys())[-1]].shape[0] == self.n_outputs:
                self.net.load_state_dict(net_state)
                self.optimizer.load_state_dict(optimizer_state)
            else:
                feat_state_dict = OrderedDict()
                for key, value in net_state.items():
                    if key.startswith("feature"):
                        feat_state_dict[".".join(key.split(".")[1:])] = value
                self.net.feature_extraction.load_state_dict(feat_state_dict)
                self.tracker = TrainingTracker(self.save_path, self.name)
        except FileNotFoundError:
            LOG.error(f"Model file not found: {model_path}")

    def compute_accuracy(self, y_pred, target):
        # Compute accuracy from 2 vectors of labels.
        correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
        return correct / len(target)

    def get_feature_weights(self):
        weights = []
        for layer in self.net.feature_extraction:
            if hasattr(layer, "weight"):
                weights.append(layer.weight.cpu().detach().numpy())
        return weights

    def get_clf_weights(self):
        weights = []
        for layer in self.net.classif:
            if hasattr(layer, "weight"):
                weights.append(layer.weight.detach().numpy())
        return weights

    def plot_accuracy(self, option="both"):
        return self.tracker.plot_accuracy(option)

    def plot_loss(self, option="both"):
        return self.tracker.plot_loss(option)

    def save(self, model_path: str = None):
        self.tracker.set_model_path(model_path)
        checkpoint = {
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self.tracker.save(checkpoint)


class TrainingTracker:
    def __init__(self, save_path, name, model_path: str = None):
        self.progress = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }
        self.best = {
            "train_loss": float("inf"),
            "train_accuracy": 0,
            "validation_loss": float("inf"),
            "validation_accuracy": 0,
            "epoch": 0,
        }
        self.patience_state = 0
        self.save_path = save_path
        self.name = name

        self.set_model_path(model_path)

    def set_model_path(self, model_path: str = None) -> None:
        self.model_path = (
            os.path.join(self.save_path, self.name + ".pt")
            if model_path is None
            else model_path
        )

    def update(
        self, epoch, tloss, tacc, vloss, vacc, net, optimizer, early_stop: str = "loss"
    ) -> None:
        assert early_stop in (
            "loss",
            "accuracy",
        ), f"{early_stop} is not a valid early_stop option."

        if epoch < len(self.progress['train_loss']):
            self.progress["train_loss"][epoch] = tloss
            self.progress["train_accuracy"][epoch] = tacc
            self.progress["validation_loss"][epoch] = vloss
            self.progress["validation_accuracy"][epoch] = vacc
            self.patience_state += 1
        else:
            self.progress["train_loss"].append(tloss)
            self.progress["train_accuracy"].append(tacc)
            self.progress["validation_loss"].append(vloss)
            self.progress["validation_accuracy"].append(vacc)
            self.patience_state += 1

        check = {
            "loss": vloss < self.best["validation_loss"],
            "accuracy": vacc > self.best["validation_accuracy"],
        }
        if check[early_stop]:
            self.best["train_loss"] = tloss
            self.best["train_accuracy"] = tacc
            self.best["validation_loss"] = vloss
            self.best["validation_accuracy"] = vacc
            self.best["epoch"] = epoch
            self.patience_state = 0
            checkpoint = {"state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
            self.save(checkpoint)

    def save(self, checkpoint) -> None:
        """Save model to file."""
        mat_path = self.model_path[:-2] + "mat"
        try:
            torch.save(checkpoint, self.model_path)
            save_dict = {key: value for key, value in self.progress.items()}
            save_dict.update({key: value for key, value in self.best.items()})
            savemat(mat_path, save_dict)
        except OSError:
            LOG.error(f"Error saving model to file: {self.model_path}")

    def load(self, mat_path):
        data = loadmat(mat_path)
        for key, value in data.items():
            if key in self.progress.keys():

                self.progress[key] = value
            elif key in self.best.keys():
                self.best[key] = value

    def plot_metric(self, metric_type: str, option: str = "both", early_stop: bool = True):
        assert option in ["both", "train", "valid"]
        assert metric_type in ["accuracy", "loss"]

        fig, ax = plt.subplots()

        if option in ("both", "train"):
            plt.plot(
                np.array(self.progress[f"train_{metric_type}"]).squeeze(),
                label=f"Training {metric_type.capitalize()}",
            )
        if option in ("both", "valid"):
            plt.plot(
                np.array(self.progress[f"validation_{metric_type}"]).squeeze(),
                label=f"Validation {metric_type.capitalize()}",
            )

        epochs = len(self.progress[f"train_{metric_type}"])
        step = max(1, epochs // 10)  # Ensure at most 10 ticks
        ticks = list(range(0, epochs, step))  # Use epoch indices for ticks

        if early_stop:
            plt.axvline(x=self.best["epoch"] , label="early stop", color="green")
            if self.best["epoch"] not in ticks:
                ticks.append(self.best["epoch"] )

        ax.set_xticks(sorted(ticks))
        ax.set_xticklabels([x + 1 for x in sorted(ticks)])  # Labels should start from 1
        ax.set_ylabel(metric_type.capitalize())
        ax.set_xlabel("Epoch")
        plt.legend()
        plt.plot()
        return fig

    def plot_accuracy(self, option: str = "both", early_stop: bool = True):
        return self.plot_metric(metric_type="accuracy", option=option, early_stop=early_stop)

    def plot_loss(self, option: str = "both", early_stop: bool = True):
        return self.plot_metric(metric_type="loss", option=option, early_stop=early_stop)
