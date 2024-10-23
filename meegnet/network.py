import os
from collections import defaultdict, OrderedDict
import logging
from typing import Tuple
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import numpy as np
from huggingface_hub import hf_hub_download

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
        "vanputnet": lambda: VanPutNet(input_size, n_outputs),
    }

    if net_option.lower() not in net_options:
        raise AttributeError(f"Invalid network option: {net_option}")

    if net_option.lower() in ["mlp", "custom"] and net_params is None:
        raise ValueError("net_params is required for MLP and custom networks")

    return net_options[net_option.lower()]()


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
    _get_lin_size(layers)
        Computes the output size of a sequence of layers.
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
        super(CustomNet, self).__init__()
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

    def _get_lin_size(self, layers: nn.Sequential) -> int:
        """
        Computes the output size of a sequence of layers.

        Args:
        layers (nn.Sequential): Sequence of layers.

        Returns:
        int: Output size of the sequence.
        """
        return layers(torch.zeros((1, *self.input_size))).shape[-1]


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
        layer_list = [
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
        ]

        self.feature_extraction = nn.Sequential(*layer_list)

        lin_size = self._get_lin_size(self.feature_extraction)
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
        super(VGG16, self).__init__(input_size, n_outputs)

        in_channels = input_size[0]
        layer_list = [
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
        ]
        layers = nn.ModuleList(layer_list)
        self.feature_extraction = nn.Sequential(*layers)

        lin_size = self._get_lin_size(layers)
        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(lin_size, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, n_outputs),
                ]
            )
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
        super(MLP, self).__init__(input_size, n_outputs)
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
        super(MEEGNet, self).__init__(input_size, n_outputs)

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

        lin_size = self._get_lin_size(self.feature_extraction)
        self.classif = nn.Sequential(
            nn.Linear(lin_size, n_linear // 2),
            nn.Linear(n_linear // 2, n_outputs),
        )


class FullNet(nn.Module):
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
        super(FullNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_size[0], nchan, (input_size[1], 1)),
            nn.ReLU(),
            *self._build_conv_layers(hlayers, filter_size, nchan, batchnorm, maxpool),
            Flatten(),
            nn.Dropout(dropout),
        )

        lin_size = self._get_lin_size(self.feature_extraction)
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
        super(VanPutNet, self).__init__(input_size, n_output)

        conv_layers = nn.Sequential(
            nn.Conv2d(1, 100, 3),
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

        lin_size = self._get_lin_size(conv_layers)
        self.model = nn.Sequential(
            conv_layers,
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
    """
    A class representing a neural network model.

    Attributes
    ----------
    name : str
        Model name.
    net : nn.Module
        Neural network architecture.
    input_size : tuple
        Input shape (C, H, W) or (S, C, T) for EEG data.
    n_outputs : int
        Number of output classes.
    save_path : str
        Model save path.
    lr : float
        Learning rate.
    optimizer : callable
        Optimizer function.
    criterion : callable
        Loss function.
    n_folds : int
        Number of folds for cross-validation.
    device : str
        Device to use (cuda or cpu).
    tracker : TrainingTracker
        Training progress tracker.

    Methods
    -------
    train(dataset, batch_size=128, patience=20, max_epoch=None, model_path=None, num_workers=4)
        Train the model on the provided dataset.
    fit(*args, **kwargs)
        Alias for train method.
    train_batch(batch)
        Train on a single batch.
    display_progress(i, epoch, loss, n_batches)
        Display training progress.
    train_epoch(epoch, trainloader)
        Train loop for a single epoch.
    evaluate(dataloader)
        Evaluate model on a dataloader.
    test(dataset)
        Evaluate model on a test dataset.
    from_huggingface(repo=None)
        Load pre-trained model from repository.
    load(model_path=None)
        Load model from file.
    compute_accuracy(y_pred, target)
        Compute accuracy from predictions and targets.
    get_feature_weights()
        Get feature extraction weights.
    get_clf_weights()
        Get classification weights.
    """

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
        Initializes the Model class.

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

        Raises
        ------
        AssertionError
            If input size is not a tuple or does not have 3 dimensions.

        Notes
        -----
        This method initializes the model with the provided parameters and sets up the
        neural network architecture, optimizer, and loss function.
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
        self.checkpoint = defaultdict(lambda: 0)
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
        patience: int = 20,
        max_epoch: int = None,
        model_path: str = None,
        num_workers: int = 4,
    ) -> None:
        """
        Trains the model on the provided dataset.

        Parameters
        ----------
        dataset
            Dataset to train on.
        batch_size : int, optional
            Batch size for training. Defaults to 128.
        patience : int, optional
            Number of epochs without improvement before early stopping. Defaults to 20.
        max_epoch : int, optional
            Maximum number of epochs to train. Defaults to None.
        model_path : str, optional
            Path to save the model. Defaults to None.
        num_workers : int, optional
            Number of workers for data loading. Defaults to 4.

        Raises
        ------
        AssertionError
            If dataset sample dimensions does not match network input size.

        Notes
        -----
        This method trains the model using the provided dataset and hyperparameters.
        It uses early stopping based on the validation loss and saves the model periodically.
        The training process is logged to the console.

        Returns
        -------
        None
        """
        assert (
            dataset.data[0].shape == self.input_size
        ), "Dataset sample size must match network input size."

        # Set training mode and batch size
        self.net.train()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create data loaders
        LOG.info("Creating DataLoaders...")
        train_index, valid_index, _ = dataset.data_split(0.8, 0.1, 0.1)
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

        epoch = 1
        while self.tracker.patience_state < patience and (
            max_epoch is None or epoch <= max_epoch
        ):
            self.train_epoch(epoch, trainloader)
            train_loss, train_acc = self.evaluate(trainloader)
            valid_loss, valid_acc = self.evaluate(validloader)
            self.tracker.update(
                epoch, train_loss, valid_loss, train_acc, valid_acc, self.net, self.optimizer
            )
            LOG.info(f"Epoch: {epoch}")
            LOG.info(f" [LOSS] TRAIN {train_loss} / VALID {valid_loss}")
            LOG.info(f" [ACC] TRAIN {train_acc} / VALID {valid_acc}")
            epoch += 1

    def fit(self, *args, **kwargs):
        """
        Alias for the train method, provided for compatibility purposes.

        Parameters
        ----------
        *args : list
            Variable length argument list passed to the train method.
        **kwargs : dict
            Keyword arguments passed to the train method.

        See Also
        --------
        train : The method that performs the actual training.
        """
        return self.train(args, *kwargs)

    def train_batch(self, batch):
        """
        Trains the model on a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data, containing input and target tensors.

        Returns
        -------
        loss : torch.Tensor
            Training loss for the batch.
        """
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

    def display_progress(self, i, epoch, loss, n_batches):
        """
        Displays training progress.

        Parameters
        ----------
        i : int
            Current batch index.
        epoch : int
            Current epoch number.
        loss : float
            Current batch loss.
        n_batches : int
            Total number of batches.

        Returns
        -------
        None

        Notes
        -----
        Logs progress at regular intervals, providing updates on epoch, batch, and loss.
        """
        progress = f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {loss:.5f}"
        if n_batches > 10:
            if i % (n_batches // 10) == 0:
                LOG.info(progress)
        else:
            LOG.info(progress)

    def train_epoch(self, epoch, trainloader):
        """
        Trains the model for a single epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        trainloader : DataLoader
            Training data loader.

        Returns
        -------
        None

        Notes
        -----
        Iterates over the training dataset, calling train_batch for each batch.
        Displays training progress using display_progress.
        """
        # Train loop for a single epoch
        n_batches = len(trainloader)
        for i, batch in enumerate(trainloader):
            loss = self.train_batch(batch)
            self.display_progress(i, epoch, loss, n_batches)

    def evaluate(self, dataloader):
        """
        Evaluates the model's performance on a given dataset.

        Parameters
        ----------
        dataloader : DataLoader
            Data loader for evaluation.

        Returns
        -------
        loss : float
            Average loss over the dataset.
        accuracy : float
            Average accuracy over the dataset.

        Notes
        -----
        Disables gradient computation during evaluation.
        Computes loss and accuracy for each batch and returns averages.
        """
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
        """
        Evaluates the model's performance on a test dataset.

        Parameters
        ----------
        dataset
            The complete datset, data_split will be performed according to the parameters set in the dataset object.

        Returns
        -------
        loss : float
            Average loss over the test dataset.
        accuracy : float
            Average accuracy over the test dataset.

        Notes
        -----
        Splits the dataset into test batches and evaluates the model using evaluate.
        Logs the test loss and accuracy.
        """
        _, _, test_index = dataset.data_split(0.8, 0.1, 0.1)
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

    def _get_from_hub(self, repo=None):
        """
        Retrieves a pre-trained model from the Hugging Face Model Hub.

        Parameters
        ----------
        repo : str, optional
            Repository name on the Hugging Face Model Hub. Defaults to "lamaroufle/meegnet".

        Returns
        -------
        model_path : str
            Path to the downloaded model file.

        Notes
        -----
        Downloads the model and its corresponding metadata file.
        Uses the model name, input size, and number of outputs to construct the file name.
        """
        if repo is None:
            repo = "lamaroufle/meegnet"
        model_name = "_".join(self.name.split("_")[:-2])
        input_size_string = "_".join(map(str, self.input_size))
        filename = f"{model_name}_{input_size_string}_{self.n_outputs}"
        model_path = hf_hub_download(repo_id="lamaroufle/meegnet", filename=filename + ".pt")
        hf_hub_download(repo_id="lamaroufle/meegnet", filename=filename + ".mat")
        return model_path

    def from_huggingface(self, repo=None):
        """
        Loads a pre-trained model from the Hugging Face Model Hub.

        Parameters
        ----------
        repo : str, optional
            Repository name on the Hugging Face Model Hub. Defaults to "lamaroufle/meegnet".

        Returns
        -------
        None

        Notes
        -----
        Downloads the pre-trained model using _get_from_hub and loads its weights.
        Calls load to initialize the model with the downloaded weights.
        """
        model_path = self._get_from_hub(repo)
        self.load(model_path)

    def _load_net(self, model_path: str = None) -> Tuple:
        """
        Loads a pre-trained neural network model from a file.

        Parameters
        ----------
        model_path : str
            Path to the model file (.pth, .pt, or .ckpt).

        Returns
        -------
        None

        Notes
        -----
        Loads the model architecture and weights from the file using PyTorch.
        Initializes the network with the loaded weights.
        """
        if model_path is None:
            model_path = os.path.join(self.save_path, self.name + ".pt")
        if os.path.exists(model_path):
            LOG.info("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            net_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
        mat_path = model_path[:-2] + "mat"
        if os.path.exists(mat_path):
            self.tracker.load(mat_path)
        else:
            LOG.warning(f"Error while loading checkpoint from {model_path}")
        return net_state, optimizer_state

    def load(self, model_path=None):
        """
        Loads a pre-trained model from a file.

        Parameters
        ----------
        model_path : str, optional
            Path to the model file (.pth, .pt, or .ckpt). Defaults to None.
        map_location : str, optional
            Device to load the model onto (e.g., "cuda", "cpu", "cuda:0"). Defaults to "cuda".

        Returns
        -------
        None

        Notes
        -----
        Supports loading models from PyTorch checkpoints and TorchScript models.
        Calls _load_net to load the model architecture and weights.
        Moves the model to the specified device.
        """
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
        """
        Computes the accuracy of the model's predictions.

        Parameters
        ----------
        outputs : torch.Tensor
            Model's output predictions (shape: [batch_size, num_classes]).
        targets : torch.Tensor
            Ground truth labels (shape: [batch_size]).

        Returns
        -------
        accuracy : float
            Accuracy of the model's predictions.
        """
        correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
        return correct / len(target)

    def get_feature_weights(self):
        """
        Retrieves the feature weights learned by the model.

        Returns
        -------
        weights : Dict[str, float]
            Dictionary with layer names as keys and weights as values.
        """
        weights = []
        for layer in self.net.feature_extraction:
            if hasattr(layer, "weight"):
                weights.append(layer.weight.cpu().detach().numpy())
        return weights

    def get_clf_weights(self):
        """
        Retrieves the classification weights learned by the model.

        Returns
        -------
        weights : Dict[str, float]
            Dictionary with layer names as keys and weights as values.
        """
        weights = []
        for layer in self.net.classif:
            if hasattr(layer, "weight"):
                weights.append(layer.weight.detach().numpy())
        return weights


class TrainingTracker:
    """
    Tracks and saves the progress of neural network training.

    Attributes
    ----------
    progress : dict
        Stores training and validation accuracy and loss histories.
    best : dict
        Stores the best validation accuracy, loss, and corresponding training metrics.
    patience_state : int
        Tracks the number of epochs without improvement.
    save_path : str
        Path to save model checkpoints.
    name : str
        Model name.

    Methods
    -------
    update(epoch, tloss, vloss, tacc, vacc, net, optimizer)
        Updates progress and saves model if validation accuracy improves.
    save(checkpoint, name, model_path=None)
        Saves model checkpoint to file.
    load(mat_path)
        Loads training progress from saved .mat file.
    """

    def __init__(self, save_path, name):
        """
        Initializes the TrainingTracker.

        Parameters
        ----------
        save_path : str
            Path to save model checkpoints.
        name : str
            Model name.
        """
        self.progress = {
            "validation_accuracies": [],
            "train_accuracies": [],
            "validation_losses": [],
            "train_losses": [],
        }
        self.best = {
            "validation_accuracy": 0,
            "train_accuracy": 0,
            "validation_loss": float("inf"),
            "train_loss": float("inf"),
            "epoch": 0,
        }
        self.patience_state = 0
        self.save_path = save_path
        self.name = name

    def update(self, epoch, tloss, vloss, tacc, vacc, net, optimizer):
        """
        Updates progress and saves model if validation accuracy improves.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        tloss : float
            Training loss.
        vloss : float
            Validation loss.
        tacc : float
            Training accuracy.
        vacc : float
            Validation accuracy.
        net
            Neural network model.
        optimizer
            Model optimizer.
        """
        self.progress["validation_accuracies"].append(vacc)
        self.progress["train_accuracies"].append(tacc)
        self.progress["validation_losses"].append(vloss)
        self.progress["train_losses"].append(tloss)
        self.patience_state += 1

        if vacc > self.best["validation_accuracy"]:
            self.best["train_loss"] = tloss
            self.best["validation_loss"] = vloss
            self.best["train_accuracy"] = tacc
            self.best["valid_accuracy"] = vacc
            self.best["epoch"] = epoch
            self.patience_state = 0
            checkpoint = {"state_dict": net.state_dict, "optimizer": optimizer}
            self.save(checkpoint, self.name)

    def save(self, checkpoint, name: str, model_path: str = None) -> None:
        """
        Saves model checkpoint to file.

        Parameters
        ----------
        checkpoint : dict
            Model checkpoint dictionary.
        name : str
            Model name.
        model_path : str, optional
            Custom model path. Use object save_path and model name for saving if left at default (None).
        """
        if model_path is None:
            model_path = os.path.join(self.save_path, name + ".pt")
        mat_path = model_path[:-2] + "mat"
        try:
            torch.save(checkpoint, model_path)
            save_dict = {key: value for key, value in self.progress.items()}
            save_dict.update({key: value for key, value in self.best.items()})
            savemat(mat_path, save_dict)
        except OSError:
            LOG.error(f"Error saving model to file: {model_path}")

    def load(self, mat_path):
        """
        Loads training progress from saved .mat file.

        Parameters
        ----------
        mat_path : str
            Path to saved .mat file.
        """
        data = loadmat(mat_path)
        for key, value in data.items():
            if key in self.progress.keys():
                self.progress[key] = value
            elif key in self.best.keys():
                self.best[key] = value
