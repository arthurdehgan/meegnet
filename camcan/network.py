import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def create_net(net_option, name, input_size, n_outputs, device, args):
    if net_option == "MLP":
        return MLP(
            name=name,
            input_size=input_size,
            n_outputs=n_outputs,
            hparams={
                "mlp_width": args.linear,
                "mlp_depth": args.hlayers,
                "mlp_dropout": args.dropout,
            },
        ).to(device)
    elif net_option == "best_net":
        return testNet(name, input_size, n_outputs).to(device)
    elif net_option == "custom_net":
        return FullNet(
            name,
            input_size,
            n_outputs,
            args.hlayers,
            args.filters,
            args.nchan,
            args.linear,
            args.dropout,
            args.batchnorm,
            args.maxpool,
        ).to(device)
    elif net_option == "VGG":
        return VGG16_NET(
            name,
            input_size,
            n_outputs,
        ).to(device)
    elif net_option == "EEGNet":
        return EEGNet(
            name,
            input_size,
            n_outputs,
        ).to(device)
    elif net_option == "vanPutNet":
        return vanPutNet(
            name,
            input_size,
            n_outputs,
        ).to(device)


class Flatten(nn.Module):
    # Flatten layer used to connect between feature extraction and classif parts of a net.
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, depthwise_multiplier=1, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depthwise_multiplier,
            kernel_size,
            groups=in_channels,
            **kwargs
        )

    def forward(self, x):
        return self.depthwise(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class customNet(nn.Module):
    def __init__(self, name, input_size, n_outputs):
        super(customNet, self).__init__()
        self.input_size = input_size
        self.name = name
        self.n_outputs = n_outputs

    def forward(self, x):
        feats = self.feature_extraction(x)
        outs = self.classif(feats)
        return outs

    def _get_lin_size(self, layers):
        return nn.Sequential(*layers)(torch.zeros((1, *self.input_size))).shape[-1]


class EEGNet(customNet):
    def __init__(
        self,
        name,
        input_size,
        n_outputs,
        filter_size=64,
        n_filters=16,
        n_linear=150,
        dropout=0.5,
        dropout_option="Dropout",
        depthwise_multiplier=2,
    ):
        customNet.__init__(self, name, input_size, n_outputs)
        if dropout_option == "SpatialDropout2D":
            dropoutType = nn.Dropout2d
        elif dropout_option == "Dropout":
            dropoutType = nn.Dropout
        else:
            raise ValueError(
                "dropoutType must be one of SpatialDropout2D "
                "or Dropout, passed as a string."
            )

        n_channels = input_size[1]
        layer_list = [
            nn.Conv2d(
                input_size[0], n_filters, (1, filter_size), padding="same", bias=False
            ),
            nn.BatchNorm2d(n_filters),
            # depthwise_constraind=maxnorm(1.) not used
            DepthwiseConv2d(
                n_filters,
                (n_channels, 1),
                depthwise_multiplier=depthwise_multiplier,
                padding="valid",
                bias=False,
            ),
            # Changed n_filters to *2 becaus of dimension error,
            # TODO check if it was originally a typo in our code
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutType(dropout),
            SeparableConv2d(
                n_filters * 2, n_filters * 2, (1, 16), padding="same", bias=False
            ),
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutType(dropout),
            Flatten(),
        ]

        layers = nn.ModuleList(layer_list)
        lin_size = self._get_lin_size(layers)

        self.feature_extraction = nn.Sequential(*layers)

        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    # not using the kernel_constraint=max_norm(norm_rate) parameter
                    nn.Linear(lin_size, n_outputs),
                ]
            )
        )

    def forward(self, x):
        feats = self.feature_extraction(x)
        outs = self.classif(feats)
        return outs


# This implementation is rather common and found on various blogs/github repos
class VGG16_NET(customNet):
    def __init__(self, name, input_size, n_outputs):
        super(VGG16_NET, self).__init__(name, input_size, n_outputs)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_list = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            self.maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            self.maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            self.maxpool,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            self.maxpool,
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            self.maxpool,
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
                    nn.Linear(4096, self.n_outputs),
                ]
            )
        )


class MLP(customNet):
    """Just  an MLP"""

    def __init__(self, name, input_size, n_outputs, hparams):
        customNet.__init__(self, name, input_size, n_outputs)
        self.name = name
        n_inputs = np.prod(input_size)
        self.flatten = Flatten()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.flatten(x)
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class testNet(customNet):
    def __init__(
        self,
        name,
        input_size,
        n_outputs,
        n_linear=2000,
        dropout=0.5,
    ):
        super(testNet, self).__init__(name, input_size, n_outputs)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 20), stride=1)
        layer_list = [
            nn.Conv2d(input_size[0], 100, (input_size[1], 1)),
            nn.ReLU(),
            nn.Conv2d(100, 200, (1, 9)),
            self.maxpool,
            nn.ReLU(),
            nn.Conv2d(200, 200, (1, 9)),
            self.maxpool,
            nn.ReLU(),
            nn.Conv2d(200, 100, (1, 9)),
            self.maxpool,
            nn.ReLU(),
            Flatten(),
            nn.Dropout(dropout),
        ]

        layers = nn.ModuleList(layer_list)
        lin_size = self._get_lin_size(layers)

        self.feature_extraction = nn.Sequential(*layers)
        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(lin_size, int(n_linear / 2)),
                    nn.Linear(int(n_linear / 2), n_outputs),
                ]
            )
        )


class FullNet(nn.Module):
    def __init__(
        self,
        name,
        input_size,
        n_outputs,
        hlayers=2,
        filter_size=7,
        nchan=5,
        n_linear=150,
        dropout=0.25,
        batchnorm=False,
        maxpool=0,
    ):
        super(FullNet, self).__init__()
        self.input_size = input_size
        self.name = name

        layer_list = [
            nn.Conv2d(input_size[0], nchan, (input_size[1], 1)),
            nn.ReLU(),
        ]
        prev = nchan
        for i in range(0, int(hlayers / 2)):
            nex = prev * 2
            layer_list += [nn.Conv2d(prev, nex, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(nex)]
            if maxpool != 0:
                layer_list += [nn.MaxPool2d((1, maxpool), 1)]
            layer_list += [nn.ReLU()]
            prev = nex

        if hlayers % 2 != 0:
            layer_list += [nn.Conv2d(prev, prev, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(prev)]
            if maxpool != 0:
                layer_list += [nn.MaxPool2d((1, maxpool), 1)]
            layer_list += [nn.ReLU()]

        for i in range(0, int(hlayers / 2)):
            nex = int(prev / 2)
            layer_list += [nn.Conv2d(prev, nex, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(nex)]
            if maxpool != 0:
                layer_list += [nn.MaxPool2d((1, maxpool), 1)]
            layer_list += [nn.ReLU()]
            prev = nex

        layer_list += [
            Flatten(),
            nn.Dropout(dropout),
        ]

        layers = nn.ModuleList(layer_list)
        lin_size = self._get_lin_size(layers)

        self.feature_extraction = nn.Sequential(*layers)
        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(lin_size, int(n_linear / 2)),
                    nn.Linear(int(n_linear / 2), n_outputs),
                ]
            )
        )

    def forward(self, x):
        feats = self.feature_extraction(x)
        outs = self.classif(feats)
        return outs

    def _get_lin_size(self, layers):
        return nn.Sequential(*layers)(torch.zeros((1, *self.input_size))).shape[-1]


class vanPutNet(customNet):
    def __init__(self, model_name, input_size, n_output, dropout=0.25):
        customNet.__init__(self, model_name, input_size, n_output)
        layers = nn.ModuleList(
            [
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
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(300, 100, (1, 3)),
                nn.Conv2d(100, 100, (1, 3)),
                Flatten(),
            ]
        )

        lin_size = self._get_lin_size(layers)
        layers.append(nn.Linear(lin_size, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(customNet):
    def __init__(
        self,
        model_name,
        input_size,
    ):
        customNet.__init__(self, model_name, input_size)

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
