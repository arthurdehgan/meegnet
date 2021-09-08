import logging
import torch
from torch import nn


class Flatten(nn.Module):
    # Flatten layer used to connect between feature extraction and classif parts of a net.
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class CustomNet(nn.Module):
    def __init__(self, model_name, input_size):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.name = model_name
        logging.info(model_name)

    def _get_lin_size(self, layers):
        return nn.Sequential(*layers)(torch.zeros((1, *self.input_size))).shape[-1]

    def forward(self, x):
        return self.model(x)

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"


class FullNet(CustomNet):
    def __init__(
        self,
        model_name,
        input_size,
        hlayers=2,
        filter_size=7,
        nchan=5,
        n_linear=150,
        dropout=0.25,
        dropout_option="same",
        batchnorm=False,
    ):
        CustomNet.__init__(self, model_name, input_size)
        if dropout_option == "same":
            dropout1 = dropout
            dropout2 = dropout
        else:
            assert (
                dropout < 0.5
            ), "dropout cannot be higher than .5 in this configuration"
            if dropout_option == "double":
                dropout1 = dropout
                dropout2 = dropout * 2
            elif dropout_option == "inverted":
                dropout1 = dropout * 2
                dropout2 = dropout
            else:
                logging.warning("{} is not a valid option".format(dropout_option))

        layer_list = [
            # equivalent to doing nn.Linear(input_size[0], nchan)
            nn.Conv2d(input_size[0], nchan, (input_size[1], 1)),
            nn.ReLU(),
        ]
        prev = nchan
        for i in range(0, int(hlayers / 2)):
            nex = prev * 2
            layer_list += [nn.Conv2d(prev, nex, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(nchan)]
            layer_list += [nn.ReLU()]
            prev = nex

        if hlayers % 2 != 0:
            layer_list += [nn.Conv2d(prev, prev, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(nchan)]
            layer_list += [nn.ReLU()]

        for i in range(0, int(hlayers / 2)):
            nex = int(prev / 2)
            layer_list += [nn.Conv2d(prev, nex, (1, filter_size))]
            if batchnorm:
                layer_list += [nn.BatchNorm2d(nchan)]
            layer_list += [nn.ReLU()]
            prev = nex

        layer_list += [
            Flatten(),
            nn.Dropout(dropout),
        ]

        layers = nn.ModuleList(layer_list)
        lin_size = self._get_lin_size(layers)

        # 2606 Somehow, these linear layers were added in the feature extraction, this is wrong
        # and i don't remember when or why i would have done that, probably a copy pase mistake
        # layers.extend(
        #     nn.ModuleList(
        #         [
        #             nn.Linear(lin_size, n_linear),
        #             nn.Linear(n_linear, int(n_linear / 2)),
        #         ]
        #     )
        # )

        # Previous version: comment out this line in order to use previous state dicts
        self.feature_extraction = nn.Sequential(*layers)

        # layers = nn.ModuleList(
        #     [
        #         nn.Conv2d(input_size[0], 100, 3),
        #         nn.ReLU(),
        #         nn.MaxPool2d((2, 2)),
        #         nn.Dropout(dropout),
        #         nn.Conv2d(100, 100, 3),
        #         nn.MaxPool2d((2, 2)),
        #         nn.Dropout(dropout),
        #         nn.Conv2d(100, 300, (2, 3)),
        #         nn.MaxPool2d((2, 2)),
        #         nn.Dropout(dropout),
        #         nn.Conv2d(300, 300, (1, 7)),
        #         nn.MaxPool2d((2, 2)),
        #         nn.Dropout(dropout),
        #         nn.Conv2d(300, 100, (1, 3)),
        #         nn.Conv2d(100, 100, (1, 3)),
        #         Flatten(),
        #     ]
        # )

        # nn.Conv2d(input_size[0], n_channels, (input_size[1], 1)),
        # nn.BatchNorm2d(n_channels),
        # nn.ReLU(),
        # nn.Conv2d(n_channels, 2 * n_channels, (1, filter_size)),
        # nn.BatchNorm2d(2 * n_channels),
        # nn.ReLU(),
        # nn.MaxPool2d((1, 4)),
        # nn.Conv2d(2 * n_channels, 4 * n_channels, (1, int(filter_size / 2))),
        # nn.BatchNorm2d(4 * n_channels),
        # # nn.Dropout(dropout1),
        # nn.ReLU(),
        # nn.MaxPool2d((1, 4)),
        # nn.Conv2d(4 * n_channels, 8 * n_channels, (1, int(filter_size / 4))),
        # # nn.ReLU(),
        # # nn.MaxPool2d((1, 5)),
        # # nn.Conv2d(8 * n_channels, 16 * n_channels, (1, int(filter_size / 5))),
        # # nn.BatchNorm2d(16 * n_channels),
        # nn.BatchNorm2d(8 * n_channels),
        # nn.ReLU(),
        # Flatten(),

        # Previous version: unceomment this line and comment the next in order to use previous
        # state dicts Don't forget to remove unpacking (*)
        # layers.extend(
        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(lin_size, int(n_linear / 2)),
                    nn.Linear(int(n_linear / 2), 2),
                ]
            )
        )
        # Previous version: uncomment this line and comment out forward method in order to use
        # previous state dicts
        # self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.classif(self.feature_extraction(x))


class vanPutNet(CustomNet):
    def __init__(self, model_name, input_size, dropout=0.25):

        CustomNet.__init__(self, model_name, input_size)
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


class AutoEncoder(CustomNet):
    def __init__(
        self,
        model_name,
        input_size,
    ):
        CustomNet.__init__(self, model_name, input_size)

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
