import os
from collections import defaultdict, OrderedDict
import logging
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import numpy as np
from huggingface_hub import hf_hub_download

LOG = logging.getLogger("meegnet")


def create_net(net_option, input_size, n_outputs, net_params=None):
    if net_option == "MLP":
        return MLP(
            input_size=input_size,
            n_outputs=n_outputs,
            hparams={
                "mlp_width": net_params["linear"],
                "mlp_depth": net_params["hlayers"],
                "mlp_dropout": net_params["dropout"],
            },
        )
    elif net_option == "meegnet":
        return meegnet(input_size, n_outputs)
    elif net_option == "custom_net":
        return FullNet(
            input_size,
            n_outputs,
            net_params["hlayers"],
            net_params["filters"],
            net_params["nchan"],
            net_params["linear"],
            net_params["dropout"],
            net_params["batchnorm"],
            net_params["maxpool"],
        )
    elif net_option in ("VGG", "vgg"):
        return VGG16_NET(input_size, n_outputs)
    elif net_option in ("EEGNet", "eegnet"):
        return EEGNet(input_size, n_outputs)
    elif net_option == "vanPutNet":
        return vanPutNet(input_size, n_outputs)
    else:
        raise AttributeError(f"Bad network option: {net_option}")


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
            **kwargs,
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
    def __init__(self, input_size, n_outputs):
        super(customNet, self).__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs

    def forward(self, x):
        feats = self.feature_extraction(x).to(torch.float64)
        outs = self.classif(feats).to(torch.float64)
        return outs

    def _get_lin_size(self, layers):
        return nn.Sequential(*layers)(torch.zeros((1, *self.input_size))).shape[-1]


class EEGNet(customNet):
    def __init__(
        self,
        input_size,
        n_outputs,
        filter_size=64,
        n_filters=16,
        dropout=0.5,
        dropout_option="Dropout",
        depthwise_multiplier=2,
    ):
        customNet.__init__(self, input_size, n_outputs)
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
                input_size[0],
                n_filters,
                (1, filter_size),
                padding=(1, (filter_size) // 2),
                # padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters),
            # depthwise_constraind=maxnorm(1.) not used
            DepthwiseConv2d(
                n_filters,
                (n_channels, 1),
                depthwise_multiplier=depthwise_multiplier,
                padding=0,
                bias=False,
            ),
            # Changed n_filters to *2 becaus of dimension error,
            # TODO check if it was originally a typo in our code
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutType(dropout),
            SeparableConv2d(n_filters * 2, n_filters * 2, (1, 16), padding=(1, 8), bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutType(dropout),
            Flatten(),
        ]

        layers = nn.ModuleList(layer_list).to(torch.float64)
        lin_size = self._get_lin_size(layers)

        self.feature_extraction = nn.Sequential(*layers).to(torch.float64)

        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    # not using the kernel_constraint=max_norm(norm_rate) parameter
                    nn.Linear(lin_size, n_outputs).to(torch.float64),
                ]
            )
        )


# This implementation is rather common and found on various blogs/github repos
class VGG16_NET(customNet):
    def __init__(self, input_size, n_outputs):
        super(VGG16_NET, self).__init__(input_size, n_outputs)

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
    """Just an MLP"""

    def __init__(self, input_size, n_outputs, hparams):
        customNet.__init__(self, input_size, n_outputs)
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


class meegnet(customNet):
    def __init__(
        self,
        input_size,
        n_outputs,
        n_linear=2000,
        dropout=0.5,
    ):
        super(meegnet, self).__init__(input_size, n_outputs)
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

        self.feature_extraction = nn.Sequential(*layers).to(torch.float64)
        self.classif = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Linear(lin_size, int(n_linear / 2)).to(torch.float64),
                    nn.Linear(int(n_linear / 2), n_outputs).to(torch.float64),
                ]
            )
        )


class FullNet(nn.Module):
    def __init__(
        self,
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

    def _get_lin_size(self, layers):
        return (
            nn.Sequential(*layers)
            .to(torch.float64)(torch.zeros((1, *self.input_size)))
            .shape[-1]
        )


class vanPutNet(customNet):
    def __init__(self, input_size, n_output, dropout=0.25):
        customNet.__init__(self, input_size, n_output)
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
        input_size,
    ):
        customNet.__init__(self, input_size)

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
        name,
        net_option,
        input_size,
        n_outputs,
        save_path=None,
        learning_rate=0.00001,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        n_folds=5,
        device="cuda",
    ):
        self.name = name
        self.input_size = input_size  # TODO here put assertions on the shape
        self.net = create_net(net_option, input_size, n_outputs)
        self.n_outputs = n_outputs
        self.n_folds = n_folds
        self.criterion = criterion
        self.save_path = save_path
        self.lr = learning_rate
        self.optimizer = optimizer(self.net.parameters(), lr=learning_rate)
        self.results = defaultdict(lambda: 0)
        self.checkpoint = defaultdict(lambda: 0)

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
        batch_size=128,
        patience=20,
        max_epoch=None,
        model_path=None,
        num_workers=4,
    ):
        assert (
            dataset.data[0].shape == self.input_size
        ), "Size of the dataset samples should match the input size of the network."
        train_accs = []
        valid_accs = []
        train_losses = []
        valid_losses = []
        best_vloss = float("inf")
        best_vacc = 0.5
        best_epoch = 0
        patience_state = self.results["current_patience"]
        epoch = self.checkpoint["epoch"]
        self.net.train()
        self.batch_size = batch_size
        self.num_workers = num_workers

        LOG.info("Creating DataLoaders...")
        train_index, valid_index, test_index = dataset.data_split(0.8, 0.1, 0.1)
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

        LOG.info("Starting Training with:")
        LOG.info(f"batch size: {batch_size}")
        LOG.info(f"learning rate: {self.lr}")
        LOG.info(f"patience: {patience}")
        while patience_state < patience:
            epoch += 1
            n_batches = len(trainloader)
            for i, batch in enumerate(trainloader):
                self.optimizer.zero_grad()
                if len(batch) > 2:
                    X, y, groups = batch
                else:
                    X, y = batch
                y = y.view(-1)
                targets = Variable(y.type(torch.LongTensor)).to(self.device)
                X = X.view(-1, *self.input_size).to(torch.float64).to(self.device)
                out = self.net.forward(X)
                loss = self.criterion(out, targets)
                loss = loss.to(torch.float64)
                loss.backward()
                self.optimizer.step()
                progress = f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {loss:.5f}"
                if n_batches > 10:
                    if i % (n_batches // 10) == 0:
                        LOG.info(progress)
                else:
                    LOG.info(progress)
            train_loss, train_acc = self.evaluate(trainloader)
            valid_loss, valid_acc = self.evaluate(validloader)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_loss < best_vloss:
                best_vacc = valid_acc
                best_vloss = valid_loss
                best_epoch = epoch
                patience_state = 0
                self.results = {
                    "acc_score": [best_vacc],
                    "loss_score": [best_vloss],
                    "acc": valid_accs,
                    "train_acc": train_accs,
                    "valid_loss": valid_losses,
                    "train_loss": train_losses,
                    "best_epoch": best_epoch,
                    "patience": patience,
                    "current_patience": patience_state,
                }
                self.checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if self.save_path is not None:
                    self.save(model_path)
            else:
                patience_state += 1
            LOG.info("Epoch: {}".format(epoch))
            LOG.info(" [LOSS] TRAIN {} / VALID {}".format(train_loss, valid_loss))
            LOG.info(" [ACC] TRAIN {} / VALID {}".format(train_acc, valid_acc))
            if max_epoch is not None:
                if epoch == max_epoch:
                    LOG.info("Max number of epoch reached. Stopping training.")
                    break

    def fit(self, *args, **kwargs):
        return self.train(args, *kwargs)

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
                X = X.view(-1, *self.input_size).to(torch.float64).to(self.device)
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

    def _load_net(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_path, self.name + ".pt")
        if os.path.exists(model_path):
            LOG.info("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            net_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
        mat_path = model_path[:-2] + "mat"
        if os.path.exists(mat_path):
            mat_data = loadmat(mat_path)
        else:
            LOG.warning(f"Error while loading checkpoint from {model_path}")
        return net_state, optimizer_state, mat_data

    def load(self, model_path=None):
        net_state, optimizer_state, mat_data = self._load_net(model_path)
        if net_state[list(net_state.keys())[-1]].shape[0] == self.n_outputs:
            self.net.load_state_dict(net_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.results = mat_data
        else:
            feat_state_dict = OrderedDict()
            for key, value in net_state.items():
                if key.startswith("feature"):
                    feat_state_dict[".".join(key.split(".")[1:])] = value
            self.net.feature_extraction.load_state_dict(feat_state_dict)

    def save(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_path, self.name + ".pt")
        mat_path = model_path[:-2] + "mat"
        torch.save(self.checkpoint, model_path)
        savemat(mat_path, self.results)

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
