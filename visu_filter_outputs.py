import matplotlib.pyplot as plt
from parsing import parser
from params import TIME_TRIAL_LENGTH
from cnn import create_net
from utils import load_checkpoint
from torch import nn
from viz import generate_topomap, load_info

DEVICE = "cpu"


if __name__ == "__main__":
    parser.add_argument(
        "--fold",
        default=None,
        help="will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if notest option is true",
    )
    args = parser.parse_args()

    suffixes = ""
    if args.batchnorm:
        suffixes += "_BN"
    if args.maxpool != 0:
        suffixes += f"_maxpool{args.maxpool}"

    if args.feature == "bins":
        trial_length = 241
    if args.feature == "bands":
        trial_length = 5
    elif args.feature == "temporal":
        trial_length = TIME_TRIAL_LENGTH
    elif args.feature == "cov":
        # TODO
        pass
    elif args.feature == "cosp":
        # TODO
        pass

    if args.elec == "MAG":
        n_channels = 102
    elif args.elec == "GRAD":
        n_channels = 204
    elif args.elec == "ALL":
        n_channels = 306

    input_size = (n_channels // 102, 102, trial_length)

    # WARNING: using an older version of networks: fold was saved from 0 to 4 instead of 1 to 5 !! TODO
    name = f"{args.model_name}_{args.seed}_fold{args.fold}_{args.elec}_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
    name += suffixes

    if args.subclf:
        # TODO
        raise "not yet implemented for subclf"
    else:
        n_outputs = 2

    info = load_info(args.path)

    model_filepath = "models/" + name + ".pt"
    net = create_net(args.net_option, name, input_size, n_outputs, args)
    epoch, net_state, optimizer_state = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)
    net.to(DEVICE)

    print(net)
    model_weights = []
    conv_layers = []
    model_children = list(net.children())

    counter = 0
    for seq in model_children:
        for layer in seq:
            if type(layer) == nn.Conv2d:
                counter += 1
                model_weights.append(layer.weight)
                conv_layers.append(layer)

    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(10, 10, i + 1)
        _ = generate_topomap(filter[0, :, :].detach(), info)
        plt.axis("off")
    plt.savefig("figures/filter.png")

    # TODO implement https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    # We need to see data after each filter
