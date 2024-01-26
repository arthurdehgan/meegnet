import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--testsplit",
    type=int,
    default=None,
    choices=[0, 1, 2, 3, 4, None],
    help="Will remove the 20% holdout set by default and usit for cross-val. Using 5-Fold instead of 4-Fold.",
)
parser.add_argument(
    "--flat",
    action="store_true",
    help="will flatten sensors dimension",
)
parser.add_argument(
    "--epoched",
    action="store_true",
    help="will load epoched data instead of the whole signal for each subject. This option only works for passive and smt data in which there are events that we use to generate epochs.",
)
parser.add_argument(
    "--randomsearch",
    action="store_true",
    help="Launches one cross-val on a subset of data or full random search depending on testsplit parameter",
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if testsplit option is true",
)
parser.add_argument(
    "--net-option",
    default="best_net",
    choices=["custom_net", "best_net", "EEGNet", "VGG", "vanPutNet", "MLP"],
    help="cNet is the custom net.",
)
parser.add_argument(
    "--dattype",
    default="rest",
    choices=["rest", "smt", "passive"],
    help="the type of data to be loaded",
)
parser.add_argument(
    "--eventclf",
    action="store_true",
    help="launches event classification instead of gender classification.",
)
parser.add_argument(
    "--subclf",
    action="store_true",
    help="launches subject classification instead of gender classification.",
)
parser.add_argument(
    "--crossval",
    action="store_true",
    help="wether to do a 4-FOLD cross-validation on the train+valid set.",
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help="limit of number of samples per subjects",
)
parser.add_argument(
    "-f", "--filters", default=8, type=int, help="The size of the first convolution"
)
parser.add_argument(
    "--sfreq",
    type=int,
    default=500,
    help="The sampling frequency of the data, mainly for loading the correct data",
)
parser.add_argument(
    "--maxpool",
    type=int,
    default=0,
    help="adds a maxpool layer in between conv layers",
)
parser.add_argument(
    "--batchnorm",
    action="store_true",
    help="adds a batchnorm layer in between conv layers",
)
parser.add_argument(
    "--permute-labels",
    action="store_true",
    help="Permutes the labesl in order to test for chance level",
)
parser.add_argument(
    "--printmem",
    action="store_true",
    help="Shows RAM information before and during the data loading process.",
)
parser.add_argument(
    "--log",
    action="store_true",
    help="stores all prints in a logfile instead of printing them",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.00001,
    help="the starting learning rate of the optimizer",
)
parser.add_argument(
    "--hlayers",
    type=int,
    default=1,
    help="number of hidden layers",
)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="patience for early stopping",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="net",
    help="Name of the network for file_save",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="number of workers to load data while gpu is processing",
)
parser.add_argument(
    "--train-size",
    type=float,
    default=0.8,
    help="The proportion of data to use in the train set",
)
parser.add_argument(
    "--save-path",
    type=str,
    default=".",
    help="The path where the model will be saved.",
)
parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="The path where the data samples can be found.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed to use for random splits.",
)
parser.add_argument(
    "--max-subj",
    default=1000,
    type=int,
    help="maximum number of subjects to use (1000 uses all subjects)",
)
parser.add_argument(
    "--age-min",
    default=1,
    type=int,
    help="The minimum age of the subjects to be included in the learning and testing process",
)
parser.add_argument(
    "--age-max",
    default=100,
    type=int,
    help="The maximum age of the subjects to be included in the learning and testing process",
)
parser.add_argument(
    "-s",
    "--sensors",
    default="MAG",
    choices=["GRAD", "MAG", "ALL"],
    help="The type of sensors to keep, default=MAG",
)
parser.add_argument(
    "--feature",
    default="temporal",
    choices=["temporal", "bands", "bins", "cov", "cosp"],
    help="Data type to use.",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    help="The batch size used for learning.",
)
parser.add_argument(
    "-d",
    "--dropout",
    default=0.25,
    type=float,
    help="The dropout rate of the linear layers",
)
parser.add_argument(
    "--times",
    action="store_true",
    help="Instead of running the training etc, run a series of test in order to choose best set of workers and batch sizes to get faster epochs.",
)
parser.add_argument(
    "--chunkload",
    action="store_true",
    help="Chunks the data and loads data batch per batch. Will be slower but is necessary when RAM size is too low to handle whole dataset.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="loads dummy data in the net to ensure everything is working fine",
)
parser.add_argument(
    "--dropout-option",
    default="same",
    choices=["same", "double", "inverted"],
    help="sets if the first dropout and the second are the same or if the first one or the second one should be bigger",
)
parser.add_argument(
    "-l", "--linear", default=100, type=int, help="The size of the second linear layer"
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["overwrite", "continue", "empty_run", "evaluate"],
    default="continue",
    help="Changes the mode the script is run for: overwrite will restart from scratch and overwrite any files with the same name; continue will load previous state and continue from last checkpoint; empty_run will run the training and testing without saving anything; evaluate will load the model to evaluate it on the test set.",
)
parser.add_argument(
    "-n",
    "--nchan",
    default=100,
    type=int,
    help="the number of channels for the first convolution, the other channel numbers scale with this one",
)
