import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--cores",
    default=-1,
    type=int,
    help="The number of cores to use, default=-1 (all cores)",
)
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="display more info must be int from 0 to 3, default=1",
)
parser.add_argument(
    "--clf",
    choices=["XGBoost", "SVM", "LDA", "QDA", "RF", "perceptron"],
    default="LDA",
    help="The classifier that will be used for the classification, default=LDA",
)
parser.add_argument(
    "-p",
    "--permutations",
    type=int,
    default=None,
    help="The number of permutations, default=None",
)
# parser.add_argument(
#     "-d",
#     "--data_type",
#     choices=["task", "rest", "passive"],
#     default="rest",
#     help="The type of data to use for classification",
# )
# parser.add_argument(
#     "--clean_type",
#     choices=["mf", "transdef_mf", "raw"],
#     default="transdef_mf",
#     help="The type of preprocessing step to use for classification",
# )
parser.add_argument(
    "-l",
    "--label",
    choices=["sex", "age", "subject"],
    default="sex",
    help="The type of classification to run, default=sex",
)
parser.add_argument(
    "-e", "--elec", default="MAG", help="The type of electrodes to keep, default=MAG"
)
parser.add_argument(
    "-f",
    "--feature",
    choices=["bands", "bins"],
    default="bands",
    help="The type of features to use, default=bands",
)
parser.add_argument(
    "--n_crossval",
    type=int,
    default=1000,
    help="The number of cross-validations to do, default=1000",
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.5,
    help="The percentage of the dataset to use as test set, default=.5",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=100,
    help="The number of iterations to do for random search hyperparameter optimization, default=100",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Launch the pipeline in test mode : will not save and will only do 2 iteration for each loop",
)
parser.add_argument(
    "--force-load",
    action="store_true",
    help="forces loading and printing of shapes even if classification has been done",
)
parser.add_argument(
    "-t", "--time", action="store_true", help="keeps time and prints it at the end"
)
parser.add_argument(
    "-o",
    "--out_path",
    default=".",
    help="Where to save the result matrices, data path in config.ini file + results/, by default",
)
parser.add_argument(
    "-i", "--in_path", default=".", help="Where is the data to load"
)
parser.add_argument(
    "--elec_axis",
    type=int,
    default=1,
    help="The axis of the data where the electrodes are, default=1",
)
args = parser.parse_args()
