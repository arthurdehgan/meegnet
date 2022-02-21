from subprocess import call
import argparse
import random
import pandas as pd

N_TESTS = 100

tests = {
    "f": (6, 9, 12, 18, 24),
    "linear": (250, 500, 1000, 1500, 2000),
    "d": (0.5, 0.5),
    "nchan": (25, 50, 100, 200),
    "batchnorm": (True, False),
    "maxpool": (0, 10, 20),
    "hlayers": (1, 3, 5, 7),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save",
    type=str,
    help="The path where the model will be saved.",
)
parser.add_argument(
    "--script",
    type=str,
    help="script path.",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="The path where the data samples can be found.",
)
parser.add_argument(
    "--local",
    action="store_true",
    help="if run in local don't use sbatch",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode, load less data and fixed learning rate.",
)
parser.add_argument(
    "--chunkload",
    action="store_true",
    help="Chunks the data and loads data batch per batch. Will be slower but is necessary when RAM size is too low to handle whole dataset.",
)
parser.add_argument(
    "--options",
    type=str,
    help="add more options from parser.py in this tag, they will be transfered to the final script",
)

args = parser.parse_args()
data_path = args.path
save_path = args.save
debug = args.debug
local = args.local
if debug:
    N_TESTS = 1

if not save_path.endswith("/"):
    save_path += "/"
script_path = args.script if args.script is not None else "cnn.py"
chunkload = args.chunkload
options = args.options

params_set = set()
n_test = 0
tested = pd.DataFrame()
seed = 42
while n_test < N_TESTS:
    params = {
        "f": random.choice(tests["f"]),
        "linear": random.choice(tests["linear"]),
        "d": random.choice(tests["d"]),
        "hlayers": random.choice(tests["hlayers"]),
        "nchan": random.choice(tests["nchan"]),
        "batchnorm": random.choice(tests["batchnorm"]),
        "maxpool": random.choice(tests["maxpool"]),
    }
    if tuple(params.values()) not in params_set:
        arguments = f"--feature=temporal --path={data_path} --save={save_path} --model-name=sub_RS_{n_test} -e=ALL -b=2048 -f={params['f']} --patience=20 --seed={seed} --hlayers={params['hlayers']} --lr=0.00002 --linear={params['linear']} -d={params['d']} --maxpool={params['maxpool']} --nchan={params['nchan']} --crossval --subclf "
        if params["batchnorm"]:
            arguments += " --batchnorm"
        if options is not None:
            arguments += f" {options}"
        if debug:
            arguments += " --debug"
        if local:
            to_run = f"python {script_path} {arguments}"
        else:
            to_run = f"sbatch -o '/home/mila/d/dehganar/randomsearch_%j.log' -J randomsearch_{n_test} randomsearch.sh '{arguments}'"
        print(to_run)
        call(to_run, shell=True)
        params_set.add(tuple(params.values()))
        n_test += 1
        tested = tested.append(params, ignore_index=True)
        break

tested.to_csv(f"tested_params_seed{seed}.csv")
