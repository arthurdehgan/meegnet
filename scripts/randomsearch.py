from subprocess import call
import argparse
import random
import os
import pandas as pd


def run_script(params, fold=None, local=False, debug=False, options=None):
    arguments = f"--feature=temporal --data-path={args.data_path} --save-path={args.save_path} --model-name=sub_RS_{n_test} -s=ALL -b={params['bs']} -f={params['f']} --patience=20 --seed={args.seed} --hlayers={params['hlayers']} --lr={params['lr']} --linear={params['linear']} -d={params['d']} --maxpool={params['maxpool']} --nchan={params['nchan']} --subclf --log"
    if params["batchnorm"]:
        arguments += " --batchnorm"
    if options is not None:
        arguments += f" {options}"
    if fold is not None:
        arguments += f" --fold={fold}"
    else:
        arguments += " --crossval"
    if debug:
        arguments += " --debug --max-subj=10 --patience=1"
    if local:
        to_run = f"python {script_path} {arguments}"
    else:
        to_run = f"sbatch -o '/home/mila/d/dehganar/randomsearch_%j.log' -J randomsearch_{n_test} ../../randomsearch.sh '{arguments}'"
    print(to_run)
    call(to_run, shell=True)


tests = {
    "f": (6, 9, 12, 18),
    "linear": (250, 500, 500, 1000, 1500),
    "d": (0.5, 0.5),
    "nchan": (25, 50, 100, 200),
    "batchnorm": (True, False),
    "maxpool": (0, 10, 20),
    "hlayers": (1, 3, 5),
    "lr": (0.0001, 0.00005, 0.00001, 0.0005),
    "bs": (128, 256, 512),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-path",
    type=str,
    help="The path where the model will be saved.",
)
parser.add_argument(
    "--script",
    type=str,
    help="script path.",
)
parser.add_argument(
    "--data-path",
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
    "--seed",
    type=int,
    default=42,
    help="the seed to use",
)
parser.add_argument(
    "--n-tests",
    type=int,
    default=300,
    help="number of random searches to perform",
)
parser.add_argument(
    "--options",
    type=str,
    help="add more options from parser.py in this tag, they will be transfered to the final script",
)

args = parser.parse_args()

script_path = args.script if args.script is not None else "run.py"
chunkload = args.chunkload
options = args.options

params_set = set()
n_test = 0
csv_file = os.path.join(args.save_path, f"tested_params_seed{args.seed}.csv")
if os.path.exists(csv_file):
    tested = pd.read_csv(csv_file, index_col=0)
else:
    tested = pd.DataFrame()

if len(tested) >= args.n_tests:
    for i, row in tested.iterrows():
        for j in range(5):
            key = f"fold{j}"
            if row[key] == 0:
                params = dict(row)
                run_script(
                    params,
                    fold=j,
                    local=args.local,
                    debug=args.debug,
                    options=args.options,
                )
        if i > args.n_tests:
            break
else:
    n_done = len(tested)
    while n_test < args.n_tests - n_done:
        params = {
            "f": random.choice(tests["f"]),
            "linear": random.choice(tests["linear"]),
            "d": random.choice(tests["d"]),
            "hlayers": random.choice(tests["hlayers"]),
            "nchan": random.choice(tests["nchan"]),
            "batchnorm": random.choice(tests["batchnorm"]),
            "maxpool": random.choice(tests["maxpool"]),
            "bs": random.choice(tests["bs"]),
            "lr": random.choice(tests["lr"]),
        }
        scores = {
            "fold0": 0,
            "fold1": 0,
            "fold2": 0,
            "fold3": 0,
            "fold4": 0,
        }
        current_tested = tested.copy()
        if len(current_tested) != 0:
            for key, val in params.items():
                current_tested = current_tested.loc[current_tested[key] == float(val)]
        if len(current_tested) == 0:
            run_script(params, local=args.local, debug=args.debug, options=args.options)
            params.update(scores)
            tested = tested._append(params, ignore_index=True)
            tested.to_csv(csv_file)
            n_test += 1
            if args.debug:
                break
