from subprocess import call
import argparse
import random

N_TESTS = 100

tests = {
    "f": (12, 18, 24),
    "linear": (400, 800, 1600),
    "d": (0.5, 0.5),
    "nchan": (25, 50, 100),
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
script_path = args.script
chunkload = args.chunkload
options = args.options

params_set = set()
n_test = 27
while n_test < N_TESTS:
    params = {
        "f": random.choice(tests["f"]),
        "linear": random.choice(tests["linear"]),
        "d": random.choice(tests["d"]),
        "nchan": random.choice(tests["nchan"]),
    }
    if tuple(params.values()) not in params_set:
        arguments = f"--feature=temporal --path={data_path} --save={save_path} --model-name=randomsearchANN_{n_test} -e=ALL -b=32 -f={params['f']} --patience=20 --lr=0.00001 --linear={params['linear']} -d={params['d']} --nchan={params['nchan']} {options}"
        if debug:
            arguments += " --debug"
        if local:
            to_run = f"python cnn.py " + arguments
        else:
            to_run = f"sbatch -o '/home/mila/d/dehganar/randomsearch_%j.log' -J randomsearch_{n_test} randomsearch.sh '{arguments}'"
        call(to_run, shell=True)
        params_set.add(tuple(params.values()))
        n_test += 1
