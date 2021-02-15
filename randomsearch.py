from subprocess import call
import argparse
import random

N_TESTS = 100

tests = {
    "f": (3, 5, 7, 9, 12),
    "linear": (100, 200, 400, 800, 1000),
    "d": (0.25, 0.35, 0.5),
    "nchan": (10, 25, 50, 100, 200),
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

args = parser.parse_args()
data_path = args.path
save_path = args.save
if not save_path.endswith("/"):
    save_path += "/"
script_path = args.script

params_set = set()
n_test = 0
while n_test < N_TESTS:
    params = {
        "f": random.choice(tests["f"]),
        "linear": random.choice(tests["linear"]),
        "d": random.choice(tests["d"]),
        "nchan": random.choice(tests["nchan"]),
    }
    if tuple(params.values()) not in params_set:
        call(
            f"python {script_path} --feature=temporal --path=/home/arthur/github/camcan/data/datasamples/ --save=/home/arthur/github/camcan/models/ --model-name=randomsearchANN_{n_test} -e=\"ALL\" -b=32 --chunkload -f={params['f']} --patience=20 --lr=0.00001 --linear={params['linear']} -d={params['d']} --nchan={params['nchan']}",
            shell=True,
        )
        params_set.add(tuple(params.values()))
        n_test += 1
