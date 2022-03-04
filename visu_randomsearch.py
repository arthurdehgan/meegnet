import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from parsing import parser

if __name__ == "__main__":
    """
    - Lister tous les mat files
    - Lister les log files
    - Si un mat n'est pas dans les log files:
        - on l'ajoutes a not_computed
      Sinon:
        - on prend les scores des 4 folds, on les average et on met tout dans un dictionnaire
    """

    N_TESTS = 100
    N_SEEDS = 10
    N_FOLDS = 5
    N_TOP = 1  # The number of best architecutres we print per seed
    found = []
    not_computed = []
    dico = {}
    for file in os.listdir("models/"):
        if (
            file.startswith("randomsearchANN")
            and file.endswith(".mat")
            and "fold0" in file
            and "0_b" in file
        ):
            try:
                sc = 0
                for j in range(N_SEEDS):
                    for i in range(N_FOLDS):
                        name = (
                            "_".join(file.split("_")[:2])
                            + f"_{j}_"
                            + "_".join(file.split("_")[3:4])
                            + f"_fold{i+1}_"
                            + "_".join(file.split("_")[5:])
                        )
                        sc += loadmat("models/" + name)["acc_score"] / (
                            N_FOLDS * N_SEEDS
                        )
                found.append("_".join(file.split("_")[:4]))
                dico[file] = sc
                # print(sc)
            except OSError as e:
                not_computed.append("_".join(file.split("_")[:4]))

    for file in os.listdir("models/"):
        if (
            file.startswith("randomsearchANN")
            and file.endswith(".log")
            and file[:-4] not in found
        ):
            not_computed.append(file)

    l = [(v, k) for k, v in dico.items()]
    for v, k in sorted(l):
        print(k, v)
    print()

    seed_archi_list = [defaultdict(int) for _ in range(N_TESTS)]
    for j in range(N_SEEDS):
        for file in os.listdir("models/"):
            if "fold0" in file and "0_b" in file and file.endswith(".mat"):
                for i in range(N_FOLDS):
                    name = (
                        "_".join(file.split("_")[:2])
                        + f"_{j}_"
                        + "_".join(file.split("_")[3:4])
                        + f"_fold{i}_"
                        + "_".join(file.split("_")[5:])
                    )
                    try:
                        seed_archi_list[j][file] += (
                            loadmat("models/" + name)["acc_score"] / N_FOLDS
                        )
                    except OSError as e:
                        continue
        for i in range(1, N_TOP + 1):
            print(sorted(seed_archi_list[j].items(), key=lambda x: x[1])[-i][::-1])
        print()

    counter = defaultdict(int)
    for j in range(N_SEEDS):
        rs = sorted(seed_archi_list[j].items(), key=lambda x: x[1])
        for i in range(1, N_TOP + 1):
            counter[rs[-i][0].split("_")[1]] += 1

    for key, val in counter.items():
        if val > 1:
            print(key, ":", val)
