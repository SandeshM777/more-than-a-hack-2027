"""Run a frozen model on the provided match and write the submission CSV.

    python -m possession.predict                       # models/hmm_v1.json -> evaluation/results/hmm_v1.csv
    python -m possession.predict --model models/hmm_v1.json

Reads only the provided automatic tracking (via possession/barca.py) and the committed model. No
training, no tuning and no ground truth happen here: the weights and costs were fixed on external
games by possession/train.py.
"""

import argparse
import json
import os
import sys

import numpy as np

from possession import REPO_ROOT, barca
from possession.game import LETTERS
from possession.hmm import HMM

RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")


def predict(frozen, game):
    method = HMM()
    feats = method.features(game)
    model = {"team": frozen["team"], "dead": frozen["dead"]}
    return method.decode(method.emissions(model, feats), game.period, [frozen["costs"]])[0]


def write_submission(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "hmm_v1.json"))
    args = parser.parse_args()
    with open(args.model) as handle:
        frozen = json.load(handle)
    codes = predict(frozen, barca.load())
    path = write_submission(os.path.join(RESULTS_DIR, frozen["name"] + ".csv"), codes)
    counts = np.bincount(codes, minlength=4)
    print("{}: H {}  A {}  D {}  -> {}".format(frozen["name"], counts[0], counts[1], counts[2],
                                              os.path.relpath(path, REPO_ROOT)))


if __name__ == "__main__":
    sys.exit(main())
