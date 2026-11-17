"""Fit the final model on all seven IDSSE games and freeze it to models/<name>.json.

    python -m possession.train                 # writes models/hmm_v1.json

Same procedure as one fold of the leave-one-game-out test, with nothing held out: fit the two
logistic regressions on every degraded game, decode every game with every cost setting, keep the
costs with the best mean spellF1. The provided match is not read here at all.

The JSON is committed to the repository, so the submission is regenerated from these exact weights
without retraining.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

from possession import REPO_ROOT, features, idsse
from possession.cv import SEEDS, pm, stable_seed
from possession.degrade import LEVELS, degrade
from possession.hmm import HMM

MODELS_DIR = os.path.join(REPO_ROOT, "models")


def train(seeds=(SEEDS[0],), log=print):
    method = HMM()
    games = [idsse.load(m) for m in idsse.match_ids()]
    examples, decoded = [], []
    for game in games:
        for li, level in enumerate(LEVELS):
            for seed in seeds:
                feats = method.features(degrade(game, level, stable_seed(game.name, li, seed)))
                examples.append((feats, game.labels))
                decoded.append((game, feats))
        log("  features {}".format(game.name))
    model = method.fit(examples)

    totals = np.zeros(len(method.grid))
    for game, feats in decoded:
        runs = pm.valid_runs(game.labels)
        paths = method.decode(method.emissions(model, feats), game.period, method.grid)
        totals += [pm.spell_scores(game.labels, paths[p], runs)["mean_f1"]
                   for p in range(len(method.grid))]
    best = int(np.argmax(totals))
    return {
        "name": "hmm_v1",
        "method": "hmm",
        "costs": method.grid[best],
        "train_spell_f1": totals[best] / len(decoded),
        "team": model["team"],
        "dead": model["dead"],
        "team_features": list(features.TEAM_FEATURES),
        "dead_features": list(features.DEAD_FEATURES),
        "trained_on": ["IDSSE-" + m for m in idsse.match_ids()],
        "noise_levels": [dataclasses.asdict(level) for level in LEVELS],
        "seeds": list(seeds),
        "numpy": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="hmm_v1")
    args = parser.parse_args()
    frozen = train(log=lambda s: print(s, flush=True))
    frozen["name"] = args.name
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, args.name + ".json")
    with open(path, "w", newline="\n") as handle:
        json.dump(frozen, handle, indent=1)
        handle.write("\n")
    print("costs {}  mean train spellF1 {:.4f}\nwrote {}".format(
        frozen["costs"], frozen["train_spell_f1"], os.path.relpath(path, REPO_ROOT)))


if __name__ == "__main__":
    sys.exit(main())
