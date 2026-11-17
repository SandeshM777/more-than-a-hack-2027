"""Leave-one-game-out scoring on IDSSE, with the competition's own scorer.

    python -m possession.cv baseline

For every IDSSE game, every noise level in degrade.LEVELS and every seed, the method's expensive
step runs once on the degraded game, then every parameter setting in its grid is decoded and
scored against that game's labels with evaluation/possession_metrics.evaluate().

Leave one game out: for each game in turn, the parameters are chosen on the other six (best mean
spellF1 over all their levels and seeds) and the score is read on the held-out one. The mean of
those seven held-out scores is our estimate of how the method will do on a match it has never
seen, which is the only honest estimate available, since nothing may be tuned on the provided
match.
"""

import argparse
import csv
import os
import sys
import time
import zlib

import numpy as np

from possession import REPO_ROOT, idsse
from possession.baseline import Baseline
from possession.degrade import LEVELS, degrade
from possession.hmm import HMM

sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

METHODS = {"baseline": Baseline, "hmm": HMM}
SEEDS = (0, 1)
OUT_DIR = os.path.join(REPO_ROOT, "cv_results")


def stable_seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def score_grid(method, games, levels=LEVELS, seeds=SEEDS, log=print):
    rows = []
    for game in games:
        for li, level in enumerate(levels):
            for seed in seeds:
                t0 = time.time()
                degraded = degrade(game, level, stable_seed(game.name, li, seed))
                state = method.prepare(degraded)
                for pi, params in enumerate(method.grid):
                    result = pm.evaluate(game.labels, method.decode(state, params))
                    rows.append({
                        "game": game.name, "level": level.label(), "seed": seed, "params": pi,
                        "spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
                        "n_pred_spells": result["n_pred_spells"],
                        "n_gt_spells": result["n_gt_spells"],
                    })
                log("  {:<13} {:<8} seed {}  {:.1f} s".format(game.name, level.label(), seed,
                                                            time.time() - t0))
    return rows


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def leave_one_game_out(rows, n_params):
    out = []
    for held in sorted({r["game"] for r in rows}):
        train = [r for r in rows if r["game"] != held]
        means = [_mean([r for r in train if r["params"] == pi], "spell_f1")
                 for pi in range(n_params)]
        best = int(np.argmax(means))  # first of any tie, i.e. the earlier grid entry
        test = [r for r in rows if r["game"] == held and r["params"] == best]
        entry = {"game": held, "params": best, "spell_f1": _mean(test, "spell_f1"),
                 "macro_f1": _mean(test, "macro_f1")}
        for level in sorted({r["level"] for r in test}):
            entry[level] = _mean([r for r in test if r["level"] == level], "spell_f1")
        out.append(entry)
    return out


def report(method, rows, logo):
    levels = sorted({r["level"] for r in rows})
    print("\n{}: mean spellF1 per parameter setting, all games, levels and seeds".format(
        method.name))
    print("  {:<28} {:>8} {:>8}  ".format("params", "spellF1", "macroF1")
          + "  ".join("{:>8}".format(lv) for lv in levels) + "  poss/gt")
    for pi, params in enumerate(method.grid):
        sel = [r for r in rows if r["params"] == pi]
        ratio = np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in sel])
        print("  {:<28} {:>8.4f} {:>8.4f}  ".format(str(params), _mean(sel, "spell_f1"),
                                                   _mean(sel, "macro_f1"))
              + "  ".join("{:>8.4f}".format(_mean([r for r in sel if r["level"] == lv],
                                                  "spell_f1")) for lv in levels)
              + "  {:>6.2f}".format(ratio))

    print("\nleave one game out (params chosen on the other games)")
    for e in logo:
        print("  {:<13} {:<28} spellF1 {:.4f}  macroF1 {:.4f}  ".format(
            e["game"], str(method.grid[e["params"]]), e["spell_f1"], e["macro_f1"])
            + "  ".join("{} {:.4f}".format(lv, e[lv]) for lv in levels))
    spells = [e["spell_f1"] for e in logo]
    print("\n  held-out spellF1 {:.4f} +/- {:.4f} (std over {} games)   macroF1 {:.4f}".format(
        np.mean(spells), np.std(spells), len(spells), np.mean([e["macro_f1"] for e in logo])))
    for lv in levels:
        print("    {:<8} {:.4f}".format(lv, np.mean([e[lv] for e in logo])))


def score_trainable(method, games, levels=LEVELS, seeds=SEEDS, log=print):
    """Leave one game out for a method that learns from data (it has fit()).

    For each held-out game: fit on the other six (all levels and seeds), decode every game with
    every cost setting, choose the costs with the best mean spellF1 on the six, then score the
    held-out game with those costs. The held-out game never touches the fit or the choice.
    """
    feats = {}
    for game in games:
        for li, level in enumerate(levels):
            for seed in seeds:
                feats[(game.name, li, seed)] = method.features(
                    degrade(game, level, stable_seed(game.name, li, seed)))
        log("  features {}".format(game.name))
    runs = {g.name: pm.valid_runs(g.labels) for g in games}

    rows, logo = [], []
    for held in games:
        t0 = time.time()
        model = method.fit([(feats[(g.name, li, s)], g.labels) for g in games if g is not held
                            for li in range(len(levels)) for s in seeds])
        train_sum, n_train, held_paths = np.zeros(len(method.grid)), 0, {}
        for game in games:
            for li in range(len(levels)):
                for seed in seeds:
                    paths = method.decode(method.emissions(model, feats[(game.name, li, seed)]),
                                          game.period, method.grid)
                    if game is held:
                        held_paths[(li, seed)] = paths
                    else:
                        train_sum += [pm.spell_scores(game.labels, paths[p], runs[game.name])
                                      ["mean_f1"] for p in range(len(method.grid))]
                        n_train += 1
        best = int(np.argmax(train_sum))
        entry = {"game": held.name, "params": best, "train_spell_f1": train_sum[best] / n_train}
        for (li, seed), paths in held_paths.items():
            result = pm.evaluate(held.labels, paths[best])
            rows.append({"game": held.name, "level": levels[li].label(), "seed": seed,
                         "params": best, "spell_f1": result["spell_f1_mean"],
                         "macro_f1": result["macro_f1"],
                         "n_pred_spells": result["n_pred_spells"],
                         "n_gt_spells": result["n_gt_spells"]})
        mine = [r for r in rows if r["game"] == held.name]
        entry["spell_f1"] = _mean(mine, "spell_f1")
        entry["macro_f1"] = _mean(mine, "macro_f1")
        for level in levels:
            entry[level.label()] = _mean([r for r in mine if r["level"] == level.label()],
                                         "spell_f1")
        logo.append(entry)
        log("  held out {:<13} {}  spellF1 {:.4f}  ({:.0f} s)".format(
            held.name, method.grid[best], entry["spell_f1"], time.time() - t0))
    return rows, logo


def report_logo(method, logo, levels=LEVELS):
    print("\n{}: leave one game out (fitted and tuned on the other games)".format(method.name))
    for e in logo:
        print("  {:<13} {:<44} spellF1 {:.4f}  macroF1 {:.4f}  ".format(
            e["game"], str(method.grid[e["params"]]), e["spell_f1"], e["macro_f1"])
            + "  ".join("{} {:.4f}".format(lv.label(), e[lv.label()]) for lv in levels))
    spells = [e["spell_f1"] for e in logo]
    print("\n  held-out spellF1 {:.4f} +/- {:.4f} (std over {} games)   macroF1 {:.4f}".format(
        np.mean(spells), np.std(spells), len(spells), np.mean([e["macro_f1"] for e in logo])))
    for lv in levels:
        print("    {:<8} {:.4f}".format(lv.label(), np.mean([e[lv.label()] for e in logo])))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--games", nargs="+", help="IDSSE match ids (default: all downloaded)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    method = METHODS[args.method]()
    ids = args.games or idsse.match_ids()
    print("loading {} IDSSE games".format(len(ids)), flush=True)
    games = [idsse.load(m) for m in ids]
    log = lambda s: print(s, flush=True)  # noqa: E731
    if hasattr(method, "fit"):
        rows, logo = score_trainable(method, games, seeds=args.seeds, log=log)
        report_logo(method, logo)
    else:
        rows = score_grid(method, games, seeds=args.seeds, log=log)
        logo = leave_one_game_out(rows, len(method.grid))
        report(method, rows, logo)
    write_csv(os.path.join(OUT_DIR, "{}_rows.csv".format(method.name)), rows)
    write_csv(os.path.join(OUT_DIR, "{}_logo.csv".format(method.name)), logo)
    print("\nwrote cv_results/{0}_rows.csv and cv_results/{0}_logo.csv".format(method.name))


if __name__ == "__main__":
    sys.exit(main())
