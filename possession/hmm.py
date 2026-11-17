"""Hidden Markov model over H, A and D, decoded with Viterbi.

Step 1, emissions: two logistic regressions trained on external games turn each frame's evidence
(features.py) into probabilities, P(dead) and P(home | alive), and from those log P(H), log P(A)
and log P(D).

Step 2, decoding: Viterbi finds the label sequence with the highest total of emission scores minus
a cost for every switch. Three costs, all tuned on external games:

    team      H <-> A, a change of possession
    dead_in   H or A -> D, the ball going dead
    dead_out  D -> H or A, a restart

A short flicker of evidence cannot pay for two switches, so it is ignored; a sustained one can,
and the switch lands where the evidence changes, not where a vote happens to tip.
"""

import itertools

import numpy as np

from possession import features, logreg
from possession.game import A, D, H

COSTS = {"team": (5, 7, 10, 14, 20, 40), "dead_in": (10, 20, 40, 80, 160),
         "dead_out": (10, 20, 40, 80, 160)}
GRID = [dict(zip(COSTS, values)) for values in itertools.product(*COSTS.values())]
SUBSAMPLE = 4          # every 4th frame is plenty for a dozen weights and keeps fitting quick
L2 = 1.0
EPS = 1e-4
ROUND = 6


def transitions(grid):
    """(P, 3, 3) transition scores, rows = from, cols = to, in H, A, D order."""
    out = np.zeros((len(grid), 3, 3))
    for p, c in enumerate(grid):
        out[p, H, A] = out[p, A, H] = -c["team"]
        out[p, H, D] = out[p, A, D] = -c["dead_in"]
        out[p, D, H] = out[p, D, A] = -c["dead_out"]
    return out


def viterbi(logp, trans):
    """Best path for each of P transition settings at once. logp (T, 3), trans (P, 3, 3)."""
    n, n_params = len(logp), len(trans)
    back = np.empty((n, n_params, 3), dtype=np.int8)
    delta = np.repeat(logp[:1], n_params, axis=0)
    for t in range(1, n):
        cand = delta[:, :, None] + trans
        back[t] = cand.argmax(axis=1)
        delta = cand.max(axis=1) + logp[t]
    path = np.empty((n_params, n), dtype=np.int8)
    path[:, -1] = delta.argmax(axis=1)
    rows = np.arange(n_params)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


class HMM:
    name = "hmm"
    grid = GRID

    def features(self, game):
        return features.compute(game)

    def fit(self, examples):
        """examples: [(features dict, labels)] from degraded external games."""
        team_x, team_y, dead_x, dead_y = [], [], [], []
        for feats, labels in examples:
            keep = np.zeros(len(labels), dtype=bool)
            keep[::SUBSAMPLE] = True
            graded = keep & (labels != 3)
            alive = graded & np.isin(labels, (H, A))
            team_x.append(feats["team"][alive])
            team_y.append(labels[alive] == H)
            dead_x.append(feats["dead"][graded])
            dead_y.append(labels[graded] == D)
        return {
            "team": logreg.fit(np.concatenate(team_x), np.concatenate(team_y), l2=L2,
                               intercept=False, center=False),
            "dead": logreg.fit(np.concatenate(dead_x), np.concatenate(dead_y), l2=L2),
        }

    def emissions(self, model, feats):
        p_dead = np.clip(logreg.predict(model["dead"], feats["dead"]), EPS, 1 - EPS)
        p_home = np.clip(logreg.predict(model["team"], feats["team"]), EPS, 1 - EPS)
        logp = np.empty((len(p_dead), 3))
        logp[:, H] = np.log1p(-p_dead) + np.log(p_home)
        logp[:, A] = np.log1p(-p_dead) + np.log1p(-p_home)
        logp[:, D] = np.log(p_dead)
        # Rounded to a fixed grid so a last-bit difference in tanh or log between two machines'
        # maths libraries cannot reach the decoder; Viterbi itself is only additions and maxima,
        # which IEEE arithmetic makes identical everywhere.
        return np.round(logp, ROUND)

    def decode(self, logp, period, grid):
        """(P, T) codes for each cost setting; D outside the halves."""
        out = np.full((len(grid), len(period)), D, dtype=np.int8)
        trans = transitions(grid)
        for p in np.unique(period[period > 0]):
            idx = np.flatnonzero(period == p)
            start, stop = int(idx[0]), int(idx[-1]) + 1
            out[:, start:stop] = viterbi(logp[start:stop], trans)
        return out
