"""The organisers' baseline (starter/baseline.py), rebuilt on our Game format.

Same two steps: the team of the nearest player wherever the ball is seen, then a centred majority
vote over +/- window frames, D only where nobody is seen across the whole window, ties to H.
Having it here lets us score the published floor on IDSSE with exactly the method that sets it.
"""

import numpy as np

from possession.game import A, D, H, NOBODY

CHUNK = 20000


def nearest_distances(game):
    """(T,) distance in metres from the ball to the nearest home and the nearest away player.

    inf where the ball or every player of that side is unseen.
    """
    out = []
    for side in (game.home, game.away):
        best = np.full(game.n_frames, np.inf, dtype=np.float32)
        for start in range(0, game.n_frames, CHUNK):
            stop = start + CHUNK
            diff = side[start:stop] - game.ball[start:stop, None, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            dist[~np.isfinite(dist)] = np.inf
            best[start:stop] = dist.min(axis=1) if dist.shape[1] else np.inf
        out.append(best)
    return out[0], out[1]


def nearest_team(game):
    """H or A per frame for the side of the nearest player, NOBODY where there is no answer."""
    home, away = nearest_distances(game)
    codes = np.where(home <= away, H, A).astype(np.int8)
    codes[~np.isfinite(np.minimum(home, away))] = NOBODY
    return codes


def smooth(codes, window):
    """Majority vote over a centred +/- window. Same rule as starter/baseline.smooth."""
    if window <= 0:
        return np.where(codes == NOBODY, D, codes).astype(np.int8)
    home_c = np.concatenate(([0], np.cumsum(codes == H)))
    away_c = np.concatenate(([0], np.cumsum(codes == A)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_c[high] - home_c[low]
    away = away_c[high] - away_c[low]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


class Baseline:
    """prepare() is the expensive, parameter-free part; decode() is cheap and takes the params."""

    name = "baseline"
    grid = [{"window": w} for w in (0, 25, 50, 75, 100, 125)]

    def prepare(self, game):
        return nearest_team(game)

    def decode(self, state, params):
        return smooth(state, params["window"])
