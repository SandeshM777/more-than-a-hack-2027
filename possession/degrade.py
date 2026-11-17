"""Make clean external tracking look like the provided automatic tracking.

IDSSE comes from an optical system with the ball on almost every frame. The provided match comes
from automatic video tracking with the ball missing on 41% of graded frames (DATA.md) and noisier
positions. Parameters tuned on clean data would be tuned for the wrong problem, so every external
game is degraded before it is scored.

The noise is drawn over wide ranges rather than copied from the provided match, so nothing here is
measured on, fitted to or calibrated against that match. The only number taken from the provided
game's documentation is the order of magnitude of the ball gap (41%), which the ranges bracket.

Everything is seeded, so a degraded game is reproducible bit for bit.
"""

import dataclasses

import numpy as np

FPS = 25


@dataclasses.dataclass(frozen=True)
class Noise:
    ball_missing: float = 0.4      # share of frames with no ball
    ball_gap_s: float = 2.0        # mean length of one ball gap
    ball_jitter_m: float = 0.3     # standard deviation of the ball position error
    player_jitter_m: float = 0.15
    player_missing: float = 0.05   # share of player observations dropped
    player_gap_s: float = 1.0
    smoothness: float = 0.9        # AR(1) coefficient of the position error per frame

    def label(self):
        return "ball{:.0f}%".format(100 * self.ball_missing)


# The robustness levels every method is scored at. The middle one brackets the provided match.
LEVELS = (Noise(ball_missing=0.2), Noise(ball_missing=0.4), Noise(ball_missing=0.6))


def gap_mask(n, missing, mean_gap_frames, rng):
    """Boolean (n,) mask, True inside a gap, from an alternating renewal process.

    Gap and visible run lengths are geometric, with means chosen so the long-run share of gap
    frames is `missing`.
    """
    if missing <= 0:
        return np.zeros(n, dtype=bool)
    if missing >= 1:
        return np.ones(n, dtype=bool)
    mean_gap = max(1.0, mean_gap_frames)
    mean_seen = mean_gap * (1 - missing) / missing
    # Draw enough alternating segments to cover n, then cut.
    n_pairs = int(n / (mean_gap + mean_seen) * 1.5) + 10
    gaps = rng.geometric(1.0 / mean_gap, n_pairs)
    seen = rng.geometric(1.0 / max(1.0, mean_seen), n_pairs)
    start_in_gap = rng.random() < missing
    lengths = np.empty(2 * n_pairs, dtype=np.int64)
    flags = np.empty(2 * n_pairs, dtype=bool)
    first, second = (gaps, seen) if start_in_gap else (seen, gaps)
    lengths[0::2], lengths[1::2] = first, second
    flags[0::2], flags[1::2] = start_in_gap, not start_in_gap
    mask = np.repeat(flags, lengths)
    while len(mask) < n:  # vanishingly rare; extend with a fresh draw
        mask = np.concatenate([mask, gap_mask(n - len(mask), missing, mean_gap_frames, rng)])
    return mask[:n]


def ar1_noise(shape, sigma, rho, rng):
    """Temporally correlated Gaussian error along axis 0, stationary std = sigma."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    white = rng.standard_normal(shape).astype(np.float32) * sigma
    out = np.empty(shape, dtype=np.float32)
    out[0] = white[0]
    scale = np.float32(np.sqrt(1 - rho * rho))
    rho = np.float32(rho)
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + scale * white[t]
    return out


def degrade(game, noise, seed):
    """A degraded copy of game. Labels and periods are untouched."""
    rng = np.random.default_rng(seed)
    g = game.copy()
    n = g.n_frames

    g.ball += ar1_noise(g.ball.shape, noise.ball_jitter_m, noise.smoothness, rng)
    g.ball[gap_mask(n, noise.ball_missing, noise.ball_gap_s * FPS, rng)] = np.nan

    for side in (g.home, g.away):
        side += ar1_noise(side.shape, noise.player_jitter_m, noise.smoothness, rng)
        for k in range(side.shape[1]):
            side[gap_mask(n, noise.player_missing, noise.player_gap_s * FPS, rng), k] = np.nan
    return g
