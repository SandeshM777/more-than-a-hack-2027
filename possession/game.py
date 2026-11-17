"""One in-memory format for every game we read, external or provided.

Positions are in metres with the origin at the centre spot, x along the length of the pitch. A
frame where an object was not observed holds NaN. Labels use the scorer's own codes, so a label
array can go straight into evaluation/possession_metrics.evaluate().

The provided match never carries labels here: its Game has labels=None by construction.
"""

import dataclasses
import os

import numpy as np

# Same order as evaluation/possession_metrics.LABELS.
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
NOBODY = -1


@dataclasses.dataclass
class Game:
    name: str
    fps: int
    pitch: tuple             # (length_m, width_m)
    ball: np.ndarray         # (T, 2) float32
    home: np.ndarray         # (T, n_home_slots, 2) float32
    away: np.ndarray         # (T, n_away_slots, 2) float32
    period: np.ndarray       # (T,) int8: 1 and 2 are the halves, 0 is outside play
    labels: np.ndarray = None  # (T,) int8 in H/A/D/X, or None

    @property
    def n_frames(self):
        return len(self.period)

    def halves(self):
        """[(start, stop)] half-open frame spans of each period, in order."""
        spans = []
        for p in np.unique(self.period[self.period > 0]):
            idx = np.flatnonzero(self.period == p)
            spans.append((int(idx[0]), int(idx[-1]) + 1))
        return spans

    def copy(self):
        return dataclasses.replace(
            self, ball=self.ball.copy(), home=self.home.copy(), away=self.away.copy(),
            period=self.period.copy(),
            labels=None if self.labels is None else self.labels.copy())

    def save(self, path):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        extra = {} if self.labels is None else {"labels": self.labels}
        np.savez_compressed(path, name=self.name, fps=self.fps, pitch=np.array(self.pitch),
                            ball=self.ball, home=self.home, away=self.away, period=self.period,
                            **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                       ball=z["ball"], home=z["home"], away=z["away"], period=z["period"],
                       labels=z["labels"] if "labels" in z.files else None)


def to_letters(codes):
    return LETTERS[np.asarray(codes)]
