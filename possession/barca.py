"""Read the provided match's automatic tracking (tactical ATD) into our Game format.

    python -m possession.barca          # parses once, caches to <MTAH_DATA_DIR>/cache/

Input only: the Game this returns has labels=None, and nothing here reads the ground truth. The
parsing mirrors starter/baseline.py: raw ATD line "<frame>:<p>;<p>;...:<ball>", each group
"x,y,speed" normalised 0-1, frame counting from 1 while submissions count from 0.

Positions are converted to metres with the origin at the centre spot, the same convention as the
IDSSE games, so every feature and parameter means the same thing on both.
"""

import os
import sys

import numpy as np

from possession import REPO_ROOT
from possession.game import Game

sys.path.insert(0, os.path.join(REPO_ROOT, "starter"))
import games  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (starter/baseline.py)

GAME = "BAR-ATH"
CACHE = os.path.join(games.DATA_DIR, "cache", "{}_tactical.npz".format(GAME))


def _xy(group, length, width):
    first = group.index(",")
    second = group.index(",", first + 1)
    return (float(group[:first]) - 0.5) * length, (float(group[first + 1:second]) - 0.5) * width


def parse(game=GAME):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game)
    n_frames = games.source_frames("tactical", game)
    length, width = games.GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)              # 0 home, 1 away, per track slot
    home_slots = np.flatnonzero(team == 0)
    away_slots = np.flatnonzero(team == 1)
    slot_pos = {int(s): ("home", i) for i, s in enumerate(home_slots)}
    slot_pos.update({int(s): ("away", i) for i, s in enumerate(away_slots)})

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    home = np.full((n_frames, len(home_slots), 2), np.nan, dtype=np.float32)
    away = np.full((n_frames, len(away_slots), 2), np.nan, dtype=np.float32)
    sides = {"home": home, "away": away}

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            if parts[2][0] != "N":
                ball[frame] = _xy(parts[2], length, width)
            for slot, group in enumerate(parts[1].split(";")):
                if group and group[0] != "N" and slot in slot_pos:
                    side, k = slot_pos[slot]
                    sides[side][frame, k] = _xy(group, length, width)

    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in games.half_video_blocks(game):
        period[first:last + 1] = half
    return Game(name=game, fps=games.GAMES[game]["fps"], pitch=(length, width), ball=ball,
                home=home, away=away, period=period, labels=None)


def load():
    if not os.path.exists(CACHE):
        parse().save(CACHE)
    return Game.load(CACHE)


if __name__ == "__main__":
    g = load()
    print("cached {}: {} frames, {} home and {} away track slots -> {}".format(
        g.name, g.n_frames, g.home.shape[1], g.away.shape[1], CACHE))
