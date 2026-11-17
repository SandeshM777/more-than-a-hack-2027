"""Read IDSSE (DFL) games into our Game format, with per-frame H/A/D labels.

    python -m possession.idsse            # parse every downloaded game into the cache
    python -m possession.idsse --stats    # label statistics per cached game

The labels come straight from the DFL position feed: BallStatus says whether the ball is in play
and BallPossession says which team owns it. That is the same kind of derivation as the challenge's
ground truth (dead where the tracking marks the ball dead, otherwise the owning team).

The XML is streamed with lxml, one FrameSet (one object for one half) at a time, so a 400 MB file
parses in about a minute without holding a tree in memory.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from lxml import etree

from possession.fetch_idsse import IDSSE_DIR
from possession.game import A, D, H, X, Game

CACHE_DIR = os.path.join(IDSSE_DIR, "cache")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
MATCH_RE = re.compile(r"DFL-MAT-(\w+)\.xml$")
FPS = 25


def match_ids(raw_dir=IDSSE_DIR):
    found = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "DFL_04_03_positions_raw_observed_*.xml"))):
        found.append(MATCH_RE.search(path).group(1))
    return found


def _path(raw_dir, kind, match_id):
    hits = glob.glob(os.path.join(raw_dir, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
    if len(hits) != 1:
        raise FileNotFoundError("{} file for {} in {}".format(kind, match_id, raw_dir))
    return hits[0]


def read_meta(path):
    """(home_id, guest_id, pitch_length, pitch_width, title)."""
    root = etree.parse(path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    return (general.get("HomeTeamId"), general.get("GuestTeamId"),
            float(env.get("PitchX")), float(env.get("PitchY")), general.get("MatchTitle"))


def iter_framesets(path):
    """Yield (team_id, person_id, period, N, X, Y, possession, status) per FrameSet.

    possession and status are only filled for the ball; for players they are None.
    """
    context = etree.iterparse(path, events=("start", "end"), tag=("FrameSet", "Frame"),
                              huge_tree=True)
    current = None
    for event, elem in context:
        if elem.tag == "FrameSet":
            if event == "start":
                is_ball = elem.get("TeamId") == "BALL"
                current = {"team": elem.get("TeamId"), "person": elem.get("PersonId"),
                           "period": SECTIONS.get(elem.get("GameSection")), "ball": is_ball,
                           "n": [], "x": [], "y": [], "poss": [], "status": []}
            else:
                c = current
                if c["period"] is not None:
                    yield (c["team"], c["person"], c["period"],
                           np.array(c["n"], dtype=np.int64),
                           np.array(c["x"], dtype=np.float32),
                           np.array(c["y"], dtype=np.float32),
                           np.array(c["poss"], dtype=np.int8) if c["ball"] else None,
                           np.array(c["status"], dtype=np.int8) if c["ball"] else None)
                current = None
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        elif event == "end" and current is not None:
            attrib = elem.attrib
            current["n"].append(attrib["N"])
            current["x"].append(attrib["X"])
            current["y"].append(attrib["Y"])
            if current["ball"]:
                current["poss"].append(attrib.get("BallPossession", "0"))
                current["status"].append(attrib.get("BallStatus", "0"))
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def parse_game(match_id, raw_dir=IDSSE_DIR):
    """One IDSSE match as a Game: both halves back to back, one X frame between them."""
    home_id, guest_id, length, width, title = read_meta(_path(raw_dir, "02_01_matchinformation",
                                                              match_id))
    balls, players = {}, {}
    for team, person, period, n, x, y, poss, status in iter_framesets(
            _path(raw_dir, "04_03_positions_raw_observed", match_id)):
        if team == "BALL":
            balls[period] = (n, x, y, poss, status)
        elif team in (home_id, guest_id):
            players.setdefault((team, person), {})[period] = (n, x, y)
        # anything else (referees) is not part of the problem

    slots = {home_id: [], guest_id: []}
    for team, person in sorted(players):
        slots[team].append(person)

    parts = []
    for period in sorted(balls):
        n, x, y, poss, status = balls[period]
        n0, length_half = int(n.min()), int(n.max() - n.min() + 1)
        ball = np.full((length_half, 2), np.nan, dtype=np.float32)
        ball[n - n0] = np.stack([x, y], axis=1)

        labels = np.full(length_half, X, dtype=np.int8)
        alive = status == 1
        labels[n[~alive] - n0] = D
        labels[n[alive & (poss == 1)] - n0] = H
        labels[n[alive & (poss == 2)] - n0] = A

        sides = []
        for team in (home_id, guest_id):
            arr = np.full((length_half, len(slots[team]), 2), np.nan, dtype=np.float32)
            for k, person in enumerate(slots[team]):
                if period not in players[(team, person)]:
                    continue
                pn, px, py = players[(team, person)][period]
                keep = (pn >= n0) & (pn < n0 + length_half)
                arr[pn[keep] - n0, k] = np.stack([px[keep], py[keep]], axis=1)
            sides.append(arr)
        parts.append((ball, sides[0], sides[1], labels, period))

    def gap(n_slots):
        return np.full((1, n_slots, 2), np.nan, dtype=np.float32)

    ball, home, away, labels, period = [], [], [], [], []
    for i, (b, h, a, lab, p) in enumerate(parts):
        if i:
            ball.append(np.full((1, 2), np.nan, dtype=np.float32))
            home.append(gap(h.shape[1]))
            away.append(gap(a.shape[1]))
            labels.append(np.array([X], dtype=np.int8))
            period.append(np.array([0], dtype=np.int8))
        ball.append(b)
        home.append(h)
        away.append(a)
        labels.append(lab)
        period.append(np.full(len(lab), p, dtype=np.int8))

    return Game(name="IDSSE-" + match_id, fps=FPS, pitch=(length, width),
                ball=np.concatenate(ball), home=np.concatenate(home),
                away=np.concatenate(away), period=np.concatenate(period),
                labels=np.concatenate(labels)), title


def cache_path(match_id):
    return os.path.join(CACHE_DIR, "{}.npz".format(match_id))


def load(match_id):
    """The cached Game for one match, parsing it first if needed."""
    path = cache_path(match_id)
    if not os.path.exists(path):
        game, _ = parse_game(match_id)
        game.save(path)
    return Game.load(path)


def load_all():
    return [load(m) for m in match_ids()]


def label_stats(game):
    """Possession count, mean possession length and dead-ball share, as the scorer counts them."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "evaluation"))
    import possession_metrics as pm
    runs = pm.valid_runs(game.labels)
    stats = pm.possession_stats(game.labels, runs)
    graded = game.labels != X
    return {
        "frames_graded": int(graded.sum()),
        "possessions": len(pm.possession_spells(game.labels, runs)),
        "mean_possession_s": stats["mean_duration_s"],
        "dead_share": float((game.labels == D).sum() / graded.sum()),
        "home_share_alive": float((game.labels == H).sum()
                                  / np.isin(game.labels, (H, A)).sum()),
        "ball_seen": float(np.isfinite(game.ball[graded, 0]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-parse even if cached")
    args = parser.parse_args()

    for match_id in match_ids():
        if args.force or not os.path.exists(cache_path(match_id)):
            game, title = parse_game(match_id)
            game.save(cache_path(match_id))
            print("cached {}  {}  {} frames".format(match_id, title, game.n_frames), flush=True)
        if args.stats:
            s = label_stats(load(match_id))
            print("{}  poss {:>4}  mean {:>5.1f} s  dead {:.3f}  home(alive) {:.3f}  "
                  "ball seen {:.3f}".format(match_id, s["possessions"], s["mean_possession_s"],
                                            s["dead_share"], s["home_share_alive"],
                                            s["ball_seen"]))


if __name__ == "__main__":
    sys.exit(main())
