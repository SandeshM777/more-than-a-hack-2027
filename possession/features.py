"""Per-frame evidence for the decoder, computed identically on IDSSE and on the provided match.

Two feature sets come out:

  team  -- evidence for H against A, built antisymmetric (positive favours the home side, swapping
           the teams flips every sign), so the model learns no home bias.
  dead  -- evidence for a dead ball against a live one.

Everything is computed inside each half separately, so no difference ever spans halftime. Frames
outside the halves get zeros; they are decoded as D and never scored.

Only positions are used. They mean the same thing in both datasets once converted to metres from
the centre spot, which Game guarantees.
"""

import warnings

import numpy as np

from possession.baseline import nearest_distances

FPS = 25
TEAM_FEATURES = (
    "near_diff",        # nearest away minus nearest home distance to the ball, m, +-15
    "close_diff",       # same, +-3, only when someone is within 2 m of the ball
    "near_diff_1s",     # near_diff averaged over the seen frames within +-1 s
    "near_diff_4s",     # ... within +-4 s
    "spread_diff",      # home minus away spread around their own centroid, m
    "xspread_diff",
    "yspread_diff",
    "centroid_adv",     # home minus away centroid along the home attacking direction, m
    "flow",             # everyone's mean velocity along the home attacking direction, m/s
    "ball_adv",         # ball position along the home attacking direction, -1..1
    "play_adv",         # all players' centroid along the home attacking direction, -1..1
)
DEAD_FEATURES = (
    "ball_seen",
    "ball_outside",     # ball beyond a touchline or goal line
    "dist_outside",     # how far beyond, m, 0..5
    "line_dist",        # distance from the ball to the nearest line when inside, m, 0..10
    "ball_speed",       # m/s, 0..30
    "ball_speed_1s",
    "ball_still",       # seen and slower than 1 m/s
    "min_dist",         # nearest player to the ball, m, 0..20
    "player_speed_1s",  # median player speed, smoothed over +-1 s, m/s
    "player_speed_4s",
    "player_speed_10s",
    "speed_trend",      # 1 s minus 4 s player speed
    "since_seen",       # seconds since the ball was last seen, 0..10
    "spread_all",       # all players' spread, m
)


def _nanmean(a, axis):
    valid = np.isfinite(a)
    total = np.where(valid, a, 0).sum(axis)
    count = valid.sum(axis)
    out = total / np.maximum(count, 1)
    return np.where(count > 0, out, np.nan)


def moving_mean(x, half):
    """Centred mean over +-half frames, ignoring NaN; NaN where the window has no value."""
    valid = np.isfinite(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0), dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    count = ccnt[hi] - ccnt[lo]
    out = (csum[hi] - csum[lo]) / np.maximum(count, 1)
    out[count == 0] = np.nan
    return out


def speed(pos, k):
    """Speed in m/s from a central difference over +-k frames. pos is (T, ..., 2)."""
    out = np.full(pos.shape[:-1], np.nan, dtype=np.float64)
    if pos.shape[0] > 2 * k:
        d = pos[2 * k:].astype(np.float64) - pos[:-2 * k]
        out[k:pos.shape[0] - k] = np.sqrt((d ** 2).sum(-1)) / (2 * k / FPS)
    return out


def x_velocity(x, k):
    out = np.full(len(x), np.nan)
    if len(x) > 2 * k:
        out[k:len(x) - k] = (x[2 * k:] - x[:-2 * k]) / (2 * k / FPS)
    return out


def since_seen(seen):
    """Frames since the last seen frame (0 on a seen frame), large before the first one."""
    idx = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 6))
    return idx - last


def attack_direction(home, away):
    """+1 if the home side attacks towards +x in this half, -1 otherwise.

    A side's average x over the whole half is on its own half of the pitch, because that is where
    its goalkeeper and back line spend the match.
    """
    return 1.0 if np.nanmean(home[..., 0]) < np.nanmean(away[..., 0]) else -1.0


def _side_shape(side):
    cx = _nanmean(side[..., 0], 1)
    cy = _nanmean(side[..., 1], 1)
    dx = side[..., 0] - cx[:, None]
    dy = side[..., 1] - cy[:, None]
    spread = np.sqrt(_nanmean(dx ** 2 + dy ** 2, 1))
    return cx, spread, np.sqrt(_nanmean(dx ** 2, 1)), np.sqrt(_nanmean(dy ** 2, 1))


def _half(ball, home, away, d_home, d_away, pitch):
    length, width = pitch
    n = len(ball)
    direction = attack_direction(home, away)
    seen = np.isfinite(ball[:, 0])
    bx = np.where(seen, ball[:, 0], 0.0).astype(np.float64)
    by = np.where(seen, ball[:, 1], 0.0).astype(np.float64)

    # team evidence
    with np.errstate(invalid="ignore"):  # inf - inf where neither side is seen near the ball
        diff = np.where(seen, np.clip(d_away.astype(np.float64) - d_home, -15, 15), np.nan)
    diff[seen & ~np.isfinite(diff)] = 0.0
    nearest = np.minimum(d_home, d_away)
    close = seen & (nearest < 2.0)
    cx_h, spread_h, xs_h, ys_h = _side_shape(home)
    cx_a, spread_a, xs_a, ys_a = _side_shape(away)
    everyone = np.concatenate([home, away], axis=1)
    cx_all = _nanmean(everyone[..., 0], 1)
    cy_all = _nanmean(everyone[..., 1], 1)
    spread_all = np.sqrt(_nanmean((everyone[..., 0] - cx_all[:, None]) ** 2
                                  + (everyone[..., 1] - cy_all[:, None]) ** 2, 1))

    team = np.column_stack([
        np.nan_to_num(diff),
        np.where(close, np.clip(np.nan_to_num(diff), -3, 3), 0.0),
        np.nan_to_num(moving_mean(diff, FPS)),
        np.nan_to_num(moving_mean(diff, 4 * FPS)),
        np.nan_to_num(spread_h - spread_a),
        np.nan_to_num(xs_h - xs_a),
        np.nan_to_num(ys_h - ys_a),
        np.nan_to_num(direction * (cx_h - cx_a)),
        np.nan_to_num(direction * moving_mean(x_velocity(cx_all, FPS // 2), FPS // 2)),
        np.where(seen, direction * bx / (length / 2), 0.0),
        np.nan_to_num(direction * cx_all / (length / 2)),
    ])

    # dead-ball evidence
    beyond = np.maximum(np.abs(bx) - length / 2, np.abs(by) - width / 2)
    ball_speed = speed(ball, 3)
    player_speed = speed(everyone, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_speed = np.nanmedian(player_speed, axis=1)
    ps1 = moving_mean(median_speed, FPS)
    ps4 = moving_mean(median_speed, 4 * FPS)
    dead = np.column_stack([
        seen.astype(np.float64),
        (seen & (beyond > 0)).astype(np.float64),
        np.where(seen, np.clip(beyond, 0, 5), 0.0),
        np.where(seen & (beyond <= 0), np.clip(-beyond, 0, 10), 0.0),
        np.clip(np.nan_to_num(ball_speed), 0, 30),
        np.clip(np.nan_to_num(moving_mean(ball_speed, FPS)), 0, 30),
        (seen & (np.nan_to_num(ball_speed, nan=99.0) < 1.0)).astype(np.float64),
        np.where(seen, np.clip(nearest, 0, 20), 0.0),
        np.clip(np.nan_to_num(ps1), 0, 8),
        np.clip(np.nan_to_num(ps4), 0, 8),
        np.clip(np.nan_to_num(moving_mean(median_speed, 10 * FPS)), 0, 8),
        np.clip(np.nan_to_num(ps1 - ps4), -4, 4),
        np.clip(since_seen(seen) / FPS, 0, 10),
        np.nan_to_num(spread_all),
    ])
    assert team.shape == (n, len(TEAM_FEATURES)) and dead.shape == (n, len(DEAD_FEATURES))
    return team, dead


def compute(game):
    """{'team': (T, n_team), 'dead': (T, n_dead)} float32."""
    team = np.zeros((game.n_frames, len(TEAM_FEATURES)), dtype=np.float32)
    dead = np.zeros((game.n_frames, len(DEAD_FEATURES)), dtype=np.float32)
    d_home, d_away = nearest_distances(game)
    for start, stop in game.halves():
        sl = slice(start, stop)
        t, d = _half(game.ball[sl], game.home[sl], game.away[sl], d_home[sl], d_away[sl],
                     game.pitch)
        team[sl], dead[sl] = t, d
    return {"team": team, "dead": dead}
