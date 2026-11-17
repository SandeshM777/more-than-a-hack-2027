# Our method: a hidden Markov model trained only on external games

Every frame gets a probability of H, A and D from two small logistic regressions over tracking
evidence. Viterbi then picks the label sequence that best fits those probabilities while paying a
cost for every switch. All weights and all three costs were fitted on the seven open IDSSE
Bundesliga games; the provided match is only read, once, by the frozen model.

## Regenerate the submission (CPU, about two minutes)

```bash
pip install -r requirements-possession.txt
python -m possession.barca        # read the provided tactical tracking into a cache
python -m possession.predict      # models/hmm_v1.json -> evaluation/results/hmm_v1.csv
```

`MTAH_DATA_DIR` must point at the match bundle, as for the starter code. The model file is
committed, so this needs no training and no external data.

## Retrain from scratch (optional, about five minutes plus a 2.6 GB download)

```bash
python -m possession.fetch_idsse  # IDSSE from figshare (CC BY 4.0), checksums verified
python -m possession.idsse        # parse to <MTAH_DATA_DIR>/external/idsse/cache
python -m possession.train        # writes models/hmm_v1.json
```

## How it was tested

`python -m possession.cv hmm --seeds 0` runs leave one game out on IDSSE: fit and tune on six
games, score the seventh with the organisers' own scorer, for each game in turn. The IDSSE games
are first degraded to look like automatic tracking (the ball hidden in bursts on 20%, 40% or 60%
of frames, correlated position jitter, dropped player observations), with every random draw
seeded. Results are in `cv_results/`.

| method | held-out spellF1 | held-out macroF1 |
|---|---|---|
| organisers' baseline (window chosen the same way) | 0.222 | 0.382 |
| this HMM | 0.417 | 0.850 |

Same degraded games for both rows. The HMM wins on all seven games and at every noise level
(0.441 / 0.415 / 0.394 with 20 / 40 / 60% of the ball hidden). The chosen costs barely matter
across the grid (0.4168 with an 80-setting grid, 0.4166 with the 150-setting one), which suggests
they will carry over. The frozen model's costs: team 7, dead_in 10, dead_out 80.

Nothing was fitted, tuned or checked against the provided match's ground truth or video.

## Files

| file | role |
|---|---|
| `game.py` | one format for every game: metres from the centre spot, NaN where unseen |
| `idsse.py`, `fetch_idsse.py` | download and parse IDSSE, labels from DFL BallStatus/BallPossession |
| `barca.py` | read the provided automatic tracking (input only, no labels) |
| `degrade.py` | make clean external tracking look like automatic tracking |
| `features.py` | per-frame evidence: who is near the ball, team shape, dead-ball cues |
| `logreg.py` | logistic regression by Newton's method, deterministic |
| `hmm.py` | emissions and the batched Viterbi decoder |
| `cv.py` | the leave-one-game-out test |
| `train.py`, `predict.py` | freeze the model, run it on the provided match |
| `baseline.py` | the organisers' baseline on the same format, for comparison |

## Reproducibility

Pinned versions in `requirements-possession.txt`. No randomness at prediction time. Dot products
are summed in a fixed order and log-probabilities are rounded to six decimals before decoding, so
last-bit differences between machines' maths libraries cannot change a label.
