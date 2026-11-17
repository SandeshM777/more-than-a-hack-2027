# Possession from tracking with an HMM trained only on other games

**Team:** Lionel Larp  
**Repository:** https://github.com/SandeshM777/more-than-a-hack-2027  
**Submission file:** produced by `python -m possession.predict` from the committed model `models/hmm_v1.json`

---

## 1. Approach

The ranking metric, spellF1, matches whole possessions by temporal overlap. A good submission has
to get the **number** of possessions and their **boundaries** right at the same time. A per-frame
classifier flickers between teams and invents possessions, while a fixed smoothing window trades one
error for the other: a short window keeps noise, a long one erases real short possessions and blurs
the boundaries. Both errors come from the same cause: a vote only sees how long a run is, not
whether anyone actually touched the ball.

So we treat the match as a **hidden Markov model** with three states, H, A and D:

1. **Emissions.** Two logistic regressions turn each frame's tracking evidence into probabilities:
   P(ball dead) and P(home has it | ball alive). These give log P(H), log P(A) and log P(D) for every
   frame.
2. **Decoding.** Viterbi finds the label sequence with the highest total log-probability minus a cost
   for every switch. There are three costs: H↔A (a change of possession), alive→D (the ball going
   dead) and D→alive (a restart).

A brief flicker of evidence cannot pay for two switches, so it is ignored. Sustained evidence can,
and the switch lands where the evidence changes, not where a vote happens to tip.

**Why this rather than something else.**
- **It carries over.** Every weight and cost must be learned on external games, which are cleaner
  than the provided automatic tracking. The model has 26 regression weights (11 team, 14 dead ball,
  one intercept) plus 3 costs, so there is little to overfit to the external domain.
- **It controls the metric directly.** The switch cost is exactly the knob spellF1 rewards: fewer
  spurious possessions without moving the real boundaries.
- **It is deterministic and fast.** No sampling, closed-form fitting, about two minutes on a CPU for
  the whole match, so byte-for-byte reproduction is straightforward.
- **Alternatives we considered and set aside:**
  - a larger learned model (gradient-boosted trees or a temporal network), which has many more
    parameters to carry from Bundesliga optical tracking to automatic video tracking;
  - detecting the ball in the video, which we could neither tune nor check on the provided video
    under the rules, and which needs CPU inference over 199,319 frames;
  - fixed-window smoothing, the organisers' baseline, which we kept as the reference.

## 2. Assets used and dropped

| Asset | Used? | How, or why not |
|---|---|---|
| Tactical automatic tracking (ATD) | **Yes** | Player and ball positions are the only model input, converted to metres from the centre spot |
| ATD metadata | **Yes** | Which team each anonymous track slot belongs to |
| Half boundaries (`games.py`) | **Yes** | Each half is decoded separately; outside the halves we write D, which is never scored |
| Smart Tagging | Not in v1 | Planned as fixed football rules (a restart ends a dead spell; the restarting team gets the ball). No external game has Smart Tagging to tune its weight on, and it was not finished in time |
| Homography | Dropped | Camera aim as a stand-in for the ball when it is missing: no external data to tune it on |
| Tactical video (image and audio) | Dropped | A pretrained ball detector or a whistle detector would need its thresholds checked or calibrated on the provided video, which the rules forbid |
| Broadcast video and tracking | Dropped | Not synchronised with the tactical timeline, and adds nothing tunable on external data |
| Public 5-minute ground truth | Not for any decision | Only ever scored as a final check that the output file is wired correctly (for example that the teams are not swapped). No parameter, feature or version choice was made from it |

## 3. Training data

**IDSSE** (Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", *Scientific Data* 12, 195, 2025; figshare article 28196177; CC BY 4.0).

- **How much:** 7 matches from the 2022/23 Bundesliga and 2. Bundesliga, 1,002,644 labelled
  frames (11.1 hours at 25 fps), 3,061 possessions.
- **Who labelled it:** nobody on our team. The labels are the `BallStatus` and `BallPossession`
  fields of the official DFL position-data feed, produced by the league's data provider.
- **How we converted them:** `BallStatus = 0` gives D; otherwise `BallPossession` 1 or 2 gives H or A.
  This is the same kind of derivation as the challenge's ground truth: dead where the tracking marks
  the ball dead, otherwise the owning team.
- **Nothing comes from the provided game.**

**Making it look like automatic tracking.** IDSSE sees the ball on every frame; the provided
tracking misses it on 41% of graded frames (DATA.md). Before fitting or scoring, every IDSSE game is
degraded:
- the ball hidden in bursts on 20%, 40% or 60% of frames (mean gap 2 s);
- temporally correlated position error (AR(1), ρ = 0.9; ball 0.3 m, players 0.15 m);
- 5% of player observations dropped, in 1 s bursts.

These ranges bracket the documented 41% rather than being measured on the provided game, and
every random draw is seeded.

## 4. Possession rules we implemented

We do not hand-code a possession definition. The labels define it, and the three costs decide where
the line falls. The numbers below describe the frozen model's behaviour on the IDSSE games, degraded
with 40% of the ball hidden. They are descriptive (the model was fitted on these games), not a score.

- **Change of possession.** A switch to the other team costs 7 (in log-probability units), so a
  flicker to the other team and back costs 14. A touch, a deflection, or a pass flying past an
  opponent is absorbed unless the evidence for the other team adds up to more than that. On
  IDSSE, 95% of our possessions last longer than 1.2 s, against 0.5 s in the labels. The model
  deliberately keeps fewer possessions (1,733 against 3,061), merging brief touches into the
  surrounding possession.
- **Loose and contested balls.** When nobody is near the ball, or the ball is not seen, the team
  evidence falls back to team shape and is weak, so the decoder keeps the last team. This matches the
  challenge's rule that possession stays with the last team while the ball is loose.
- **Challenges.** The winner's team becomes and stays nearest to the ball, which is sustained
  evidence, so the switch is placed where that evidence begins.
- **Dead ball and restarts.** Going dead costs 10; restarting costs 80. A dead spell therefore ends
  only on sustained evidence: the ball moving and back inside the lines, players speeding up. After a
  restart, the team is whichever the evidence supports, usually the player nearest the ball. On
  IDSSE this finds 701 dead spells against 713 true ones, with a median length of 19.9 s against
  19.6 s. There are no explicit restart-type rules in this version.

**What the evidence relies on** (largest standardised weights of the frozen regressions):
- **Team:**
  - difference in team width (the side with the ball spreads out);
  - nearest-player distance difference, both for close control within 2 m and averaged over 4 s;
  - everyone's movement towards one goal.
- **Dead ball:**
  - a slow ball;
  - the ball beyond a touchline or goal line;
  - no player near the ball;
  - players moving slowly over the last 4 to 10 s.

There are 11 team features and 14 dead-ball features, all listed in `possession/features.py`.

## 5. Results, all on external data

Leave one game out on IDSSE: fit and tune on six games, score the seventh with the organisers'
scorer (`evaluation/possession_metrics.py`), for each game in turn. Same degraded games for both
rows.

| Method | spellF1 (held out) | macroF1 | Ball hidden 20% / 40% / 60% |
|---|---|---|---|
| Organisers' baseline, window chosen the same way | 0.222 ± 0.013 | 0.382 | 0.242 / 0.223 / 0.202 |
| **This HMM** | **0.417 ± 0.028** | **0.850** | 0.441 / 0.415 / 0.394 |

The HMM wins on all seven games and at every noise level. The choice of costs is flat: widening the
search from 80 to 150 settings changed the held-out score by 0.0002. We have no score on the
provided match, by design.

## 6. Limitations, and what we would do with more time

1. **Label definition.** IDSSE's labels change on brief touches (mean possession 7.2 s), while the
   public excerpt averages 14.2 s over 17 possessions. Our model merges short possessions, which
   helps if the provided match's possessions are long and costs if they are short. We could not
   check which, by design.
2. **Realism of the noise.** Our missing ball is random. In real automatic tracking it is tied to
   occlusion, high balls and the ball leaving the pitch, so cues like "ball not seen" may behave
   differently on the provided match. Next: a second external dataset with genuine detection gaps
   (SkillCorner's open broadcast tracking) as an independent check.
3. **Unused signals.** Smart Tagging restarts as fixed rules, filling in the ball across short gaps,
   and the homography as a ball stand-in.
4. **Model form.** An HMM assumes possession lengths are geometric; a semi-Markov model would
   model them directly. A non-linear emission model (gradient-boosted trees, kept only if it wins
   leave-one-game-out) could capture interactions such as "close to the ball *and* facing goal".

## 7. Reproduction

```bash
pip install -r requirements-possession.txt     # numpy 2.4.6, lxml 6.1.3, pandas 2.3.3, pinned
python -m possession.barca                     # read the provided tactical tracking
python -m possession.predict                   # -> evaluation/results/hmm_v1.csv
```

`MTAH_DATA_DIR` points at the data bundle, as for the starter code. It runs on CPU in about two
minutes, with no training and no external data, because the model is committed.

To retrain from scratch (2.6 GB download, about five minutes):

```bash
python -m possession.fetch_idsse && python -m possession.idsse && python -m possession.train
```

The leave-one-game-out test is `python -m possession.cv hmm --seeds 0`.

**Determinism:**
- no randomness at prediction time;
- regressions fitted by Newton's method;
- dot products summed in a fixed order;
- log-probabilities rounded to six decimals before Viterbi, so last-bit differences between
  machines' maths libraries cannot change a label.

**Rules:**
- no hand labels on the provided game;
- nothing trained, tuned or checked against its ground truth or video;
- all training and tuning on IDSSE;
- the match data is not in the repository.
