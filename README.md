# AV Fleet Safety Metrics

**Question.** Every company testing self driving cars in California must file each disengagement, the miles each test vehicle drove and every collision with the DMV. A V&V team would want three things from these filings:
- a fleet metric they can trust
- a way to label why each disengagement happened
- a warning when an apparent trend is really a change in how a company reports

Fleet filings only say how often a disengagement happened. **Part 2** goes down to the driving itself: 3.1M vehicle trajectory records from US-101 in Los Angeles, used to measure near misses, find the lane changes that carry the risk, and test a lane change model as traffic shifts.

Part 1: [Dashboard](https://satwik-kotta.github.io/av-fleet-safety-metrics/) · [Memo](https://satwik-kotta.github.io/av-fleet-safety-metrics/memo.html)  
Part 2: [Dashboard](https://satwik-kotta.github.io/av-fleet-safety-metrics/freeway.html) · [Memo](https://satwik-kotta.github.io/av-fleet-safety-metrics/freeway_memo.html)

## Part 1: what the DMV filings show

- **The metric needs an interval.** Zoox's 2023 rate of 177,602 miles per disengagement rests on 4 events, and its exact 95% interval runs from 69K to 652K. Its 2025 rate of 60,682 rests on 20 events, so its interval is much tighter (39K to 99K).
- **Waymo's 2024 rise looks real.** Waymo's disengagement rate rose 1.77x in 2024 (exact test p < 0.001), and its cause sentences were unchanged from 2023. That makes the shift toward planning causes (+31 points) look like a real change in driving. Several other large jumps, at aiMotive, Imagry, Apple, Mercedes-Benz and Valeo, came with mostly new wording, which points to a change in how those companies filed.
- **A random split overstates the classifier.** A cause classifier scored **0.999 macro F1 on a random split**, because 98% of the test sentences were already in training. On companies it never saw, it scored **0.72**. Ten boilerplate sentences make up 43% of all 28,518 events.
- **Similarity search finds related events.** For each sentence, the 5 nearest sentences from other companies share its cause label 53% of the time, against 29% by chance.
- **Collisions differ by mode.** 60% of collisions in autonomous mode were rear end crashes, against 36% in conventional mode, and the test vehicle was at fault in 10% against 28%. The difference in crash type is significant (chi-square p < 0.001, Cramér's V 0.26).
- **What causes Zoox's disengagements.** Of 100 Zoox disengagements, 41 were described as prediction issues, 38 as planning and 21 as perception. 21% mention a pedestrian or cyclist.

## Part 1: how it works

1. **Clean** (`src/avm/clean.py`).
   - Company names arrive in 67 spellings, who started the disengagement in 13, and road type in 22. Each is mapped to one value, and an unknown company fails loudly.
   - Broken characters in the free text are repaired.
   - One company filed dates as numbers like 20230308, and these are parsed.
   - 47 blank annual mileage totals are rebuilt from the monthly columns.
   - After cleaning, the event rows match the mileage file's disengagement totals in all 111 company years.
2. **Metrics** (`src/avm/metrics.py`).
   - Miles per disengagement for every company year, overall and for disengagements the AV system started itself.
   - Exact Garwood Poisson intervals.
   - An exact conditional binomial test for year over year changes.
   - How concentrated disengagements are across vehicles.
3. **Labels** (`src/avm/labels.py`).
   - Readable rules on the free text assign each event one of 7 cause categories: perception, prediction, planning, map and localization, hardware, software and system fault, or other road users and conditions.
   - They also tag driving situations: lane change, turn, intersection, pedestrian or cyclist, pull over or parking, lane keeping, construction, emergency vehicle.
   - A clause after "root cause", "caused by" or "due to" decides. Consequence clauses ("which resulted in a motion plan...") are ignored.
   - 30% of events name more than one category. This is the label noise the model has to live with.
4. **Text model** (`src/avm/textmodel.py`).
   - TF-IDF and logistic regression, trained on 598 distinct sentences that have one clear rule label.
   - Evaluated two ways: a naive random split of events, and five folds that never put a company in both training and test.
   - Similarity search uses TF-IDF reduced to 128 dimensions with SVD and indexed in FAISS.
5. **Drift** (`src/avm/drift.py`). For each company and year it measures PSI on the cause mix and the share of cause sentences the company never used the year before. A large PSI with mostly new wording is flagged as a reporting change.
6. **Collisions** (`src/avm/collisions.py`). 707 OL 316 reports. Chi-square tests compare collision type and the test vehicle's movement between autonomous and conventional mode.

## Part 2: freeway trajectories (NGSIM US-101)

Every vehicle on a 640 m stretch of US-101, tracked ten times a second from 7:50 to 8:35 on June 15, 2005, as the morning rush builds.

| | 7:50 to 8:05 | 8:05 to 8:20 | 8:20 to 8:35 |
|---|---|---|---|
| Mean speed (m/s) | 11.7 | 9.0 | 7.6 |
| Vehicle km | 1,207 | 929 | 778 |
| Near misses per 1,000 vehicle km | 191 | 320 | 531 |

- **Near misses rose 2.8x per km** from the first period to the last (exact conditional test, p < 0.001). Hard braking rose 20% (p < 0.001). Lane changes per km did not change significantly (p = 0.07).
- **5% of lane changes carry most of the risk.** A Gaussian mixture (4 clusters, chosen by BIC) found a cluster of slow drifts across the line into a tight gap. 78% of them left a time to collision under 2 s, against 6% overall.
- **Similar manoeuvre search.** Each lane change is a 26 number snippet of lateral position and speed, indexed in FAISS. For risky queries, precision@10 is 21%, 3.4x the base rate.
- **The split decides the score, again.** Predicting a lane change 3 s ahead (LightGBM, 608,822 samples): 0.63 average precision on a random split where every test car was in training, 0.50 with cars held out (base rate 1.5%).
- **Drift.** Speed PSI from first to last period is 1.25. The model trained on the first period scores 0.55 and 0.60 on the later two, and 0.63 on the last after retraining with the middle period.

Data checks:
- Each published file stops at exactly **1,048,575 rows, the Excel row limit**. 3 vehicles cut mid trajectory are dropped; 8,353 records whose leader is one of 25 missing vehicles get no gap or TTC.
- 237 of 1,441 raw lane switches (16%) were undone within 1 s and are treated as tracking jitter.
- 615 hard braking runs contain accelerations over 8 m/s² and are rejected as tracking noise.

| Step | File |
|---|---|
| PySpark ingest: units, duplicate and gap checks, truncation, leader join for gap, closing speed and TTC, Parquet output | `src/fdb/ingest.py` |
| Event detection with minimum durations, exact Poisson intervals, rate ratio tests | `src/fdb/events.py` |
| Lane change features, Gaussian mixture clusters, FAISS search | `src/fdb/behavior.py` |
| Lane change model, leaky vs grouped vs over time evaluation, PSI | `src/fdb/predict.py` |
| Same ingest as a Databricks notebook, Delta table and Spark SQL rates | `notebooks/databricks_pipeline.py` |

Definitions: a lane change is a switch between mainline lanes 1 to 5 not undone within 1 s. Hard braking is below −3 m/s² for at least 0.5 s with no reading beyond 8 m/s². A near miss is a TTC under 1.5 s to a leader in the same lane for at least 0.3 s.

## Limits

- **Cross company comparisons are rough.** Each company decides what counts as a disengagement and how to describe it.
- **The rules have no hand labeled ground truth yet.** The held out score measures how well the model generalizes the rules to new wording, not how often the rules are right.
- **Driverless operations are missing.** They are reported under a different DMV permit and are not in this data.
- **Collision reports compare crash types, not crash rates.** They end in mid 2024 and carry no mileage by mode.
- **The trajectories are one stretch of one freeway on one morning in 2005**, tracked from video, with human drivers. The method carries over to fleet logs; the numbers do not.

## Run it

```bash
pip install -r requirements.txt
bash scripts/get_data.sh       # copies the DMV filings into data/raw
cd src && python -m avm.run    # writes docs/data/results.json and outputs/*.csv
cd .. && pytest -q tests

# Part 2
bash scripts/get_trajectories.sh   # NGSIM US-101, smoothed version, into data/ngsim/raw
cd src && python -m fdb.ingest     # PySpark, writes data/ngsim/frames.parquet
python -m fdb.run                  # writes docs/data/freeway.json and outputs/freeway/*.csv
```

Data: California DMV Autonomous Vehicle Testing Program, disengagement and mileage reports (2021 to 2025) and collision reports (OL 316). Compiled copies come from [caldanayabur/Autonomous-Vehicle-Disengagement-Analysis](https://github.com/caldanayabur/Autonomous-Vehicle-Disengagement-Analysis) and [lindalim478/CA_DMV_CollisionsPaper](https://github.com/lindalim478/CA_DMV_CollisionsPaper).

NGSIM US-101, Federal Highway Administration, public domain, in the Savitzky-Golay smoothed release (window 11) from [Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing](https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing).
