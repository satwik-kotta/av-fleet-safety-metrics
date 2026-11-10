# AV Fleet Safety Metrics

**Question.** Every company testing self driving cars in California must file each disengagement, the miles each test vehicle drove and every collision with the DMV. A V&V team would want three things from these filings:
- a fleet metric they can trust
- a way to label why each disengagement happened
- a warning when an apparent trend is really a change in how a company reports

[Dashboard](https://satwik-kotta.github.io/av-fleet-safety-metrics/) · [One page memo](https://satwik-kotta.github.io/av-fleet-safety-metrics/memo.html)

## What it found

- **The metric needs an interval.** Zoox's 2023 rate of 177,602 miles per disengagement rests on 4 events, and its exact 95% interval runs from 69K to 652K. Its 2025 rate of 60,682 rests on 20 events, so its interval is much tighter (39K to 99K).
- **Waymo's 2024 rise looks real.** Waymo's disengagement rate rose 1.77x in 2024 (exact test p < 0.001), and its cause sentences were unchanged from 2023. That makes the shift toward planning causes (+31 points) look like a real change in driving. Several other large jumps, at aiMotive, Imagry, Apple, Mercedes-Benz and Valeo, came with mostly new wording, which points to a change in how those companies filed.
- **A random split overstates the classifier.** A cause classifier scored **0.999 macro F1 on a random split**, because 98% of the test sentences were already in training. On companies it never saw, it scored **0.72**. Ten boilerplate sentences make up 43% of all 28,518 events.
- **Similarity search finds related events.** For each sentence, the 5 nearest sentences from other companies share its cause label 53% of the time, against 29% by chance.
- **Collisions differ by mode.** 60% of collisions in autonomous mode were rear end crashes, against 36% in conventional mode, and the test vehicle was at fault in 10% against 28%. The difference in crash type is significant (chi-square p < 0.001, Cramér's V 0.26).
- **What causes Zoox's disengagements.** Of 100 Zoox disengagements, 41 were described as prediction issues, 38 as planning and 21 as perception. 21% mention a pedestrian or cyclist.

## How it works

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

## Limits

- **Cross company comparisons are rough.** Each company decides what counts as a disengagement and how to describe it.
- **The rules have no hand labeled ground truth yet.** The held out score measures how well the model generalizes the rules to new wording, not how often the rules are right.
- **Driverless operations are missing.** They are reported under a different DMV permit and are not in this data.
- **Collision reports compare crash types, not crash rates.** They end in mid 2024 and carry no mileage by mode.

## Run it

```bash
pip install -r requirements.txt
bash scripts/get_data.sh       # copies the DMV filings into data/raw
cd src && python -m avm.run    # writes docs/data/results.json and outputs/*.csv
cd .. && pytest -q tests
```

Data: California DMV Autonomous Vehicle Testing Program, disengagement and mileage reports (2021 to 2025) and collision reports (OL 316). Compiled copies come from [caldanayabur/Autonomous-Vehicle-Disengagement-Analysis](https://github.com/caldanayabur/Autonomous-Vehicle-Disengagement-Analysis) and [lindalim478/CA_DMV_CollisionsPaper](https://github.com/lindalim478/CA_DMV_CollisionsPaper).
