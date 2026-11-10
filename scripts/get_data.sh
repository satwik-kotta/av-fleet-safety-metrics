#!/usr/bin/env bash
# Copies the California DMV autonomous vehicle filings into data/raw/.
# The DMV publishes the disengagement and mileage reports and the OL 316 collision
# reports at dmv.ca.gov. This script takes compiled copies from two public GitHub
# repositories so the run is reproducible offline.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/src
[ -d data/src/dis ] || git clone --depth 1 https://github.com/caldanayabur/Autonomous-Vehicle-Disengagement-Analysis.git data/src/dis
[ -d data/src/col ] || git clone --depth 1 https://github.com/lindalim478/CA_DMV_CollisionsPaper.git data/src/col
cp "data/src/dis/data/raw/2021-2025_AV_Disengagements_and_Mileage_Report.xlsx" data/raw/dmv_disengagements_mileage_2021_2025.xlsx
cp data/src/col/CA_DMV_CollisionsDataset.csv data/raw/dmv_collisions.csv
ls -la data/raw
