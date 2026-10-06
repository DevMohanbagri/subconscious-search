# Non-invasive diabetes risk stratification (BRFSS 2015)

Screening-triage pipeline: rough-set feature selection -> calibrated MLP -> fuzzy risk tiers -> equity audit.
Not an early-detection or diagnostic tool: the data are cross-sectional and the label is self-reported.

## Rules
1. The test split is evaluated once, by 99_final_test.py, after the `freeze` tag. No other script requests it.
2. Scalers, bins, reducts, MI scores and LASSO paths are fitted on training rows only.
   Calibrators and thresholds are fitted on validation, never on test.
3. results/registry.csv is append-only. Never edit a row by hand.
4. Commit before running anything that logs. Rows tagged -dirty are not citable.

## Rebuild from scratch
1. python -m venv .venv, activate it, pip install -r requirements.txt
   (CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu)
2. python 00_fetch_data.py, then python 01_stage0_split.py
3. The printed hash must be fc3774b0bdf6c0a16e80c7e518cd787d
4. python 03_gamma_check.py and python tests_fuzzy.py must pass

## Data
UCI #891 CDC Diabetes Health Indicators (BRFSS 2015): 253,680 rows; 229,474 after removing 24,206 exact
duplicates. Prevalence 0.1529. Split 70/15/15, stratified on the label, seed 42.
