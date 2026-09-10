# Results vs. baselines vs. literature

Source of truth for the numbers below is `experiments/runs.csv` (append-only
run log). This table is updated at the end of each phase — Phase 1 is
filled in; Phase 2/3 rows are placeholders until those notebooks exist.

## Phase 1 — Baselines (Basel, Tier 1, 1-day horizon, `frost_label_broad`)

| Model | Precision | Recall | F1 | Peirce skill | RMSE (°C) | MAE (°C) |
|---|---|---|---|---|---|---|
| Persistence | 0.842 | 0.842 | 0.842 | 0.771 | 2.43 | 1.87 |
| Physical/threshold (linear, physically-restricted features) | 0.858 | 0.846 | 0.852 | 0.783 | 2.35 | 1.81 |
| Logistic regression (class-weighted, direct classifier) | 0.812 | **0.947** | 0.874 | 0.848 | — | — |

*(An earlier, slightly different Persistence run — 0.837/0.837/0.837 — is
also in `runs.csv` from before the pipeline was refactored onto
`src/label_rules.py`/`src/data_loader.py`; the two differ marginally
because of the switch to the centralized loader, not a methodology
change.)*

**Reading this, honestly:** on this particular slice of data, the
class-weighted logistic-regression classifier actually got *higher*
recall than both continuous-then-threshold baselines, which is the
*opposite* of the pattern Omazić et al. (2024) found with their
class-weighted XGBoost (POD 0.81 vs. a plain threshold's POD 0.98). That's
worth reporting as-is rather than forced to match the literature —
possible reasons worth checking in Phase 3 (not concluded yet): Basel's
class imbalance under this threshold is much milder (~3.4:1, 29.6%
positive — see `01_eda.ipynb`) than a true rare-frost setting, and the
physical/threshold baseline here uses far fewer, hand-picked features than
Omazić et al.'s full XGBoost model. Re-check this comparison once Tier 2/3
data (rarer, more literature-comparable class balance) and the full
tree-ensemble models are in.

**Primary approach going forward:** regression-then-threshold (Persistence,
Physical/threshold), per the project's core design decision — logistic
regression stays the documented classifier ablation, not the headline
model, regardless of which one currently scores best on this one city and
threshold.

## Phase 2 — Feature-engineered models

*(Not yet run — `04_feature_engineering.ipynb` not started.)*

## Phase 3 — RF / XGBoost / LightGBM / CatBoost

*(Not yet run — `05_model_training.ipynb` not started.)*

## Literature reference points (not directly comparable — different
data, regions, horizons, and label definitions; positioning only, not a
reproduction claim)

| Source | Result |
|---|---|
| Chile (RF) | ~90% accuracy |
| Massachusetts (RF) | 91–95% accuracy; Peirce skill 0.88 vs. Franklin-model baseline 0.68 |
| Korea (RF/SVM) | (see Noh et al. 2021 for exact figures once Tier 2 is attempted) |
| FRUTILLA | MAE 1.4–3.3°C |
| Omazić et al. (2024), Croatia — Method 7 (`Tmin3_Td0`) threshold rule | POD 0.98, POFD 0.18 (all-year) |
| Omazić et al. (2024), Croatia — XGBoost, class-weighted classifier | POD 0.81, POFD 0.06 |
