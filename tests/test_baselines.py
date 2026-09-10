import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baselines import RUNS_CSV_COLUMNS, log_run, persistence_predict  # noqa: E402


def test_persistence_predict_is_identity():
    series = pd.Series([1.0, 2.0, 3.0])
    result = persistence_predict(series)
    assert list(result) == [1.0, 2.0, 3.0]
    # Should be a copy, not the same object, so callers can't mutate the input by accident.
    assert result is not series


def test_log_run_writes_header_once_and_matches_schema(tmp_path):
    csv_path = tmp_path / "runs.csv"

    log_run(
        str(csv_path),
        model="Persistence",
        dataset="Basel_Tier1",
        metrics={"precision": 0.837, "recall": 0.837, "f1": 0.837,
                 "peirce_skill": 0.768, "rmse": 2.352, "mae": 1.799},
    )
    log_run(
        str(csv_path),
        model="LogisticRegression",
        dataset="Basel_Tier1",
        metrics={"precision": 0.6, "recall": 0.5, "f1": 0.55, "peirce_skill": 0.4},
    )

    df = pd.read_csv(csv_path)
    assert list(df.columns) == RUNS_CSV_COLUMNS
    assert len(df) == 2
    # Second run had no rmse/mae -> should be blank, not a fabricated 0.
    assert pd.isna(df.loc[1, "rmse"])
