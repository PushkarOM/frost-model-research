"""
export_model.py — serialize the chosen Phase 3 model for Repo B.

Writes two files per export:
  - <version>_model.cbm      : CatBoost's native binary format. Loadable
                                with CatBoostRegressor().load_model(path) --
                                the only runtime dependency is the
                                `catboost` package itself, avoiding the
                                pickle/joblib cross-version fragility a
                                raw Python object dump would carry.
  - <version>_metadata.json  : everything Repo B needs to actually use the
                                model correctly, spelled out rather than
                                left for someone to reverse-engineer from
                                the binary: exact feature list and order,
                                the classification threshold, which label
                                variant it targets, a training-data
                                description, and the metrics it achieved.

NOT yet resolved here, flagged rather than silently decided:
  - Whether CatBoost's native format (and its C++ runtime) is actually
    practical on a Raspberry Pi has NOT been verified. The project brief
    explicitly defers latency/memory benchmarking until real hardware is
    available. If CatBoost proves too heavy for the Pi, ONNX export
    (estimator.save_model(path, format="onnx")) is a documented,
    ready-to-use fallback that doesn't require retraining.
  - This exports whichever model Tier-1 (European) results selected. It
    is a first-cut artifact, not a final one -- Tier 3 (Indian) regional
    calibration is still pending, and the model and/or threshold may
    change once that data exists.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


def export_catboost_model(
    model: Any,
    feature_columns: List[str],
    output_dir: str,
    version: str,
    classification_threshold: float,
    label_variant: str,
    training_description: str,
    eval_metrics: Dict[str, float],
    extra_known_limitations: List[str] = None,
) -> Dict[str, str]:
    """
    Save a fitted CatBoost-backed model (accepts either a
    models.TreeModelWrapper or a raw CatBoostRegressor) plus a metadata
    sidecar describing how to use it correctly.

    `extra_known_limitations`: situation-specific caveats for THIS export
    (e.g. "trained with one-hot city columns that won't exist at a single
    deployment site") appended after the standing, always-true ones below
    -- callers should use this rather than this function silently staying
    generic when something exportspecific and important is true.

    Returns {"model_path": ..., "metadata_path": ...}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap TreeModelWrapper if given one -- CatBoost's own save_model()
    # only exists on the underlying estimator, not the wrapper.
    estimator = getattr(model, "estimator", model)

    model_path = os.path.join(output_dir, f"{version}_model.cbm")
    estimator.save_model(model_path)

    known_limitations = [
        "Trained and evaluated on Tier-1 European data (Basel, Oslo, "
        "Perpignan, De Bilt), NOT Indian data -- see the project's "
        "India-bridge notes before treating this as deployment-ready "
        "for Warora/Vidarbha.",
        "Not yet benchmarked for latency/memory on the actual target "
        "hardware (Raspberry Pi) -- deferred per the project brief "
        "until hardware is available.",
        "Hyperparameters were chosen via a modest, non-exhaustive "
        "search (see 05_model_training.ipynb); tuning did not clearly "
        "outperform CatBoost's own defaults in that search.",
    ]
    if extra_known_limitations:
        known_limitations.extend(extra_known_limitations)

    metadata = {
        "version": version,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": type(estimator).__name__,
        "model_format": "cbm",
        "predicts": "continuous minimum temperature (deg C) over the configured horizon",
        "feature_columns_in_order": feature_columns,
        "classification_threshold_c": classification_threshold,
        "classification_rule": (
            f"frost/cold-injury risk if predicted_min_temp_c <= {classification_threshold}"
        ),
        "label_variant_targeted": label_variant,
        "training_description": training_description,
        "eval_metrics": {k: round(float(v), 4) for k, v in eval_metrics.items()},
        "known_limitations": known_limitations,
    }
    metadata_path = os.path.join(output_dir, f"{version}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Convenience loader -- Repo B (or a test) reading back what a model export declared."""
    with open(metadata_path) as f:
        return json.load(f)
