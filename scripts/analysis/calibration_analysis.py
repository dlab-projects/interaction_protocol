"""Evaluate held-out calibration for the paper's multinomial model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .heldout_accuracy import (
        DEFAULT_OUTPUT_DIR,
        VERDICT_NAMES,
        chosen_model_name,
        fit_model_on_mask,
        load_model_matrices,
        make_folds,
        model_config,
        predict,
    )
except ImportError:  # Support direct execution by file path.
    from heldout_accuracy import (
        DEFAULT_OUTPUT_DIR,
        VERDICT_NAMES,
        chosen_model_name,
        fit_model_on_mask,
        load_model_matrices,
        make_folds,
        model_config,
        predict,
    )


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Encode integer verdict labels as one-hot rows."""
    encoded = np.zeros((len(y), n_classes), dtype=float)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """Compute one-vs-rest calibration bins for every verdict class."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if probabilities.shape != (len(y_true), len(VERDICT_NAMES)):
        raise ValueError("probabilities must have one column per verdict")

    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    observed_indicators = one_hot(y_true, probabilities.shape[1])
    for class_idx, verdict in enumerate(VERDICT_NAMES):
        predicted = probabilities[:, class_idx]
        observed = observed_indicators[:, class_idx]
        # right=True assigns exact boundary values consistently, including 1.0
        # in the final bin. Clipping keeps probability 0.0 in the first bin.
        bin_indices = np.clip(
            np.digitize(predicted, edges[1:-1], right=True),
            0,
            n_bins - 1,
        )
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                continue
            mean_probability = float(predicted[mask].mean())
            observed_frequency = float(observed[mask].mean())
            rows.append(
                {
                    "verdict": verdict,
                    "bin": bin_idx + 1,
                    "bin_low": float(edges[bin_idx]),
                    "bin_high": float(edges[bin_idx + 1]),
                    "n": int(mask.sum()),
                    "mean_predicted_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "abs_calibration_error": abs(
                        observed_frequency - mean_probability
                    ),
                }
            )
    return pd.DataFrame(rows)


def classwise_expected_calibration_error(bin_frame: pd.DataFrame) -> float:
    """Return pooled one-vs-rest ECE across all verdict classes."""
    if bin_frame.empty:
        raise ValueError("At least one populated calibration bin is required")
    total = bin_frame["n"].sum()
    return float(
        (bin_frame["n"] * bin_frame["abs_calibration_error"]).sum() / total
    )


def parse_args() -> argparse.Namespace:
    """Parse cross-validation, calibration, and output settings."""
    parser = argparse.ArgumentParser(
        description="Held-out calibration for the paper multinomial model."
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--split-unit",
        choices=["debate", "dilemma", "row"],
        default="debate",
        help=(
            "Hold out experiment-specific debates (default), entire dilemmas, "
            "or individual rows."
        ),
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--debate-random-intercept", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Maximum L-BFGS iterations per fold.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2332)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "heldout_calibration_predictions.csv",
    )
    parser.add_argument(
        "--bins-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "heldout_calibration_bins.csv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "heldout_calibration_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Cross-validate probabilities and save calibration artifacts."""
    args = parse_args()
    matrices = load_model_matrices()
    config = model_config(args)
    model_name = chosen_model_name(args.debate_random_intercept)

    prediction_frames = []
    for fold_idx, train_mask, test_mask in make_folds(
        matrices, args.folds, args.seed, args.split_unit
    ):
        print(f"Fitting fold {fold_idx}/{args.folds}...", flush=True)
        model = fit_model_on_mask(
            matrices,
            train_mask,
            args.debate_random_intercept,
            config,
        )
        probabilities = predict(
            model, matrices, test_mask, args.device, args.batch_size
        )
        y_true = matrices.y[test_mask]
        predicted_labels = probabilities.argmax(axis=1)
        frame = pd.DataFrame(
            {
                "fold": fold_idx,
                "row_index": np.flatnonzero(test_mask),
                "debate_idx": matrices.debate_idx[test_mask],
                "dilemma_idx": matrices.dilemma_idx[test_mask],
                "model_idx": matrices.model_idx[test_mask],
                "round_idx": matrices.round_idx[test_mask],
                "true_label": y_true,
                "true_verdict": [VERDICT_NAMES[index] for index in y_true],
                "predicted_label": predicted_labels,
                "predicted_verdict": [
                    VERDICT_NAMES[index] for index in predicted_labels
                ],
            }
        )
        for class_idx, verdict in enumerate(VERDICT_NAMES):
            frame[f"p_{verdict}"] = probabilities[:, class_idx]
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    probability_columns = [f"p_{verdict}" for verdict in VERDICT_NAMES]
    probabilities = predictions[probability_columns].to_numpy()
    y_true = predictions["true_label"].to_numpy(dtype=np.int64)
    observed_indicators = one_hot(y_true, len(VERDICT_NAMES))
    selected_probabilities = probabilities[np.arange(len(y_true)), y_true]

    bin_frame = calibration_bins(y_true, probabilities, args.bins)
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "split_unit": args.split_unit,
                "n": len(predictions),
                "accuracy": float(
                    (predictions["predicted_label"].to_numpy() == y_true).mean()
                ),
                "log_loss": float(
                    -np.log(np.clip(selected_probabilities, 1e-12, 1.0)).mean()
                ),
                "multiclass_brier": float(
                    np.square(probabilities - observed_indicators).sum(axis=1).mean()
                ),
                "classwise_expected_calibration_error": (
                    classwise_expected_calibration_error(bin_frame)
                ),
                "n_bins": args.bins,
            }
        ]
    )

    for path in (
        args.predictions_output,
        args.bins_output,
        args.metrics_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    bin_frame.to_csv(args.bins_output, index=False)
    metrics.to_csv(args.metrics_output, index=False)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {args.predictions_output}")
    print(f"Wrote {args.bins_output}")
    print(f"Wrote {args.metrics_output}")


if __name__ == "__main__":
    main()
