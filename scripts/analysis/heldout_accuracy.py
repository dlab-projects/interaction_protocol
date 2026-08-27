"""Evaluate held-out verdict accuracy for the paper's multinomial model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from interaction_protocol.data import load_debate_data
from interaction_protocol.models import (
    DEBATE_RANDOM_INTERCEPT_SPEC,
    MAIN_MODEL_NAME,
    ComparisonModel,
    DebateExperiment,
    ModelMatrices,
    PaperModelConfig,
    fit_debate_random_intercept_model,
    fit_paper_comparison_model,
    preprocess,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis"
VERDICT_NAMES = ("NTA", "YTA", "NAH", "ESH", "INFO")

# These are the same experiments and stable model indices used by the canonical
# runner in scripts/models/fit_models.py: GPT=0, Claude=1, Gemini=2.
SYNCHRONOUS_EXPERIMENTS = (
    ("sync_h2h_cla_vs_gpt.parquet", (1, 0)),
    ("sync_h2h_gpt_vs_gem.parquet", (0, 2)),
    ("sync_h2h_cla_vs_gem.parquet", (1, 2)),
)
ROUND_ROBIN_EXPERIMENTS = (
    ("round_robin_h2h_gem_vs_gpt.parquet", (2, 0)),
    ("round_robin_h2h_gpt_vs_gem.parquet", (0, 2)),
    ("round_robin_h2h_cla_vs_gem.parquet", (1, 2)),
    ("round_robin_h2h_gem_vs_cla.parquet", (2, 1)),
    ("round_robin_h2h_gpt_vs_cla.parquet", (0, 1)),
    ("round_robin_h2h_cla_vs_gpt.parquet", (1, 0)),
    ("round_robin_3way_gem_cla_gpt.parquet", (2, 1, 0)),
    ("round_robin_3way_cla_gem_gpt.parquet", (1, 2, 0)),
    ("round_robin_3way_gpt_cla_gem.parquet", (0, 1, 2)),
    ("round_robin_3way_gpt_gem_cla.parquet", (0, 2, 1)),
    ("round_robin_3way_gem_gpt_cla.parquet", (2, 0, 1)),
    ("round_robin_3way_cla_gpt_gem.parquet", (1, 0, 2)),
)


def load_model_matrices(data_dir: Path = DATA_DIR) -> ModelMatrices:
    """Load the primary Parquet experiments and build canonical model inputs."""

    def load(specifications):
        return [
            DebateExperiment(
                load_debate_data(data_dir / filename), tuple(model_indices)
            )
            for filename, model_indices in specifications
        ]

    return preprocess(
        load(SYNCHRONOUS_EXPERIMENTS),
        load(ROUND_ROBIN_EXPERIMENTS),
    )


def chosen_model_name(debate_random_intercept: bool = False) -> str:
    """Return the registered name of the model being cross-validated."""
    if debate_random_intercept:
        return DEBATE_RANDOM_INTERCEPT_SPEC.name
    return MAIN_MODEL_NAME


def make_folds(
    matrices: ModelMatrices,
    n_folds: int,
    seed: int,
    split_unit: str,
):
    """Yield deterministic train/test masks grouped by the requested unit."""
    if split_unit == "row":
        units = np.arange(matrices.n_observations)
        row_units = units
    else:
        unit_values = {
            "debate": matrices.debate_idx,
            "dilemma": matrices.dilemma_idx,
        }
        if split_unit not in unit_values:
            raise ValueError("split_unit must be debate, dilemma, or row")
        row_units = unit_values[split_unit]
        units = np.unique(row_units)
    if not 2 <= n_folds <= len(units):
        raise ValueError("n_folds must be between 2 and the number of split units")

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(units), n_folds)
    for fold_idx, test_units in enumerate(folds, start=1):
        test_mask = np.isin(row_units, test_units)
        yield fold_idx, ~test_mask, test_mask


def fit_model_on_mask(
    matrices: ModelMatrices,
    train_mask: np.ndarray,
    debate_random_intercept: bool,
    config: PaperModelConfig,
) -> ComparisonModel:
    """Fit the canonical MAP objective while retaining all held-out ID levels."""
    fitter = (
        fit_debate_random_intercept_model
        if debate_random_intercept
        else fit_paper_comparison_model
    )
    fit = fitter(
        matrices,
        config,
        penalized=True,
        observation_mask=train_mask,
    )
    if not isinstance(fit.model, ComparisonModel):
        raise TypeError("Expected the canonical comparison-model implementation")
    return fit.model


def predict(
    model: ComparisonModel,
    matrices: ModelMatrices,
    mask: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predict probabilities for selected observations without refitting."""
    model_idx = torch.as_tensor(
        matrices.model_idx[mask], dtype=torch.long, device=device
    )
    dilemma_idx = torch.as_tensor(
        matrices.dilemma_idx[mask], dtype=torch.long, device=device
    )
    same_prev = torch.as_tensor(
        matrices.same_prev_mat[mask], dtype=torch.float32, device=device
    )
    exposure_prev = torch.as_tensor(
        matrices.exposure_prev_mat[mask], dtype=torch.float32, device=device
    )
    exposure_within = torch.as_tensor(
        matrices.exposure_within_mat[mask], dtype=torch.float32, device=device
    )
    uses_debate_effects = model.specification.debate_random_intercept
    if uses_debate_effects:
        debate_idx = torch.as_tensor(
            matrices.debate_idx[mask], dtype=torch.long, device=device
        )
        dataset = TensorDataset(
            model_idx,
            dilemma_idx,
            same_prev,
            exposure_prev,
            exposure_within,
            debate_idx,
        )
    else:
        dataset = TensorDataset(
            model_idx,
            dilemma_idx,
            same_prev,
            exposure_prev,
            exposure_within,
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if uses_debate_effects:
                mi, di, sp, ep, ew, qi = batch
                log_probabilities = model(mi, di, sp, ep, ew, qi)
            else:
                mi, di, sp, ep, ew = batch
                log_probabilities = model(mi, di, sp, ep, ew)
            probabilities.append(log_probabilities.exp().cpu().numpy())
    return np.vstack(probabilities)


def model_config(args: argparse.Namespace) -> PaperModelConfig:
    """Translate validation CLI options into the canonical fitting config."""
    return PaperModelConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )


def parse_args() -> argparse.Namespace:
    """Parse cross-validation and output settings."""
    parser = argparse.ArgumentParser(
        description="Held-out verdict accuracy for the paper multinomial model."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "heldout_accuracy.csv",
    )
    parser.add_argument(
        "--by-verdict-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "heldout_accuracy_by_verdict.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run cross-validation and save overall and verdict-level accuracy."""
    args = parse_args()
    matrices = load_model_matrices()
    config = model_config(args)
    model_name = chosen_model_name(args.debate_random_intercept)

    rows = []
    all_true = []
    all_predicted = []
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
        y_predicted = probabilities.argmax(axis=1)
        rows.append(
            {
                "fold": fold_idx,
                "model": model_name,
                "split_unit": args.split_unit,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "accuracy": float((y_predicted == y_true).mean()),
            }
        )
        all_true.append(y_true)
        all_predicted.append(y_predicted)

    y_true = np.concatenate(all_true)
    y_predicted = np.concatenate(all_predicted)
    rows.append(
        {
            "fold": "overall",
            "model": model_name,
            "split_unit": args.split_unit,
            "n_train": np.nan,
            "n_test": len(y_true),
            "accuracy": float((y_predicted == y_true).mean()),
        }
    )
    accuracy = pd.DataFrame(rows)

    verdict_rows = []
    for label_idx, verdict in enumerate(VERDICT_NAMES):
        true_mask = y_true == label_idx
        verdict_rows.append(
            {
                "verdict": verdict,
                "n": int(true_mask.sum()),
                "recall_accuracy": (
                    float((y_predicted[true_mask] == y_true[true_mask]).mean())
                    if true_mask.any()
                    else np.nan
                ),
                "predicted_share": float((y_predicted == label_idx).mean()),
                "observed_share": float(true_mask.mean()),
            }
        )
    by_verdict = pd.DataFrame(verdict_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.by_verdict_output.parent.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(args.output, index=False)
    by_verdict.to_csv(args.by_verdict_output, index=False)

    print(accuracy.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBy verdict:")
    print(by_verdict.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.by_verdict_output}")


if __name__ == "__main__":
    main()
