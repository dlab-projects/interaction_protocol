"""Load the primary experiments and fit registered debate-dynamics models.

Without flags, this script fits only the historical paper model. With
``--all-models``, it fits every nested comparison model twice: an explicitly
L2-penalized MAP fit for coefficient estimates and an unpenalized MLE fit for
standard AIC. It also reports AIC evaluated at the MAP coefficients, clearly
labeled as nonstandard.

Model checkpoints, coefficient tables, bootstrap draws, and fit metadata are
written beneath ``artifacts/models`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from interaction_protocol.data import load_debate_data
from interaction_protocol.models import (
    COMPARISON_MODEL_FITTERS,
    MAIN_MODEL_NAME,
    MODEL_FITTERS,
    SENSITIVITY_MODEL_FITTERS,
    DebateExperiment,
    ModelFit,
    PaperModelConfig,
    bootstrap_paper_model,
    model_coefficients,
    paper_model_coefficients,
    preprocess,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "models"

# The tuple order mirrors the experiment order used by the original fitting
# notebook. Each integer tuple maps agent columns to stable model identities:
# GPT=0, Claude=1, Gemini=2.
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


def load_experiments(specifications) -> list[DebateExperiment]:
    """Load Parquet datasets and attach the model order for their agent columns.

    ``specifications`` contains ``(filename, model_indices)`` pairs. Loading is
    kept outside ``models.py`` so statistical functions remain independent of
    repository paths and storage formats.
    """
    return [
        DebateExperiment(
            dataframe=load_debate_data(DATA_DIR / filename),
            model_indices=model_indices,
        )
        for filename, model_indices in specifications
    ]


def parse_args() -> argparse.Namespace:
    """Parse model selection, optimizer, bootstrap, and output options."""
    parser = argparse.ArgumentParser(
        description="Fit the paper model, or every model registered in models.py."
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Fit MAP and MLE versions of every comparison model.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run a dilemma-cluster bootstrap for the paper model.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-jobs", type=int, default=8)
    parser.add_argument("--bootstrap-seed", type=int, default=2332)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    fit: ModelFit,
    matrices,
    path: Path,
) -> None:
    """Save model weights with dimensions, configuration, and fit diagnostics."""
    torch.save(
        {
            "model_name": fit.name,
            "state_dict": fit.model.state_dict(),
            "config": asdict(fit.config),
            "dimensions": {
                "n_models": matrices.n_models,
                "n_dilemmas": matrices.n_dilemmas,
                "n_verdicts": matrices.n_verdicts,
            },
            "fit": fit.metadata(),
        },
        path,
    )


def run_bootstrap(args, matrices, config) -> None:
    """Run and save the corrected dilemma-cluster bootstrap for the paper model."""
    if not args.bootstrap:
        return
    print(
        f"Running {args.bootstrap_reps} corrected dilemma-cluster bootstrap fits...",
        flush=True,
    )
    bootstrap = bootstrap_paper_model(
        matrices,
        config,
        n_bootstrap=args.bootstrap_reps,
        n_jobs=args.bootstrap_jobs,
        base_seed=args.bootstrap_seed,
    )
    bootstrap.summary().to_csv(
        args.output_dir / f"{MAIN_MODEL_NAME}_bootstrap_summary.csv", index=False
    )
    torch.save(
        {
            "alpha": bootstrap.alpha,
            "gamma_prev": bootstrap.gamma_prev,
            "gamma_within": bootstrap.gamma_within,
            "n_bootstrap": args.bootstrap_reps,
            "base_seed": args.bootstrap_seed,
        },
        args.output_dir / f"{MAIN_MODEL_NAME}_bootstrap.pt",
    )


def fit_default_paper_model(args, matrices, config) -> None:
    """Preserve the original default command and artifact names."""
    print(f"Fitting {MAIN_MODEL_NAME}...", flush=True)
    fit = MODEL_FITTERS[MAIN_MODEL_NAME](
        matrices,
        config,
        verbose=not args.quiet,
    )
    checkpoint_path = args.output_dir / f"{MAIN_MODEL_NAME}.pt"
    save_checkpoint(fit, matrices, checkpoint_path)
    paper_model_coefficients(fit).to_csv(
        args.output_dir / f"{MAIN_MODEL_NAME}_coefficients.csv", index=False
    )
    pd.DataFrame([fit.metadata()]).to_csv(
        args.output_dir / "fit_summary.csv", index=False
    )
    run_bootstrap(args, matrices, config)
    print(
        f"Finished {MAIN_MODEL_NAME}: objective={fit.objective:.6f}; "
        f"checkpoint={checkpoint_path}"
    )


def fit_all_models(args, matrices, config) -> None:
    """Fit comparison MAP/MLE pairs and the MAP-only sensitivity model."""
    comparison_rows = []
    coefficient_tables = []

    for model_name, fitter in COMPARISON_MODEL_FITTERS.items():
        print(f"Fitting {model_name} (MAP and MLE)...", flush=True)
        map_fit = fitter(
            matrices,
            config,
            penalized=True,
            verbose=not args.quiet,
        )
        mle_fit = fitter(
            matrices,
            config,
            penalized=False,
            verbose=not args.quiet,
        )
        save_checkpoint(
            map_fit, matrices, args.output_dir / f"{model_name}_map.pt"
        )
        save_checkpoint(
            mle_fit, matrices, args.output_dir / f"{model_name}_mle.pt"
        )
        coefficients = model_coefficients(map_fit)
        coefficients.insert(1, "fit_type", "MAP")
        if not coefficients.empty:
            coefficient_tables.append(coefficients)
        comparison_rows.append(
            {
                "model": model_name,
                "n": map_fit.n_observations,
                "k": map_fit.n_parameters,
                "map_log_likelihood": map_fit.log_likelihood,
                "map_penalized_objective": map_fit.objective,
                # This uses the AIC formula at MAP coefficients, as requested,
                # but is not standard AIC because the coefficients are not MLEs.
                "aic_at_map_nonstandard": map_fit.aic_at_fit,
                "mle_log_likelihood": mle_fit.log_likelihood,
                "mle_aic": mle_fit.aic_at_fit,
                "map_iterations": map_fit.epochs_fit,
                "mle_iterations": mle_fit.epochs_fit,
                "included_in_standard_aic": True,
            }
        )
        print(
            f"Finished {model_name}: MLE AIC={mle_fit.aic_at_fit:.2f}; "
            f"AIC at MAP={map_fit.aic_at_fit:.2f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows)
    paper_row = comparison.loc[comparison["model"] == MAIN_MODEL_NAME].iloc[0]
    comparison["delta_mle_aic_vs_paper"] = (
        comparison["mle_aic"] - paper_row["mle_aic"]
    )
    comparison["delta_aic_at_map_vs_paper"] = (
        comparison["aic_at_map_nonstandard"]
        - paper_row["aic_at_map_nonstandard"]
    )
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)

    # The debate effects are fitted conditionally with a fixed prior scale. This
    # is useful as a coefficient sensitivity analysis, but it is not comparable
    # to the ordinary MLE models through standard AIC.
    sensitivity_rows = []
    for model_name, fitter in SENSITIVITY_MODEL_FITTERS.items():
        print(f"Fitting {model_name} (MAP sensitivity analysis)...", flush=True)
        fit = fitter(
            matrices,
            config,
            penalized=True,
            verbose=not args.quiet,
        )
        save_checkpoint(fit, matrices, args.output_dir / f"{model_name}_map.pt")
        coefficients = model_coefficients(fit)
        coefficients.insert(1, "fit_type", "MAP sensitivity")
        if not coefficients.empty:
            coefficient_tables.append(coefficients)
        sensitivity_rows.append(
            {
                **fit.metadata(),
                "included_in_standard_aic": False,
                "reason": "MAP-only conditional debate effects with fixed prior scale",
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "sensitivity_summary.csv", index=False
    )
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        args.output_dir / "model_coefficients.csv", index=False
    )
    run_bootstrap(args, matrices, config)


def main() -> None:
    """Preprocess once, fit selected models, and persist reproducible artifacts."""
    args = parse_args()
    # All model functions receive this exact matrix object. Preprocessing only
    # once prevents subtle candidate-model differences caused by data handling.
    matrices = preprocess(
        load_experiments(SYNCHRONOUS_EXPERIMENTS),
        load_experiments(ROUND_ROBIN_EXPERIMENTS),
    )
    print(
        f"Preprocessed {matrices.n_observations:,} verdicts across "
        f"{len(set(matrices.debate_idx.tolist())):,} debates."
    )

    config = PaperModelConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.all_models:
        fit_all_models(args, matrices, config)
    else:
        fit_default_paper_model(args, matrices, config)


if __name__ == "__main__":
    main()
