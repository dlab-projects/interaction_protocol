# Interaction Protocol Shapes Moral Judgment in Multi-Agent Debate

<p align="center">
  Pratik S. Sachdeva &nbsp;·&nbsp; Tom van Nuenen<br>
  <strong>Conference on Language Modeling (COLM), 2026</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.10002"><img src="https://img.shields.io/badge/arXiv-2510.10002-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/ucberkeley-dlab/interaction_protocol"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E.svg" alt="Hugging Face Dataset"></a>
</p>

## Abstract

As large language models (LLMs) are increasingly deployed in sensitive everyday contexts—offering personal advice, mental health support, and moral guidance—understanding their behavior in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings, and even less clear how they depend on the interaction protocols used to coordinate agentic systems. We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's “Am I the Asshole” community. To test order effects and assess verdict revision, we use both synchronous (parallel responses) and round-robin (sequential responses) deliberation structures, mirroring how multi-agent systems are increasingly orchestrated in practice. Our findings show striking behavioral differences. In the synchronous setting, GPT-4.1 showed strong inertia (0.6–3.1% revision rates), while Claude 3.7 Sonnet and Gemini 2.0 Flash were far more flexible (28–41% revision rates). Value patterns also diverged: GPT-4.1 emphasized personal autonomy and direct communication relative to its deliberation partners, while Claude 3.7 Sonnet and Gemini 2.0 Flash prioritized empathetic dialogue. We further find that deliberation format had a strong impact on model behavior: GPT-4.1 and Gemini 2.0 Flash stood out as highly conforming relative to Claude 3.7 Sonnet, with their verdict behavior strongly shaped by order effects. We provide additional results on open-source models (DeepSeek-V3.2 and Llama 3.1).

## Setup

The project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/dlab-projects/interaction_protocol.git
cd interaction_protocol
uv sync
```

Run the test suite with:

```sh
uv run pytest
```

## Data

Processed experiment data are distributed separately through the [Hugging Face dataset](https://huggingface.co/datasets/ucberkeley-dlab/interaction_protocol). Download the Parquet files into the paths expected by every analysis and plotting script:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ucberkeley-dlab/interaction_protocol",
    repo_type="dataset",
    allow_patterns="experiments/*.parquet",
    local_dir="data",
)
```

This creates `data/experiments/*.parquet`. The `data/` directory is intentionally ignored by Git; it does not need to exist before downloading.

## Fit the statistical models

The canonical model runner is `scripts/models/fit_models.py`. With no flags it fits the multinomial inertia–conformity model reported in the paper:

```sh
uv run python scripts/models/fit_models.py
```

Fit the nested comparison models, their MAP and MLE versions, and the debate-level sensitivity model with:

```sh
uv run python scripts/models/fit_models.py --all-models
```

Add `--bootstrap` to run the dilemma-cluster bootstrap. Model checkpoints, coefficient tables, bootstrap results, and fit summaries are written to `artifacts/models/`.

The held-out analyses use the same Parquet inputs and registered model implementation:

```sh
uv run python scripts/analysis/heldout_accuracy.py
uv run python scripts/analysis/calibration_analysis.py
uv run python scripts/plots/plot_extra_heldout_calibration.py
```

These commands hold out experiment-specific debates by default. Use `--split-unit dilemma` for generalization to entirely unseen dilemmas. Analysis outputs are written to `artifacts/analysis/`.

## Reproduce figures and tables

The main quantitative figures are generated independently:

```sh
uv run python scripts/plots/plot_figure_2.py
uv run python scripts/plots/plot_figure_3.py
uv run python scripts/plots/plot_figure_4.py
uv run python scripts/plots/plot_figure_5.py
```

Appendix scripts follow the same naming convention.

Figure PDFs are written to `artifacts/figures/`. Table scripts under `scripts/tables/` write Markdown and LaTeX versions to `artifacts/tables/`. These output directories are ignored by Git and created automatically by the scripts.

## Repository structure

```text
interaction_protocol/
├── huggingface/             # Hugging Face dataset card
├── prompts/                 # Debate and value-classification prompts
├── scripts/
│   ├── analysis/            # Held-out and value-classification analyses
│   ├── debates/             # Experiment runners (API access required)
│   ├── models/              # Canonical statistical-model runner
│   ├── plots/               # Main-text and appendix figures
│   └── tables/              # Appendix table generators
├── src/interaction_protocol/
│   ├── data.py              # Stable Parquet/pickle loading utilities
│   ├── models.py            # Preprocessing and statistical models
│   ├── plotting.py          # Shared plotting helpers
│   └── ...                  # Debate, processing, and value utilities
├── tests/
├── data/experiments/        # Downloaded data; ignored by Git
└── artifacts/               # Generated results; ignored by Git
```

The reproducible public pipeline does not depend on notebooks. Personal notebooks, raw intermediate files, generated figures, and archival material are intentionally excluded from the repository.

## Citation

```bibtex
@inproceedings{sachdeva2026interaction,
  title={Interaction Protocol Shapes Moral Judgment in Multi-Agent Debate},
  author={Sachdeva, Pratik S. and van Nuenen, Tom},
  booktitle={Proceedings of the Conference on Language Modeling (COLM)},
  year={2026}
}
```

## AI disclosure
This README was generated with GPT-5.6 Sol. A sizable fraction (~1/3) of the code in this repository was generated by various GPT-5x models in Codex under the supervision of the authors.
