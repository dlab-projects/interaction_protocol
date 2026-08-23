"""Convert the DeepSeek datasets used in Appendix E to Parquet."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from interaction_protocol.data import load_debate_data, save_debate_data
from interaction_protocol.processing import process_deliberation_results


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "deliberations"
OUTPUT_DIR = REPO_ROOT / "data" / "analysis"
N_DILEMMAS = 1_000
N_AGENTS = 2
MAX_ROUNDS = 4

REQUIRED_COLUMNS = (
    "n_rounds",
    "final_verdict",
    "Agent_1_verdicts",
    "Agent_2_verdicts",
    "Agent_1_messages",
    "Agent_2_messages",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one legacy input and its model-order-aware output name."""

    source_name: str
    output_name: str
    agent_1_model: str
    agent_2_model: str


DATASETS = (
    DatasetSpec(
        "exp7_sync_h2h_gpt_dsk.pkl",
        "sync_h2h_gpt_vs_deepseek.parquet",
        "GPT-4.1",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "exp8_sync_h2h_cla_dsk.pkl",
        "sync_h2h_cla_vs_deepseek.parquet",
        "Claude 3.7 Sonnet",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "exp9_sync_h2h_gem_dsk.pkl",
        "sync_h2h_gem_vs_deepseek.parquet",
        "Gemini 2.0 Flash",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "rr_h2h_gpt_dsk.pkl",
        "round_robin_h2h_gpt_vs_deepseek.parquet",
        "GPT-4.1",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "rr_h2h_dsk_gpt.pkl",
        "round_robin_h2h_deepseek_vs_gpt.parquet",
        "DeepSeek V3.2",
        "GPT-4.1",
    ),
    DatasetSpec(
        "rr_h2h_cla_dsk.pkl",
        "round_robin_h2h_cla_vs_deepseek.parquet",
        "Claude 3.7 Sonnet",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "rr_h2h_dsk_cla.pkl",
        "round_robin_h2h_deepseek_vs_cla.parquet",
        "DeepSeek V3.2",
        "Claude 3.7 Sonnet",
    ),
    DatasetSpec(
        "rr_h2h_gem_dsk.pkl",
        "round_robin_h2h_gem_vs_deepseek.parquet",
        "Gemini 2.0 Flash",
        "DeepSeek V3.2",
    ),
    DatasetSpec(
        "rr_h2h_dsk_gem.pkl",
        "round_robin_h2h_deepseek_vs_gem.parquet",
        "DeepSeek V3.2",
        "Gemini 2.0 Flash",
    ),
)


def validate_dataset(dataframe: pd.DataFrame, spec: DatasetSpec) -> None:
    """Validate the common debate schema and per-agent round alignment."""
    missing_columns = set(REQUIRED_COLUMNS).difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{spec.source_name} is missing columns: {missing}")
    if len(dataframe) != N_DILEMMAS:
        raise ValueError(
            f"{spec.source_name} contains {len(dataframe):,} rows; "
            f"expected {N_DILEMMAS:,}"
        )
    if not dataframe["n_rounds"].between(1, MAX_ROUNDS).all():
        raise ValueError(f"{spec.source_name} contains an invalid round count")

    for agent_index in range(1, N_AGENTS + 1):
        verdict_column = f"Agent_{agent_index}_verdicts"
        message_column = f"Agent_{agent_index}_messages"
        verdict_lengths = dataframe[verdict_column].map(len)
        message_lengths = dataframe[message_column].map(len)
        aligned = verdict_lengths.eq(message_lengths) & verdict_lengths.eq(
            dataframe["n_rounds"]
        )
        if not aligned.all():
            bad_rows = aligned.index[~aligned].tolist()
            raise ValueError(
                f"{spec.source_name} has misaligned Agent {agent_index} rounds "
                f"at rows {bad_rows[:10]}"
            )


def convert_dataset(spec: DatasetSpec) -> None:
    """Process, validate, and save one trusted legacy DeepSeek dataset."""
    source_path = SOURCE_DIR / spec.source_name
    output_path = OUTPUT_DIR / spec.output_name

    raw_results = pd.read_pickle(source_path)
    dataframe = process_deliberation_results(
        raw_results,
        n_agents=N_AGENTS,
        max_rounds=MAX_ROUNDS,
    ).loc[:, REQUIRED_COLUMNS]
    validate_dataset(dataframe, spec)
    save_debate_data(dataframe, output_path)

    restored = load_debate_data(output_path)
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True),
        dataframe.reset_index(drop=True),
        check_dtype=False,
    )
    print(
        f"Wrote {output_path.relative_to(REPO_ROOT)} "
        f"({spec.agent_1_model} -> {spec.agent_2_model}; "
        f"{len(restored):,} rows)"
    )


def main() -> None:
    """Regenerate all DeepSeek Parquet datasets used in Appendix E."""
    for spec in DATASETS:
        convert_dataset(spec)


if __name__ == "__main__":
    main()
