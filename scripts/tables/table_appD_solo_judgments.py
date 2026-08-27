"""Generate the two Appendix D tables for solo-model judgments."""

from pathlib import Path

import pandas as pd

from interaction_protocol.data import load_debate_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "tables"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "appD_solo_judgments.md"
LATEX_OUTPUT_PATH = OUTPUT_DIR / "appD_solo_judgments.tex"

SOLO_PATH = DATA_DIR / "solo_judgments.parquet"
N_DILEMMAS = 1_000
VERDICTS = ("NTA", "YTA", "NAH", "ESH", "INFO")
RUNS = (1, 2, 3)
DECIMAL_PLACES = 3

# Label, synchronous dataset, verdict column, solo-model column prefix.
SYNC_CONDITIONS = (
    (
        "GPT (vs. Claude)",
        "sync_h2h_cla_vs_gpt.parquet",
        "Agent_2_verdicts",
        "gpt41",
    ),
    (
        "GPT (vs. Gemini)",
        "sync_h2h_gpt_vs_gem.parquet",
        "Agent_1_verdicts",
        "gpt41",
    ),
    (
        "Claude (vs. GPT)",
        "sync_h2h_cla_vs_gpt.parquet",
        "Agent_1_verdicts",
        "claude37_sonnet",
    ),
    (
        "Claude (vs. Gemini)",
        "sync_h2h_cla_vs_gem.parquet",
        "Agent_1_verdicts",
        "claude37_sonnet",
    ),
    (
        "Gemini (vs. Claude)",
        "sync_h2h_cla_vs_gem.parquet",
        "Agent_2_verdicts",
        "gemini20_flash",
    ),
    (
        "Gemini (vs. GPT)",
        "sync_h2h_gpt_vs_gem.parquet",
        "Agent_2_verdicts",
        "gemini20_flash",
    ),
)
MODEL_GROUPS = (
    ("GPT", "gpt41"),
    ("Claude", "claude37_sonnet"),
    ("Gemini", "gemini20_flash"),
)


def load_first_round_verdicts(filename: str, column: str) -> pd.Series:
    """Load one model's Round 1 synchronous verdicts."""
    dataframe = load_debate_data(DATA_DIR / filename)
    if len(dataframe) != N_DILEMMAS:
        raise ValueError(f"{filename} contains {len(dataframe):,} rows")
    verdicts = dataframe[column].map(lambda rounds: rounds[0]).reset_index(drop=True)
    return verdicts


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load and validate the solo and synchronous judgment inputs."""
    solo = load_debate_data(SOLO_PATH).reset_index(drop=True)
    if len(solo) != N_DILEMMAS:
        raise ValueError(f"{SOLO_PATH.name} contains {len(solo):,} rows")

    synchronous = {
        label: load_first_round_verdicts(filename, column)
        for label, filename, column, _ in SYNC_CONDITIONS
    }
    return solo, synchronous


def verdict_distribution(verdicts: pd.Series) -> dict[str, float]:
    """Return verdict proportions in the manuscript's column order."""
    proportions = verdicts.value_counts(normalize=True)
    return {verdict: float(proportions.get(verdict, 0.0)) for verdict in VERDICTS}


def build_distribution_table(
    solo: pd.DataFrame, synchronous: dict[str, pd.Series]
) -> pd.DataFrame:
    """Build the solo vs. synchronous Round 1 verdict-distribution table."""
    rows: list[dict[str, str | float]] = []
    for model, prefix in MODEL_GROUPS:
        for label, _, _, condition_prefix in SYNC_CONDITIONS:
            if condition_prefix == prefix:
                rows.append(
                    {"Experiment": label, **verdict_distribution(synchronous[label])}
                )
        for run in RUNS:
            column = f"{prefix}_run{run}_judgment"
            rows.append(
                {
                    "Experiment": f"{model} (run {run})",
                    **verdict_distribution(solo[column]),
                }
            )
    return pd.DataFrame(rows, columns=("Experiment", *VERDICTS))


def build_agreement_table(
    solo: pd.DataFrame, synchronous: dict[str, pd.Series]
) -> pd.DataFrame:
    """Build the synchronous-to-solo self-agreement table."""
    rows = []
    for label, _, _, prefix in SYNC_CONDITIONS:
        row: dict[str, str | float] = {"Experiment": label}
        for run in RUNS:
            solo_verdicts = solo[f"{prefix}_run{run}_judgment"]
            row[f"Individual Run {run}"] = float(
                synchronous[label].eq(solo_verdicts).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_number(value: float) -> str:
    """Format a table entry to the paper's three-decimal precision."""
    return f"{value:.{DECIMAL_PLACES}f}"


def render_markdown_table(dataframe: pd.DataFrame) -> str:
    """Render one numeric table as GitHub-flavored Markdown."""
    headers = dataframe.columns.tolist()
    rows = [
        [str(row.iloc[0]), *(format_number(value) for value in row.iloc[1:])]
        for _, row in dataframe.iterrows()
    ]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---", *(["---:"] * (len(headers) - 1))]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_latex_table(dataframe: pd.DataFrame) -> str:
    """Render one numeric table as a caption-free LaTeX table environment."""
    headers = " & ".join(dataframe.columns) + r" \\"
    rows = [
        " & ".join(
            [str(row.iloc[0]), *(format_number(value) for value in row.iloc[1:])]
        )
        + r" \\"
        for _, row in dataframe.iterrows()
    ]
    column_format = "l" + "c" * (len(dataframe.columns) - 1)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\begin{{tabular}}{{{column_format}}}",
            r"\toprule",
            headers,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def main() -> None:
    """Print and save Markdown and LaTeX versions of both Appendix D tables."""
    solo, synchronous = load_inputs()
    distribution_table = build_distribution_table(solo, synchronous)
    agreement_table = build_agreement_table(solo, synchronous)

    markdown = "\n\n".join(
        [
            "## Solo vs. synchronous Round 1 verdict distributions\n\n"
            + render_markdown_table(distribution_table),
            "## Self-agreement between synchronous and solo verdicts\n\n"
            + render_markdown_table(agreement_table),
        ]
    )
    latex = "\n\n".join(
        [
            render_latex_table(distribution_table),
            render_latex_table(agreement_table),
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")
    LATEX_OUTPUT_PATH.write_text(latex + "\n", encoding="utf-8")

    print("Markdown:\n")
    print(markdown)
    print("\nLaTeX:\n")
    print(latex)


if __name__ == "__main__":
    main()
