"""Generate Appendix E Table 6 DeepSeek verdict distributions."""

from pathlib import Path

import pandas as pd

from interaction_protocol.data import load_debate_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "tables"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "appE_deepseek_verdict_distribution.md"
LATEX_OUTPUT_PATH = OUTPUT_DIR / "appE_deepseek_verdict_distribution.tex"

EXPERIMENTS = (
    ("DeepSeek vs GPT-4.1", "sync_h2h_gpt_vs_deepseek.parquet"),
    ("DeepSeek vs Claude 3.7", "sync_h2h_cla_vs_deepseek.parquet"),
    ("DeepSeek vs Gemini 2.0", "sync_h2h_gem_vs_deepseek.parquet"),
)
DEEPSEEK_VERDICT_COLUMN = "Agent_2_verdicts"
VERDICTS = ("NTA", "YTA", "ESH", "NAH", "INFO")
N_DILEMMAS = 1_000
DECIMAL_PLACES = 3
LATEX_COLUMN_FORMAT = "lrrrrr"


def deepseek_round_one_verdicts(path: Path) -> pd.Series:
    """Load and validate DeepSeek's Round 1 verdicts for one experiment."""
    dataframe = load_debate_data(path)
    if len(dataframe) != N_DILEMMAS:
        raise ValueError(f"{path.name} contains {len(dataframe):,} rows")

    verdicts = dataframe[DEEPSEEK_VERDICT_COLUMN].map(lambda rounds: rounds[0])
    if verdicts.isna().any():
        raise ValueError(f"{path.name} contains missing DeepSeek Round 1 verdicts")
    return verdicts


def build_table() -> pd.DataFrame:
    """Calculate DeepSeek's Round 1 verdict distribution in each pairing."""
    rows = []
    for experiment, filename in EXPERIMENTS:
        verdicts = deepseek_round_one_verdicts(DATA_DIR / filename)
        distribution = verdicts.value_counts(normalize=True)
        rows.append(
            {
                "Experiment": experiment,
                **{
                    verdict: float(distribution.get(verdict, 0.0))
                    for verdict in VERDICTS
                },
            }
        )
    return pd.DataFrame(rows, columns=("Experiment", *VERDICTS))


def format_rate(value: float) -> str:
    """Format a verdict proportion to the manuscript's precision."""
    return f"{value:.{DECIMAL_PLACES}f}"


def render_markdown(dataframe: pd.DataFrame) -> str:
    """Render Table 6 as GitHub-flavored Markdown."""
    headers = dataframe.columns.tolist()
    rows = [
        [
            f"**{row['Experiment']}**",
            *(format_rate(row[verdict]) for verdict in VERDICTS),
        ]
        for _, row in dataframe.iterrows()
    ]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---", *(["---:"] * len(VERDICTS))]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_latex(dataframe: pd.DataFrame) -> str:
    """Render Table 6 as a caption-free LaTeX table environment."""
    rows = [
        " & ".join(
            [
                rf"\textbf{{{row['Experiment']}}}",
                *(format_rate(row[verdict]) for verdict in VERDICTS),
            ]
        )
        + r" \\"
        for _, row in dataframe.iterrows()
    ]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\begin{{tabular}}{{{LATEX_COLUMN_FORMAT}}}",
            r"\toprule",
            "Experiment & " + " & ".join(VERDICTS) + r" \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def main() -> None:
    """Print and save Markdown and LaTeX versions of Appendix E Table 6."""
    dataframe = build_table()
    markdown = render_markdown(dataframe)
    latex = render_latex(dataframe)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")
    LATEX_OUTPUT_PATH.write_text(latex + "\n", encoding="utf-8")

    print("Markdown:\n")
    print(markdown)
    print("\nLaTeX:\n")
    print(latex)


if __name__ == "__main__":
    main()
