"""Generate Table 2 no-consensus rates for system-prompt experiments."""

from pathlib import Path

import pandas as pd

from interaction_protocol.data import load_debate_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "tables"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "appC_no_consensus_rates.md"
LATEX_OUTPUT_PATH = OUTPUT_DIR / "appC_no_consensus_rates.tex"

PAIRINGS = (
    (
        "Claude 3.7 Sonnet vs. GPT-4.1",
        "sync_h2h_cla_vs_gpt.parquet",
        "sync_h2h_balanced_cla_vs_gpt.parquet",
        "sync_h2h_adversarial_cla_vs_gpt.parquet",
    ),
    (
        "Claude 3.7 Sonnet vs. Gemini 2.0 Flash",
        "sync_h2h_cla_vs_gem.parquet",
        "sync_h2h_balanced_cla_vs_gem.parquet",
        "sync_h2h_adversarial_cla_vs_gem.parquet",
    ),
    (
        "GPT-4.1 vs. Gemini 2.0 Flash",
        "sync_h2h_gpt_vs_gem.parquet",
        "sync_h2h_balanced_gpt_vs_gem.parquet",
        "sync_h2h_adversarial_gpt_vs_gem.parquet",
    ),
)
CONDITION_COLUMNS = ("Original", "Balanced", "Adversarial")
FINAL_VERDICT_COLUMN = "final_verdict"
DECIMAL_PLACES = 3
LATEX_COLUMN_FORMAT = "lccc"


def no_consensus_rate(path: Path) -> float:
    """Return the fraction of debates without a final consensus verdict."""
    dataframe = load_debate_data(path)
    return float(dataframe[FINAL_VERDICT_COLUMN].isna().mean())


def build_table() -> pd.DataFrame:
    """Calculate no-consensus rates for all pairings and prompt conditions."""
    rows = []
    for pairing, *filenames in PAIRINGS:
        rates = [no_consensus_rate(DATA_DIR / filename) for filename in filenames]
        rows.append({"Pairing": pairing, **dict(zip(CONDITION_COLUMNS, rates, strict=True))})
    return pd.DataFrame(rows)


def format_rate(value: float) -> str:
    """Format a rate using the paper's fixed three-decimal precision."""
    return f"{value:.{DECIMAL_PLACES}f}"


def render_markdown(dataframe: pd.DataFrame) -> str:
    """Render the table as GitHub-flavored Markdown."""
    headers = dataframe.columns.tolist()
    formatted_rows = [
        [row["Pairing"], *(format_rate(row[column]) for column in CONDITION_COLUMNS)]
        for _, row in dataframe.iterrows()
    ]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---", "---:", "---:", "---:"]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted_rows]
    return "\n".join([header, separator, *body])


def render_latex(dataframe: pd.DataFrame) -> str:
    """Render the table as a complete LaTeX table environment."""
    rows = [
        " & ".join(
            [row["Pairing"], *(format_rate(row[column]) for column in CONDITION_COLUMNS)]
        )
        + r" \\"
        for _, row in dataframe.iterrows()
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{LATEX_COLUMN_FORMAT}}}",
        r"\toprule",
        "Pairing & Original & Balanced & Adversarial " + r"\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Print and save Markdown and LaTeX versions of Table 2."""
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
