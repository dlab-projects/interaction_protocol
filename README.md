# [Interaction Protocol Shapes Moral Judgment in Multi-Agent Debate](https://arxiv.org/pdf/2510.10002)

**Pratik S. Sachdeva and Tom van Nuenen**

*Accepted at COLM 2026*

## Abstract

As large language models (LLMs) are increasingly deployed in sensitive everyday contexts—offering personal advice, mental health support, and moral guidance—understanding their behavior in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings, and even less clear how they depend on the interaction protocols used to coordinate agentic systems.

We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's “Am I the Asshole” community.

To test order effects and assess verdict revision, we use both synchronous (parallel responses) and round-robin (sequential responses) deliberation structures, mirroring how multi-agent systems are increasingly orchestrated in practice. Our findings show striking behavioral differences. In the synchronous setting, GPT-4.1 showed strong inertia (0.6–3.1% revision rates), while Claude 3.7 Sonnet and Gemini 2.0 Flash were far more flexible (28–41% revision rates).

Value patterns also diverged: GPT-4.1 emphasized personal autonomy and direct communication (relative to its deliberation partners), while Claude 3.7 Sonnet and Gemini 2.0 Flash prioritized empathetic dialogue. We further find that deliberation format had a strong impact on model behavior: GPT-4.1 and Gemini 2.0 Flash stood out as highly conforming relative to Claude 3.7 Sonnet, with their verdict behavior strongly shaped by order effects. We provide additional results on open-source models (DeepSeek-V3.2 and Llama 3.1).

## Repository Structure

```text
interaction_protocol/
├── artifacts/
│   └── figures/
├── data/
│   ├── analysis/
│   └── deliberations/
├── experiments/
├── figures/
├── notebooks/
├── prompts/
├── scripts/
│   ├── analysis/
│   ├── debates/
│   └── plots/
├── src/
│   └── interaction_protocol/
│       ├── __init__.py
│       ├── batch_deliberation.py
│       ├── batch_judges.py
│       ├── data.py
│       ├── deliberation.py
│       ├── multilevel.py
│       ├── plotting.py
│       ├── processing.py
│       ├── round_robin_delib.py
│       ├── sync_delib.py
│       ├── utils.py
│       └── values.py
├── tables/
├── tests/
├── README.md
├── pyproject.toml
└── uv.lock
```
