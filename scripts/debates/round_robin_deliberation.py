import asyncio, os
import pandas as pd
import pickle
from typing import Dict, List
from collections import defaultdict

from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import ModelInfo


from interaction_protocol.deliberation import extract_verdict
from interaction_protocol.round_robin_delib import round_robin_delib
from outlines import Template
from pyprojroot import here
from tqdm import tqdm

MAX_ROUNDS = 4
N_AGENTS = 2
MAX_TURNS = N_AGENTS * MAX_ROUNDS
TEMPERATURE = 1
DILEMMAS_PATH = here("data/processed/scenarios_verdicts.csv")
PROMPT_PATH = here("prompts/round_robin_h2h.txt")
TRACKING_PATH = here("data/deliberations/rr_h2h_dsk_gem.pkl")

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DILEMMAS_PATH)
template = Template.from_file(PROMPT_PATH)
dilemmas = df['selftext_cleaned'].iloc[:1000]

# ── Clients ───────────────────────────────────────────────────────────────────
claude = AnthropicChatCompletionClient(
    model="claude-3-7-sonnet-20250219",
    temperature=TEMPERATURE,
    api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    temperature=TEMPERATURE,
    api_key=os.getenv("GEMINI_API_KEY"))
gpt = OpenAIChatCompletionClient(
    model="gpt-4.1-2025-04-14",
    temperature=TEMPERATURE,
    api_key=os.getenv("OPENAI_API_KEY"))
deepseek = OpenAIChatCompletionClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=TEMPERATURE,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_info=ModelInfo(
        model_type='chat-completion',
        vision=False,
        function_calling=True,
        structured_output=True,
        json_output=True,
        family="deepseek"
    ))

clients = {'Agent1': deepseek, 'Agent2': gemini}
model_names = ['Agent 1', 'Agent 2']

#clients = {'Agent1': agent1, 'Agent2': agent2, 'Agent3': agent3}
#model_names = ['Agent 1', 'Agent 2', 'Agent 3']

if os.path.exists(TRACKING_PATH):
    with open(TRACKING_PATH, 'rb') as file:
        outputs = pickle.load(file)
else:
    outputs = []

n_outputs = len(outputs)
sub = dilemmas.iloc[n_outputs:]

for idx, dilemma in tqdm(enumerate(sub), total=len(sub)):
    try:
        result = asyncio.run(round_robin_delib(
            dilemma=dilemma,
            clients=clients,
            model_names=model_names,
            system_prompt_template=template,
            max_rounds=MAX_ROUNDS,
            verbose=False
        ))
    except Exception as e:
        print(f"Error processing dilemma {idx + n_outputs}")
        print(e)
        break
    outputs.append(result)

    with open(TRACKING_PATH, 'wb') as file:
        pickle.dump(outputs, file)
