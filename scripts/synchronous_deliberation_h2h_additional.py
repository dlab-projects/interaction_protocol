import asyncio, os
import pandas as pd
import pickle
import traceback

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import ModelInfo
from outlines import Template
from pyprojroot import here
from tqdm import tqdm

from llm_deliberation.sync_delib import sync_delib


# ── config ───────────────────────────────────────────────────────────────────
MAX_ROUNDS = 4
TEMPERATURE = 1
DILEMMAS_PATH = here("data/processed/scenarios_verdicts.csv")
PROMPT_PATH = here("prompts/sync_delib_h2h_v2.txt")
TRACKING_PATH = here("data/deliberations/sync_h2h_llama70b_gpt.pkl")

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DILEMMAS_PATH)
dilemmas = df['selftext_cleaned'].iloc[:1000]
template = Template.from_file(PROMPT_PATH)

# ── model clients ────────────────────────────────────────────────────────────
model_names = ['Agent1', 'Agent2']

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
llama8b = OpenAIChatCompletionClient(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    temperature=TEMPERATURE,
    api_key=os.getenv("TOGETHER_API_KEY"),
    model_info=ModelInfo(
        model_type='chat-completion',
        vision=False,
        function_calling=True,
        structured_output=True,
        json_output=True,
        family="llama-3.1"
    ))
llama70b = OpenAIChatCompletionClient(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    temperature=TEMPERATURE,
    api_key=os.getenv("TOGETHER_API_KEY"),
    model_info=ModelInfo(
        model_type='chat-completion',
        vision=False,
        function_calling=True,
        structured_output=True,
        json_output=True,
        family="llama-3.1"
    ))

clients = {'Agent1': gpt, 'Agent2': llama70b}

if os.path.exists(TRACKING_PATH):
    with open(TRACKING_PATH, 'rb') as file:
        outputs = pickle.load(file)
else:
    outputs = []

n_outputs = len(outputs)
sub = dilemmas.iloc[n_outputs:]

for idx, dilemma in tqdm(enumerate(sub), total=len(sub)):
    try:
        result = asyncio.run(sync_delib(
            dilemma=dilemma,
            clients=clients,
            model_names=model_names,
            system_prompt_template=template,
            verbose=False,
            max_rounds=MAX_ROUNDS))
    except Exception as e:
        print(f"Error processing dilemma {idx + n_outputs}: {e}")
        traceback.print_exc()
        break
    outputs.append(result)

    with open(TRACKING_PATH, 'wb') as file:
        pickle.dump(outputs, file)
