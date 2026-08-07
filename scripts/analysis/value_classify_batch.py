import json
import pickle
from pathlib import Path
from typing import Any

import chz  # pyright: ignore[reportMissingImports]
import pandas as pd
from loguru import logger  # pyright: ignore[reportMissingImports]
from outlines import Template

from interaction_protocol.values import Values
from interaction_protocol.batch_judges import submit_gemini_batch, submit_openai_batch


RESPONSE_SCHEMA = Values.model_json_schema()


@chz.chz
class Config:
    """Typed configuration parsed by chz; CLI flags map directly to these fields."""

    agent_id: int
    system_prompt_path: str
    input_pickle_path: str
    batch_output_path: str
    judge: str = "gemini"
    model: str = "models/gemini-2.5-flash"
    max_output_tokens: int = 5000
    temperature: float = 1.0
    api_key_env_var: str | None = None
    upload_display_name: str | None = None
    openai_schema_name: str = "ValuesResponse"


def run(config: Config) -> None:
    """Emit a batch job using the configured judge provider."""

    # Resolve user-provided paths to Path objects for convenience.
    input_path = Path(config.input_pickle_path)
    output_path = Path(config.batch_output_path)
    system_prompt_path = Path(config.system_prompt_path)

    logger.info(
        "Preparing batch for agent {} using {} -> {}",
        config.agent_id,
        input_path,
        output_path,
    )

    # Load the previously saved deliberation dataframe.
    logger.info("Loading deliberation dataframe from {}", input_path)
    with input_path.open("rb") as file:
        df: pd.DataFrame = pickle.load(file)

    # Render the system prompt template on the fly so updates propagate automatically.
    logger.info("Rendering system prompt template from {}", system_prompt_path)
    system_prompt_template = Template.from_file(system_prompt_path)
    system_prompt = system_prompt_template()

    # Pull verdict text for the selected agent from every scenario.
    column = f"Agent_{config.agent_id}_messages"
    prompts: list[str] = []
    for messages in df[column]:
        for message in messages:
            lines = message.splitlines()
            prompts.append("\n".join(lines[1:]))

    logger.info("Collected {} prompts for batching", len(prompts))

    judge_name = config.judge.lower()
    api_key_env_var = config.api_key_env_var
    logger.info("Writing batch requests to {}", output_path)

    if judge_name == "gemini":
        job_id = submit_gemini_batch(
            prompts,
            system_prompt=system_prompt,
            schema=RESPONSE_SCHEMA,
            output_path=output_path,
            model=config.model,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            api_key_env_var=api_key_env_var or "GEMINI_API_KEY",
            display_name=config.upload_display_name
        )
    elif judge_name in {"gpt", "openai"}:
        job_id = submit_openai_batch(
            prompts,
            system_prompt=system_prompt,
            schema=RESPONSE_SCHEMA,
            output_path=output_path,
            model=config.model,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            api_key_env_var=api_key_env_var or "OPENAI_API_KEY",
            schema_name=config.openai_schema_name,
        )
    else:
        raise ValueError(f"Unsupported judge '{config.judge}'")

    logger.success(
        "Submitted batch job {} with payload {} using judge {}",
        job_id,
        output_path,
        config.judge,
    )


def load_config_from_file(config_path: str | None) -> dict[str, Any]:
    """Load configuration values from a JSON file if one was specified."""

    if not config_path:
        return {}
    path = Path(config_path)
    logger.info("Loading configuration from {}", path)
    with path.open() as file:
        return json.load(file)


def main(
    config: str | None = None,
    judge: str | None = None,
    agent_id: int | None = None,
    system_prompt_path: str | None = None,
    input_pickle_path: str | None = None,
    batch_output_path: str | None = None,
    model: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    api_key_env_var: str | None = None,
    upload_display_name: str | None = None,
    openai_schema_name: str | None = None,
) -> None:
    """CLI entry point – merge file-based config with CLI overrides and run the job."""

    config_path = config
    base_config = load_config_from_file(config_path)
    overrides = {
        "judge": judge,
        "agent_id": agent_id,
        "system_prompt_path": system_prompt_path,
        "input_pickle_path": input_pickle_path,
        "batch_output_path": batch_output_path,
        "model": model,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "api_key_env_var": api_key_env_var,
        "upload_display_name": upload_display_name,
        "openai_schema_name": openai_schema_name,
    }

    config_kwargs = {**base_config}
    for key, value in overrides.items():
        if value is not None:
            config_kwargs[key] = value

    resolved_config = Config(**config_kwargs)
    logger.info(
        "Starting value classification batch for agent %s via judge %s (config %s)",
        resolved_config.agent_id,
        resolved_config.judge,
        config_path or "<inline>",
    )
    run(resolved_config)


if __name__ == "__main__":
    chz.entrypoint(main)
