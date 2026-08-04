"""Helpers for submitting batch requests to Gemini or OpenAI."""

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from google.genai import Client as GeminiClient, types as gemini_types
from openai import OpenAI


def submit_gemini_batch(
    prompts: Sequence[str],
    *,
    system_prompt: str,
    schema: Mapping[str, object],
    output_path: Path,
    model: str,
    max_output_tokens: int,
    temperature: float,
    api_key_env_var: str = "GEMINI_API_KEY",
    display_name: str | None = None,
) -> str:
    """Write a Gemini batch payload, upload it, and return the batch job name."""

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{api_key_env_var}' must be set for Gemini batching."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for idx, prompt in enumerate(prompts):
            payload = {
                "key": f"request{idx}",
                "request": {
                    "contents": [
                        {"parts": [{"text": prompt}], "role": "user"}
                    ],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": {
                        "max_output_tokens": max_output_tokens,
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                        "response_json_schema": schema,
                    },
                },
            }
            fout.write(json.dumps(payload))
            fout.write("\n")

    client = GeminiClient(api_key=api_key)
    upload = client.files.upload(
        file=str(output_path),
        config=gemini_types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    batch_job = client.batches.create(
        model=model,
        src=upload.name,
    )
    return batch_job.name


def submit_openai_batch(
    prompts: Sequence[str],
    *,
    system_prompt: str,
    schema: Mapping[str, object],
    output_path: Path,
    model: str,
    max_output_tokens: int,
    temperature: float,
    api_key_env_var: str = "OPENAI_API_KEY",
    schema_name: str = "ValuesResponse",
) -> str:
    """Write an OpenAI batch payload, upload it, and return the batch job id."""

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{api_key_env_var}' must be set for OpenAI batching."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for idx, prompt in enumerate(prompts):
            payload = {
                "custom_id": f"request{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": schema,
                        },
                    },
                },
            }
            fout.write(json.dumps(payload))
            fout.write("\n")

    client = OpenAI(api_key=api_key)
    with output_path.open("rb") as handle:
        uploaded_file = client.files.create(
            purpose="batch",
            file=handle,
        )

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id
