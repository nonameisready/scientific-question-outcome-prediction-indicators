"""Thin OpenAI chat wrapper with deterministic on-disk caching.

Every call is cached under a hash of (model, messages, response format), so
re-running any pipeline stage is free and reproducible. Generation and
judging both go through here.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path("data/cache/llm")

GENERATION_MODEL = os.environ.get("PAPER3_GENERATION_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("PAPER3_JUDGE_MODEL", "gpt-4o-mini")
SECOND_JUDGE_MODEL = os.environ.get("PAPER3_SECOND_JUDGE_MODEL", "gpt-4o")

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()
    return _client


def chat_json(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    seed: int = 7,
    max_retries: int = 5,
) -> dict:
    """Chat completion constrained to a JSON object, cached on disk."""
    key_src = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "seed": seed},
        sort_keys=True,
        ensure_ascii=False,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / model / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    client = _get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(parsed, ensure_ascii=False))
            return parsed
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts") from last_err
