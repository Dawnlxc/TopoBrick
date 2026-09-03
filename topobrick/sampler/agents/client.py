from __future__ import annotations

import json
import re
import time
from typing import Any

RETRY_DELAYS = (0, 5, 15)
RETRYABLE_ERRORS = (
    "429",
    "rate",
    "timeout",
    "connection",
    "overloaded",
    "decode",
    "json",
)


def _parse_json_response(content: str) -> dict[str, Any]:
    if not content:
        return {}
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    block = match.group(1) if match else content[content.rfind("{") :]
    candidates = [block]
    if "}" in block:
        candidates.append(block[: block.rfind("}") + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
    return {}


def request_json(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int = 16000,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if "gpt-oss" in model.lower():
        request["extra_body"] = {"reasoning_effort": "low"}
    response = client.chat.completions.create(**request)
    return _parse_json_response(response.choices[0].message.content or "")


def request_with_retry(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
) -> tuple[dict[str, Any], Exception | None]:
    last_error = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            return request_json(client, model, system_prompt, user_message), None
        except Exception as error:
            last_error = error
            if not any(token in str(error).lower() for token in RETRYABLE_ERRORS):
                break
    return {}, last_error
