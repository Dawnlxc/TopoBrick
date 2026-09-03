from __future__ import annotations

from typing import Any

from topobrick.sampler.agents.client import request_with_retry
from topobrick.sampler.agents.prompts import SAMPLER_SYSTEM_PROMPT


def select_exogenous_actions(
    client, model: str, topology_context: str
) -> tuple[dict[str, Any], Exception | None]:
    return request_with_retry(
        client,
        model,
        SAMPLER_SYSTEM_PROMPT,
        topology_context,
    )
