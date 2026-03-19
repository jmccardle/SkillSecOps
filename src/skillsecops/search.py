"""Query the SkillNet public API. Returns raw results with no trust assumptions.

Results include upstream evaluation scores, author strings, and mutable GitHub
URLs. None of these are trusted — they are display hints for the human reviewer
deciding what to triage next.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import requests

logger = logging.getLogger(__name__)

SKILLNET_API = "http://api-skillnet.openkg.cn"


def search(
    query: str,
    *,
    mode: Literal["keyword", "vector"] = "keyword",
    category: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    min_stars: int = 0,
    sort_by: str = "stars",
    threshold: float = 0.8,
    api_url: str = SKILLNET_API,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Search SkillNet. Returns raw API results as dicts.

    The caller is responsible for treating these as untrusted metadata.
    Evaluation scores, author fields, and skill URLs are upstream claims
    with no cryptographic binding to content.
    """
    params: dict[str, Any] = {
        "q": query,
        "mode": mode,
        "limit": limit,
    }
    if category is not None:
        params["category"] = category

    if mode == "keyword":
        params.update(page=page, min_stars=min_stars, sort_by=sort_by)
    elif mode == "vector":
        params["threshold"] = threshold

    response = requests.get(
        f"{api_url.rstrip('/')}/v1/search",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    body = response.json()
    if not body.get("success", False):
        logger.warning("SkillNet API returned success=false for query %r", query)
        return []

    return body.get("data", [])
