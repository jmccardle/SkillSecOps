"""Fetch a skill from GitHub at a specific commit SHA.

Downloads text files only (SKILL.md, references, scripts) via the GitHub
Contents API, pins to an exact commit, and computes a content hash over
the normalized file tree. The result lands in cache/untrusted/ and is
NOT trusted until it passes the analysis pipeline and is signed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Only fetch text-like files. Refuse binaries.
TEXT_EXTENSIONS = frozenset({
    ".md", ".txt", ".py", ".sh", ".bash", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg", ".rst", ".csv",
})


def parse_github_url(url: str) -> tuple[str, str, str]:
    """Extract (owner, repo, path) from a GitHub tree URL.

    Raises ValueError if the URL doesn't match expected patterns.
    """
    # https://github.com/owner/repo/tree/branch/path/to/skill
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/tree/[^/]+/(.+)", url
    )
    if match:
        return match.group(1), match.group(2), match.group(3)

    # https://github.com/owner/repo (root)
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?$", url)
    if match:
        return match.group(1), match.group(2), ""

    raise ValueError(f"Cannot parse GitHub URL: {url}")


def resolve_head_commit(
    owner: str,
    repo: str,
    branch: str = "main",
    *,
    token: Optional[str] = None,
    timeout: int = 15,
) -> str:
    """Get the current HEAD commit SHA for a branch."""
    headers = _auth_headers(token)
    resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/commits/{branch}",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["sha"]


def fetch_skill(
    owner: str,
    repo: str,
    path: str,
    commit_sha: str,
    cache_dir: Path,
    *,
    token: Optional[str] = None,
    timeout: int = 15,
) -> tuple[Path, str]:
    """Download a skill tree at a pinned commit into cache/untrusted/.

    Returns (skill_dir, content_sha256).
    """
    skill_name = path.rstrip("/").split("/")[-1] if path else repo
    dest = cache_dir / "untrusted" / f"{skill_name}-{commit_sha[:12]}"

    if dest.exists():
        content_hash = _hash_directory(dest)
        logger.info("Cache hit: %s (%s)", dest, content_hash[:16])
        return dest, content_hash

    dest.mkdir(parents=True, exist_ok=True)

    headers = _auth_headers(token)
    _fetch_tree_recursive(
        owner, repo, path, commit_sha, dest, headers, timeout
    )

    content_hash = _hash_directory(dest)
    logger.info("Fetched %s at %s → %s (%s)", path, commit_sha[:12], dest, content_hash[:16])
    return dest, content_hash


def _fetch_tree_recursive(
    owner: str,
    repo: str,
    path: str,
    ref: str,
    dest: Path,
    headers: dict,
    timeout: int,
) -> None:
    """Recursively fetch directory contents, text files only."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()

    items = resp.json()
    if not isinstance(items, list):
        items = [items]

    for item in items:
        name = item["name"]
        item_type = item["type"]

        if item_type == "dir":
            (dest / name).mkdir(exist_ok=True)
            _fetch_tree_recursive(
                owner, repo, item["path"], ref, dest / name, headers, timeout
            )
        elif item_type == "file":
            ext = os.path.splitext(name)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                logger.debug("Skipping non-text file: %s", name)
                continue
            _fetch_file(item, dest / name, headers, timeout)


def _fetch_file(
    item: dict, dest_path: Path, headers: dict, timeout: int
) -> None:
    """Download a single file via its download_url."""
    download_url = item.get("download_url")
    if not download_url:
        logger.warning("No download_url for %s", item.get("path"))
        return

    resp = requests.get(download_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    dest_path.write_text(resp.text, encoding="utf-8")


def _hash_directory(directory: Path) -> str:
    """Compute a deterministic SHA-256 over sorted file contents."""
    hasher = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(directory)
            hasher.update(str(rel).encode("utf-8"))
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _auth_headers(token: Optional[str]) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
