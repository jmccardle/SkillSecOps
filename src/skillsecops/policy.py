"""Operator policy enforcement.

Checks skill catalog entries against a local, unsigned policy that
defines what an agent runtime is allowed to use. The policy specifies
trusted reviewers, required signature counts, allowed advisory flags,
and blocked skills.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skillsecops.models import CatalogEntry, OperatorPolicy, SkillAdvisory

logger = logging.getLogger(__name__)


def load_policy(path: Path) -> OperatorPolicy:
    """Load operator policy from a JSON file.

    Raises FileNotFoundError if file doesn't exist.
    Raises ValueError if JSON is invalid or doesn't match schema.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return OperatorPolicy.model_validate(data)


def check_advisories(
    skill_advisory: SkillAdvisory,
    allowed: SkillAdvisory,
) -> list[str]:
    """Compare skill advisory flags against allowed advisories.

    Returns list of violation descriptions.
    For each boolean field: if skill says True but policy says False, violation.
    For tool_invocations: skill's tools must be a subset of allowed tools.
    """
    violations: list[str] = []

    bool_fields = ["network", "system_access", "external_services", "file_write", "credential_access"]
    for field in bool_fields:
        skill_val = getattr(skill_advisory, field)
        allowed_val = getattr(allowed, field)
        if skill_val and not allowed_val:
            violations.append(
                f"Skill declares {field}=True but policy allows {field}=False"
            )

    # Check tool invocations
    allowed_tools = {t.lower() for t in allowed.tool_invocations}
    skill_tools = {t.lower() for t in skill_advisory.tool_invocations}
    disallowed = skill_tools - allowed_tools
    if disallowed and allowed_tools:  # only check if policy specifies allowed tools
        violations.append(
            f"Skill uses tools not in allowed list: {disallowed}"
        )

    return violations


def check_skill(
    entry: CatalogEntry,
    policy: OperatorPolicy,
) -> tuple[bool, list[str]]:
    """Check a skill catalog entry against operator policy.

    Returns (allowed, reasons). Reasons explain why blocked (empty if allowed).
    """
    reasons: list[str] = []

    # 1. Blocked list
    if entry.skill_name in policy.blocked_skills:
        reasons.append(f"Skill '{entry.skill_name}' is in the blocked list")

    # 2. Trusted reviewers
    if policy.trusted_reviewers and entry.signer not in policy.trusted_reviewers:
        reasons.append(
            f"Signer '{entry.signer}' is not in trusted reviewers: "
            f"{policy.trusted_reviewers}"
        )

    # 3. Advisory violations
    advisory_violations = check_advisories(entry.advisory, policy.allowed_advisories)
    reasons.extend(advisory_violations)

    return (len(reasons) == 0, reasons)
