"""Tests for operator policy enforcement."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from skillsecops.models import (
    AnalysisVerdict,
    CatalogEntry,
    OperatorPolicy,
    SkillAdvisory,
)
from skillsecops.policy import check_advisories, check_skill, load_policy


def _make_entry(**overrides) -> CatalogEntry:
    defaults = dict(
        skill_name="csv-tool",
        source_url="https://github.com/test/repo",
        pinned_commit="abc123",
        content_sha256="dead" * 16,
        upstream_author="author",
        signer="reviewer-2026",
        catalog_version=1,
        analyzed_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return CatalogEntry(**defaults)


class TestPolicyLoading:
    def test_load_valid_policy(self, tmp_path):
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps({
            "trusted_reviewers": ["reviewer-2026"],
            "require_signatures": 1,
            "blocked_skills": ["evil-tool"],
        }))

        policy = load_policy(policy_file)
        assert policy.trusted_reviewers == ["reviewer-2026"]
        assert "evil-tool" in policy.blocked_skills

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_policy(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path):
        policy_file = tmp_path / "bad.json"
        policy_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_policy(policy_file)


class TestCheckAdvisories:
    def test_no_violations(self):
        skill = SkillAdvisory(network=False, system_access=False)
        allowed = SkillAdvisory(network=False, system_access=False)
        assert check_advisories(skill, allowed) == []

    def test_network_violation(self):
        skill = SkillAdvisory(network=True)
        allowed = SkillAdvisory(network=False)
        violations = check_advisories(skill, allowed)
        assert any("network" in v for v in violations)

    def test_multiple_violations(self):
        skill = SkillAdvisory(
            network=True, system_access=True, credential_access=True
        )
        allowed = SkillAdvisory()
        violations = check_advisories(skill, allowed)
        assert len(violations) == 3

    def test_tool_invocation_violation(self):
        skill = SkillAdvisory(tool_invocations=["read_file", "send_http"])
        allowed = SkillAdvisory(tool_invocations=["read_file"])
        violations = check_advisories(skill, allowed)
        assert any("send_http" in v for v in violations)

    def test_empty_allowed_tools_permits_all(self):
        skill = SkillAdvisory(tool_invocations=["anything"])
        allowed = SkillAdvisory(tool_invocations=[])
        violations = check_advisories(skill, allowed)
        assert violations == []


class TestCheckSkill:
    def test_allowed_skill(self):
        entry = _make_entry()
        policy = OperatorPolicy(trusted_reviewers=["reviewer-2026"])
        allowed, reasons = check_skill(entry, policy)
        assert allowed
        assert reasons == []

    def test_blocked_skill_name(self):
        entry = _make_entry(skill_name="evil-tool")
        policy = OperatorPolicy(blocked_skills=["evil-tool"])
        allowed, reasons = check_skill(entry, policy)
        assert not allowed
        assert any("blocked" in r for r in reasons)

    def test_untrusted_reviewer(self):
        entry = _make_entry(signer="unknown-reviewer")
        policy = OperatorPolicy(trusted_reviewers=["reviewer-2026"])
        allowed, reasons = check_skill(entry, policy)
        assert not allowed
        assert any("trusted reviewers" in r for r in reasons)

    def test_empty_trusted_reviewers_trusts_all(self):
        entry = _make_entry(signer="anyone")
        policy = OperatorPolicy(trusted_reviewers=[])
        allowed, reasons = check_skill(entry, policy)
        assert allowed

    def test_advisory_violation_blocks(self):
        entry = _make_entry(
            advisory=SkillAdvisory(network=True)
        )
        policy = OperatorPolicy(
            trusted_reviewers=["reviewer-2026"],
            allowed_advisories=SkillAdvisory(network=False),
        )
        allowed, reasons = check_skill(entry, policy)
        assert not allowed
        assert any("network" in r for r in reasons)
