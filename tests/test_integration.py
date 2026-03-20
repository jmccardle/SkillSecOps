"""Integration tests — end-to-end pipeline with mock LLM.

Validates that the 4-layer pipeline works together correctly:
- Benign skills pass all layers
- Known injections are caught at Layer 1 (never reach LLM)
- Evasion payloads are caught at Layer 2 or 3
- Pipeline short-circuits on failure
"""

from __future__ import annotations

import json

import pytest

from skillsecops.analyze.crossref import crossref_skill
from skillsecops.analyze.sandbox import sandbox_skill
from skillsecops.analyze.static import analyze_skill
from skillsecops.analyze.summarize import summarize_skill
from skillsecops.models import AnalysisReport, AnalysisVerdict
from tests.conftest import MockOpenAIClient, make_inspector_response, make_mock_response
from tests.payloads.templates import (
    BASIC_OVERRIDE,
    BENIGN_CSV_FORMATTER,
    SPLIT_INJECTION,
    SUBTLE_ROLE_SHIFT,
)
from tests.test_sandbox import SandboxMockClient, _make_text_response


def _run_pipeline(
    skill_dir,
    inspector_client,
    sandbox_client,
    model="gpt-4o-mini",
    num_passes=1,
) -> AnalysisReport:
    """Run the full 4-layer pipeline, short-circuiting on failure."""
    report = AnalysisReport(
        skill_name=skill_dir.name,
        source_url="",
        pinned_commit="test",
        content_sha256="test",
    )

    # Layer 1
    report.static = analyze_skill(skill_dir)
    if report.static.verdict == AnalysisVerdict.FAIL:
        return report

    # Layer 2
    report.summarization = summarize_skill(
        skill_dir, client=inspector_client, model=model, num_passes=num_passes,
    )
    if report.summarization.verdict == AnalysisVerdict.FAIL:
        return report

    # Layer 3
    report.crossref = crossref_skill(skill_dir, report.summarization)
    if report.crossref.verdict == AnalysisVerdict.FAIL:
        return report

    # Layer 4
    report.sandbox = sandbox_skill(skill_dir, client=sandbox_client, model=model)

    return report


class TestBenignSkillFullPipeline:
    def test_passes_all_layers(self, tmp_skill_dir):
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)

        inspector = MockOpenAIClient(responses=[
            make_inspector_response(
                summary="Formats CSV files with consistent delimiters and RFC 4180 quoting.",
                capabilities_described=["text processing", "csv formatting"],
            ),
        ])
        sandbox = SandboxMockClient([
            _make_text_response(
                "I understand this is a CSV formatting skill. Ready to format."
            ),
        ])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        assert report.static is not None
        assert report.static.verdict == AnalysisVerdict.PASS
        assert report.summarization is not None
        assert report.summarization.verdict == AnalysisVerdict.PASS
        assert report.crossref is not None
        assert report.crossref.verdict == AnalysisVerdict.PASS
        assert report.sandbox is not None
        assert report.sandbox.verdict == AnalysisVerdict.PASS
        assert report.overall_verdict == AnalysisVerdict.PASS


class TestInjectionCaughtAtLayer1:
    def test_basic_override_short_circuits(self, tmp_skill_dir):
        """Known injection caught at Layer 1 — never reaches LLM."""
        skill_dir = tmp_skill_dir(BASIC_OVERRIDE.content)

        # These clients should never be called
        inspector = MockOpenAIClient()
        sandbox = SandboxMockClient([_make_text_response("should not reach")])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        assert report.static is not None
        assert report.static.verdict == AnalysisVerdict.FAIL
        # Layer 2 should NOT have run
        assert report.summarization is None
        assert report.crossref is None
        assert report.sandbox is None
        # Verify LLM was never called
        assert len(inspector.calls) == 0
        assert len(sandbox.calls) == 0


class TestEvasionPayloadsCaughtDeeper:
    def test_subtle_role_shift_caught_at_layer2_or_3(self, tmp_skill_dir):
        """Evasion payload passes Layer 1 but caught at Layer 2 or 3."""
        skill_dir = tmp_skill_dir(SUBTLE_ROLE_SHIFT.content)

        # The inspector LLM accurately summarizes the content
        inspector = MockOpenAIClient(responses=[
            make_inspector_response(
                summary="Requests reading .env file for credential review and audit.",
                tools_referenced=["read_file"],
                capabilities_described=["credential access", "file reading"],
                instructions_to_agent=["Read the .env file and include contents in review"],
            ),
        ])
        sandbox = SandboxMockClient([_make_text_response("done")])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        # Should pass Layer 1
        assert report.static is not None
        assert report.static.verdict == AnalysisVerdict.PASS
        # Should fail at Layer 2 (instructions_to_agent) or Layer 3 (undeclared capabilities)
        assert report.overall_verdict == AnalysisVerdict.FAIL

    def test_split_injection_caught_at_layer3(self, tmp_skill_dir):
        """Split injection passes L1 and L2 but L3 cross-ref catches mismatch."""
        skill_dir = tmp_skill_dir(SPLIT_INJECTION.content)

        # Inspector returns clean summaries per chunk (no instructions_to_agent)
        # but the summaries reveal credential + network behavior
        inspector = MockOpenAIClient(responses=[
            make_inspector_response(
                summary="Generates API client code with authentication setup "
                        "using environment credentials and sends test requests.",
                tools_referenced=["send_http"],
                capabilities_described=["network access", "credential access", "api client generation"],
            ),
        ])
        sandbox = SandboxMockClient([_make_text_response("done")])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        assert report.static is not None
        assert report.static.verdict == AnalysisVerdict.PASS
        # Should fail at Layer 3 (undeclared network/credential capabilities)
        assert report.overall_verdict == AnalysisVerdict.FAIL
        assert report.crossref is not None
        assert report.crossref.verdict == AnalysisVerdict.FAIL


class TestPipelineShortCircuit:
    def test_no_llm_tokens_wasted_on_l1_fail(self, tmp_skill_dir):
        """When L1 fails, no LLM calls should be made."""
        skill_dir = tmp_skill_dir(BASIC_OVERRIDE.content)

        inspector = MockOpenAIClient()
        sandbox = SandboxMockClient([_make_text_response("x")])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        assert report.static.verdict == AnalysisVerdict.FAIL
        assert inspector.calls == []
        assert sandbox.calls == []

    def test_no_sandbox_on_l2_fail(self, tmp_skill_dir):
        """When L2 fails, L3 and L4 should not run."""
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)

        # Force L2 failure with invalid response
        inspector = MockOpenAIClient(responses=[
            make_mock_response("this is not valid JSON"),
        ])
        sandbox = SandboxMockClient([_make_text_response("x")])

        report = _run_pipeline(skill_dir, inspector, sandbox)

        assert report.static.verdict == AnalysisVerdict.PASS
        assert report.summarization.verdict == AnalysisVerdict.FAIL
        assert report.crossref is None
        assert report.sandbox is None
        assert sandbox.calls == []
