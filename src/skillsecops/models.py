"""Data models for SkillSecOps.

Defines the catalog schema, analysis results, and operator policy.
These are the contract between offline scripts and any future web interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

class AnalysisVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class StaticAnalysisResult(BaseModel):
    """Layer 1: deterministic pattern matching, no LLM."""
    verdict: AnalysisVerdict
    patterns_checked: int
    patterns_matched: list[str] = Field(default_factory=list)
    details: str = ""


class ChunkSummary(BaseModel):
    """Layer 2 output: what one chunk of a skill does, per the inspector LLM."""
    chunk_index: int
    chunk_hash: str
    summary: str
    declared_tools: list[str] = Field(default_factory=list)
    declared_capabilities: list[str] = Field(default_factory=list)
    instructions_to_agent: list[str] = Field(default_factory=list)
    schema_violation: bool = False
    raw_response_tokens: int = 0


class SummarizationResult(BaseModel):
    """Layer 2 aggregate."""
    verdict: AnalysisVerdict
    chunk_count: int
    pass_count: int
    chunks: list[ChunkSummary] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class CrossRefResult(BaseModel):
    """Layer 3: summaries vs declared metadata."""
    verdict: AnalysisVerdict
    declared_description: str
    mismatches: list[str] = Field(default_factory=list)
    undeclared_capabilities: list[str] = Field(default_factory=list)


class SandboxToolCall(BaseModel):
    """A single tool invocation observed in the Layer 4 honeypot."""
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    timestamp: datetime


class SandboxResult(BaseModel):
    """Layer 4: honeypot execution."""
    verdict: AnalysisVerdict
    tool_calls: list[SandboxToolCall] = Field(default_factory=list)
    undeclared_tool_calls: list[SandboxToolCall] = Field(default_factory=list)
    canary_leaked: bool = False
    details: str = ""


class AnalysisReport(BaseModel):
    """Full pipeline result for one skill at one version."""
    skill_name: str
    source_url: str
    pinned_commit: str
    content_sha256: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    static: Optional[StaticAnalysisResult] = None
    summarization: Optional[SummarizationResult] = None
    crossref: Optional[CrossRefResult] = None
    sandbox: Optional[SandboxResult] = None

    @property
    def overall_verdict(self) -> AnalysisVerdict:
        layers = [self.static, self.summarization, self.crossref, self.sandbox]
        for layer in layers:
            if layer is not None and layer.verdict != AnalysisVerdict.PASS:
                return layer.verdict
        if any(layer is None for layer in layers):
            return AnalysisVerdict.ERROR
        return AnalysisVerdict.PASS


# ---------------------------------------------------------------------------
# Catalog (signed artifact index)
# ---------------------------------------------------------------------------

class SkillAdvisory(BaseModel):
    """Declared capability flags — what the skill says it needs."""
    network: bool = False
    system_access: bool = False
    tool_invocations: list[str] = Field(default_factory=list)
    external_services: bool = False
    file_write: bool = False
    credential_access: bool = False


class CatalogEntry(BaseModel):
    """One skill version in the signed catalog."""
    skill_name: str
    source_url: str
    pinned_commit: str
    content_sha256: str
    upstream_author: str
    signer: str
    catalog_version: int
    analyzed_at: datetime
    advisory: SkillAdvisory = Field(default_factory=SkillAdvisory)
    analysis_summary: str = ""
    static_verdict: AnalysisVerdict = AnalysisVerdict.ERROR
    summarization_verdict: AnalysisVerdict = AnalysisVerdict.ERROR
    crossref_verdict: AnalysisVerdict = AnalysisVerdict.ERROR
    sandbox_verdict: AnalysisVerdict = AnalysisVerdict.ERROR


class Catalog(BaseModel):
    """The signed skill catalog — analogous to ageless store's catalog.json."""
    version: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entries: list[CatalogEntry] = Field(default_factory=list)
    signers: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Operator policy (local, unsigned)
# ---------------------------------------------------------------------------

class OperatorPolicy(BaseModel):
    """Local policy for what an agent runtime may use.

    This is not signed — it's the operator's local configuration.
    The agent runtime enforces this by checking skill signatures
    and advisory flags before allowing tool use.
    """
    trusted_reviewers: list[str] = Field(default_factory=list)
    require_signatures: int = 1
    allowed_advisories: SkillAdvisory = Field(default_factory=SkillAdvisory)
    blocked_skills: list[str] = Field(default_factory=list)
    auto_approve_matching: bool = False
