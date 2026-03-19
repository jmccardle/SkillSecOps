# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

SkillSecOps is a security-first triage proxy for AI skill repositories like SkillNet. It adds version pinning, cryptographic signatures (minisign/Ed25519), multi-layer analysis, and role-separated review to an ecosystem that has none of these.

SkillSecOps is NOT a fork of SkillNet. It consumes the SkillNet public REST API (`api-skillnet.openkg.cn/v1`) and GitHub directly. It provides its own incompatible API because SkillNet's API lacks version binding, content integrity, and provenance.

## Architecture

Offline scripts organized as importable modules — designed for CLI use now, web interface later.

```
skillsecops/
├── search.py        — Query SkillNet API, return results with no trust assumptions
├── fetch.py         — Download skill at specific GitHub commit SHA, hash contents
├── analyze/
│   ├── static.py    — Layer 1: deterministic pattern matching (no LLM)
│   ├── summarize.py — Layer 2: chunked LLM summarization (inspector-as-decompiler)
│   ├── crossref.py  — Layer 3: compare summaries against declared metadata
│   └── sandbox.py   — Layer 4: honeypot execution with mock tools
├── catalog.py       — Signed catalog management (minisign, modeled on ageless store)
├── policy.py        — Operator trust policy enforcement
└── models.py        — Shared data models
```

### Analysis Pipeline

Layer 1 (static) is deterministic and sound — catches known injection patterns with certainty.
Layer 2 (summarize) is probabilistic — inspector LLM summarizes chunks, never judges safety.
Layer 3 (crossref) compares summaries to metadata — never sees raw skill content.
Layer 4 (sandbox) runs skills with mock tools — any undeclared tool invocation is a signal.

A skill must pass all four layers before a human reviewer is asked to sign off.

### Trust Model

Three artifacts, three roles:
- **Skill artifact** (tarball + .minisig): signed by proxy, means "contents match this SHA at this commit"
- **Analysis report** (in signed catalog): signed by reviewer(s), means "behavior matches declared advisory"
- **Operator policy** (local config): unsigned, means "my agents may use skills with these signatures and advisories"

Security reviewers and operators are deliberately different roles. Reviewers handle cybersecurity triage. Operators authorize tools based on domain expertise, trusting reviewer signatures.

### Signing

Uses minisign (Ed25519), same scheme as the ageless linux store (~/Development/agelesslinux/store.agelesslinux.org). Key properties:
- Catalog is signed with trusted comments (anti-downgrade via catalog version)
- Multi-signer support for reviewer counter-signatures
- Revocations via signed revocations.json
- Signature is the trust boundary, not the transport

## Development

```bash
pip install -e .
```

No external services required for Layer 1 (static analysis). Layers 2-3 require an OpenAI-compatible LLM endpoint. Layer 4 requires only Python.

## Key Design Constraints

- **Never trust SkillNet evaluation scores.** They are stale, unversioned, and unbound to content.
- **The inspector LLM is a decompiler, not a judge.** It summarizes; a separate system decides if the summary indicates danger.
- **Static analysis is the bedrock.** It's the only layer that is sound (no false negatives for known patterns).
- **All skills are untrusted until signed by a reviewer.** The cache/untrusted/ directory exists to enforce this.
- **No LLM generates attack payloads.** Test injections are template-based (see tests/payloads/).
