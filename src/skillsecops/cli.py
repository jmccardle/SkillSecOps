"""SkillSecOps CLI — offline scripts for skill triage.

Usage:
    skillsecops search <query> [--mode keyword|vector] [--limit N]
    skillsecops fetch <github-url> [--cache-dir PATH]
    skillsecops analyze <skill-dir> [--model MODEL]
    skillsecops sign <skill-dir> --key KEY_PATH
    skillsecops verify <catalog-path> --pubkey PUBKEY_PATH
    skillsecops catalog build <skills-dir> --key KEY_PATH
    skillsecops policy check <skill-name> --catalog CATALOG --policy POLICY_FILE

Environment:
    OPENAI_API_KEY      Required for analyze (layers 2-4)
    SKILLSECOPS_KEY     Path to minisign secret key (for sign/catalog commands)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from skillsecops.models import AnalysisReport, AnalysisVerdict

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="skillsecops",
        description="Security-first triage proxy for AI skill repositories.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search SkillNet")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--mode", default="keyword", choices=["keyword", "vector"])
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--threshold", type=float, default=0.8)

    # --- fetch ---
    p_fetch = subparsers.add_parser("fetch", help="Fetch skill at pinned commit")
    p_fetch.add_argument("url", help="GitHub URL of the skill")
    p_fetch.add_argument("--cache-dir", default="./cache", help="Cache directory")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    p_analyze.add_argument("skill_dir", help="Path to skill directory")
    p_analyze.add_argument(
        "--model", default="sonnet",
        help="LLM model: sonnet (fast/cheap), opus (thorough), haiku (fastest)"
    )
    p_analyze.add_argument("--passes", type=int, default=2, help="Summarization passes")
    p_analyze.add_argument(
        "--backend", default="claude", choices=["claude", "openai"],
        help="LLM backend: claude (subscription via CLI) or openai (API key)"
    )

    # --- sign ---
    p_sign = subparsers.add_parser("sign", help="Sign a reviewed skill")
    p_sign.add_argument("skill_dir", help="Path to skill directory")
    p_sign.add_argument("--key", required=True, help="Path to minisign secret key")
    p_sign.add_argument("--output-dir", default="./output", help="Output directory")

    # --- verify ---
    p_verify = subparsers.add_parser("verify", help="Verify catalog and artifacts")
    p_verify.add_argument("catalog", help="Path to output directory with catalog.json")
    p_verify.add_argument("--pubkey", required=True, help="Path to public key")

    # --- catalog build ---
    p_catalog = subparsers.add_parser("catalog", help="Catalog operations")
    catalog_sub = p_catalog.add_subparsers(dest="catalog_command", required=True)
    p_cat_build = catalog_sub.add_parser("build", help="Build catalog from signed skills")
    p_cat_build.add_argument("skills_dir", help="Directory of signed skill tarballs")
    p_cat_build.add_argument("--key", required=True, help="Path to minisign secret key")
    p_cat_build.add_argument("--output-dir", default="./output")

    # --- policy check ---
    p_policy = subparsers.add_parser("policy", help="Policy operations")
    policy_sub = p_policy.add_subparsers(dest="policy_command", required=True)
    p_pol_check = policy_sub.add_parser("check", help="Check skill against policy")
    p_pol_check.add_argument("skill_name", help="Skill name to check")
    p_pol_check.add_argument("--catalog", required=True, help="Path to catalog.json")
    p_pol_check.add_argument("--policy", required=True, help="Path to policy JSON")
    p_pol_check.add_argument("--pubkey", required=True, help="Path to public key")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    dispatch = {
        "search": cmd_search,
        "fetch": cmd_fetch,
        "analyze": cmd_analyze,
        "sign": cmd_sign,
        "verify": cmd_verify,
        "catalog": cmd_catalog,
        "policy": cmd_policy,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


def cmd_search(args: argparse.Namespace) -> None:
    """Search SkillNet, display results with trust warnings."""
    from skillsecops.search import search

    print(
        "WARNING: SkillNet evaluation scores are untrusted. "
        "They are stale, unversioned, and unbound to content.",
        file=sys.stderr,
    )

    results = search(
        args.query,
        mode=args.mode,
        limit=args.limit,
        threshold=args.threshold,
    )

    json.dump(results, sys.stdout, indent=2, default=str)
    print()
    print(f"\n{len(results)} result(s) for '{args.query}'", file=sys.stderr)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch a skill at pinned commit, run Layer 1, report."""
    from skillsecops.analyze.static import analyze_skill
    from skillsecops.fetch import fetch_skill, parse_github_url, resolve_head_commit

    cache_dir = Path(args.cache_dir)
    owner, repo, path = parse_github_url(args.url)
    token = os.environ.get("GITHUB_TOKEN")

    print(f"Resolving HEAD for {owner}/{repo}...", file=sys.stderr)
    commit_sha = resolve_head_commit(owner, repo, token=token)
    print(f"Pinned to commit: {commit_sha}", file=sys.stderr)

    skill_dir, content_hash = fetch_skill(
        owner, repo, path, commit_sha, cache_dir, token=token
    )
    print(f"Fetched to: {skill_dir}", file=sys.stderr)
    print(f"Content SHA-256: {content_hash}", file=sys.stderr)

    print("\nRunning Layer 1 (static analysis)...", file=sys.stderr)
    result = analyze_skill(skill_dir)

    output = {
        "skill_dir": str(skill_dir),
        "commit_sha": commit_sha,
        "content_sha256": content_hash,
        "static_analysis": {
            "verdict": result.verdict.value,
            "patterns_checked": result.patterns_checked,
            "patterns_matched": result.patterns_matched,
        },
    }
    json.dump(output, sys.stdout, indent=2)
    print()

    if result.verdict == AnalysisVerdict.FAIL:
        print(
            f"\nSTATIC ANALYSIS FAILED: {len(result.patterns_matched)} pattern(s) matched",
            file=sys.stderr,
        )
    else:
        print("\nStatic analysis passed.", file=sys.stderr)


def _make_client(args: argparse.Namespace):
    """Create LLM client based on --backend flag."""
    backend = getattr(args, "backend", "claude")
    if backend == "claude":
        from skillsecops.llm import ClaudeClient
        return ClaudeClient(default_model=args.model)
    else:
        import openai
        return openai.OpenAI()


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full analysis pipeline (layers 1-4)."""
    from skillsecops.analyze.crossref import crossref_skill
    from skillsecops.analyze.sandbox import sandbox_skill
    from skillsecops.analyze.static import analyze_skill
    from skillsecops.analyze.summarize import summarize_skill

    skill_dir = Path(args.skill_dir)
    if not skill_dir.is_dir():
        print(f"ERROR: {skill_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = AnalysisReport(
        skill_name=skill_dir.name,
        source_url="",
        pinned_commit="",
        content_sha256="",
    )

    # Layer 1: Static (no LLM)
    print("Layer 1: Static analysis...", file=sys.stderr)
    report.static = analyze_skill(skill_dir)
    print(f"  Verdict: {report.static.verdict.value}", file=sys.stderr)

    if report.static.verdict == AnalysisVerdict.FAIL:
        print("  Short-circuiting: static analysis failed.", file=sys.stderr)
        json.dump(json.loads(report.model_dump_json()), sys.stdout, indent=2)
        print()
        sys.exit(1)

    # Build LLM client for layers 2-4
    client = _make_client(args)
    print(f"  LLM backend: {args.backend} (model: {args.model})", file=sys.stderr)

    # Layer 2: Summarization
    print(f"Layer 2: Chunked summarization ({args.passes} passes)...", file=sys.stderr)
    report.summarization = summarize_skill(
        skill_dir, client=client, model=args.model, num_passes=args.passes
    )
    print(f"  Verdict: {report.summarization.verdict.value}", file=sys.stderr)
    if report.summarization.flags:
        for flag in report.summarization.flags:
            print(f"    Flag: {flag}", file=sys.stderr)

    if report.summarization.verdict == AnalysisVerdict.FAIL:
        print("  Short-circuiting: summarization failed.", file=sys.stderr)
        json.dump(json.loads(report.model_dump_json()), sys.stdout, indent=2)
        print()
        sys.exit(1)

    # Layer 3: Cross-reference (deterministic, no LLM)
    print("Layer 3: Cross-reference...", file=sys.stderr)
    report.crossref = crossref_skill(skill_dir, report.summarization)
    print(f"  Verdict: {report.crossref.verdict.value}", file=sys.stderr)
    if report.crossref.undeclared_capabilities:
        for cap in report.crossref.undeclared_capabilities:
            print(f"    Undeclared: {cap}", file=sys.stderr)

    if report.crossref.verdict == AnalysisVerdict.FAIL:
        print("  Short-circuiting: cross-reference failed.", file=sys.stderr)
        json.dump(json.loads(report.model_dump_json()), sys.stdout, indent=2)
        print()
        sys.exit(1)

    # Layer 4: Sandbox
    print("Layer 4: Honeypot sandbox...", file=sys.stderr)
    report.sandbox = sandbox_skill(skill_dir, client=client, model=args.model)
    print(f"  Verdict: {report.sandbox.verdict.value}", file=sys.stderr)
    print(f"    Tool calls: {len(report.sandbox.tool_calls)}", file=sys.stderr)
    print(f"    Canary leaked: {report.sandbox.canary_leaked}", file=sys.stderr)

    # Print cost summary if using Claude backend
    if hasattr(client, 'total_cost_usd'):
        print(f"\n  Total LLM cost: ${client.total_cost_usd:.4f} ({client.call_count} calls)", file=sys.stderr)

    json.dump(json.loads(report.model_dump_json()), sys.stdout, indent=2)
    print()

    if report.overall_verdict == AnalysisVerdict.PASS:
        print("\nAll layers passed. Skill is ready for human review.", file=sys.stderr)
    else:
        print(f"\nAnalysis result: {report.overall_verdict.value}", file=sys.stderr)
        sys.exit(1)


def cmd_sign(args: argparse.Namespace) -> None:
    """Sign a reviewed skill."""
    from skillsecops.catalog import build_tarball, sha256_file, sign_skill

    skill_dir = Path(args.skill_dir)
    output_dir = Path(args.output_dir)
    secret_key = Path(args.key)

    tarball = build_tarball(skill_dir, output_dir / "skills")
    content_hash = sha256_file(tarball)
    sig = sign_skill(tarball, secret_key, skill_dir.name, "manual")

    print(f"Tarball: {tarball}", file=sys.stderr)
    print(f"SHA-256: {content_hash}", file=sys.stderr)
    print(f"Signature: {sig}", file=sys.stderr)

    json.dump({
        "tarball": str(tarball),
        "content_sha256": content_hash,
        "signature": str(sig),
    }, sys.stdout, indent=2)
    print()


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify catalog and all artifacts."""
    from skillsecops.catalog import verify_all

    output_dir = Path(args.catalog)
    pubkey = Path(args.pubkey)

    verify_all(output_dir, pubkey)
    print("Verification successful.", file=sys.stderr)


def cmd_catalog(args: argparse.Namespace) -> None:
    """Catalog operations."""
    if args.catalog_command == "build":
        _cmd_catalog_build(args)


def _cmd_catalog_build(args: argparse.Namespace) -> None:
    """Build catalog from signed skills."""
    from skillsecops.catalog import (
        Catalog,
        build_catalog,
        parse_minisign_pubkey,
        sign_catalog,
    )

    # This is a placeholder that builds an empty catalog.
    # In practice, this would scan a directory of analysis reports.
    output_dir = Path(args.output_dir)
    secret_key = Path(args.key)

    catalog = Catalog(version=1, signers={})
    catalog_path = sign_catalog(catalog, output_dir, secret_key)

    print(f"Catalog written and signed: {catalog_path}", file=sys.stderr)


def cmd_policy(args: argparse.Namespace) -> None:
    """Policy operations."""
    if args.policy_command == "check":
        _cmd_policy_check(args)


def _cmd_policy_check(args: argparse.Namespace) -> None:
    """Check skill against operator policy."""
    from skillsecops.catalog import verify_catalog
    from skillsecops.policy import check_skill, load_policy

    pubkey = Path(args.pubkey)
    catalog = verify_catalog(Path(args.catalog), pubkey)
    policy = load_policy(Path(args.policy))

    entry = None
    for e in catalog.entries:
        if e.skill_name == args.skill_name:
            entry = e
            break

    if entry is None:
        print(f"Skill '{args.skill_name}' not found in catalog.", file=sys.stderr)
        sys.exit(1)

    allowed, reasons = check_skill(entry, policy)

    result = {
        "skill_name": args.skill_name,
        "allowed": allowed,
        "reasons": reasons,
    }
    json.dump(result, sys.stdout, indent=2)
    print()

    if not allowed:
        print(f"BLOCKED: {'; '.join(reasons)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("ALLOWED", file=sys.stderr)
