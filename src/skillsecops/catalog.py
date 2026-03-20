"""Catalog signing — minisign-based supply chain integrity.

Ported from ~/Development/agelesslinux/store.agelesslinux.org/ageless_publish.py.
Uses Ed25519 via minisign for signing, deterministic tarballs for reproducibility,
and a signed catalog with trusted comments for anti-downgrade protection.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from skillsecops.models import (
    AnalysisReport,
    AnalysisVerdict,
    Catalog,
    CatalogEntry,
    SkillAdvisory,
)

logger = logging.getLogger(__name__)

TARBALL_MTIME = 0
TARBALL_UID = 0
TARBALL_GID = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def require_minisign() -> None:
    """Verify minisign is on PATH. Raise RuntimeError if not."""
    if not shutil.which("minisign"):
        raise RuntimeError(
            "minisign is not installed or not on PATH. "
            "Install it from https://jedisct1.github.io/minisign/"
        )


def _run_minisign(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run minisign subprocess."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        raise RuntimeError(f"minisign failed: {' '.join(args)}\n  {stderr}")
    return result


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(data: dict, path: Path) -> None:
    """Write JSON with deterministic formatting."""
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _now_utc() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Minisign operations (ported from ageless_publish.py)
# ---------------------------------------------------------------------------

def sign_file(file_path: Path, secret_key: Path, trusted_comment: str) -> Path:
    """Sign a file with minisign. Returns path to .minisig file.

    Command: minisign -S -s <key> -m <file> -t <trusted_comment>
    """
    require_minisign()
    file_path = Path(file_path)
    _run_minisign([
        "minisign", "-S",
        "-s", str(secret_key),
        "-m", str(file_path),
        "-t", trusted_comment,
    ])
    sig_path = file_path.with_suffix(file_path.suffix + ".minisig")
    if not sig_path.is_file():
        raise RuntimeError(f"minisign produced no signature at {sig_path}")
    return sig_path


def verify_file(file_path: Path, pubkey_path: Path) -> None:
    """Verify a file's minisign signature. Raises RuntimeError on failure.

    Command: minisign -Vm <file> -p <pubkey>
    """
    require_minisign()
    file_path = Path(file_path)
    sig_path = file_path.with_suffix(file_path.suffix + ".minisig")
    if not sig_path.is_file():
        raise RuntimeError(f"Signature file not found: {sig_path}")
    result = _run_minisign(
        ["minisign", "-Vm", str(file_path), "-p", str(pubkey_path)],
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        raise RuntimeError(
            f"Signature verification failed: {file_path}\n"
            f"  pubkey: {pubkey_path}\n  {stderr}"
        )


def parse_minisign_pubkey(pubkey_path: Path) -> tuple[str, str]:
    """Parse a minisign public key file.

    Returns (key_id_hex, base64_key_line).
    Key ID is bytes 2-10 of the decoded pubkey.
    """
    pubkey_path = Path(pubkey_path)
    lines = pubkey_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(
            f"{pubkey_path}: expected at least 2 lines (comment + base64 key)"
        )
    b64_line = lines[1].strip()
    try:
        raw = base64.b64decode(b64_line)
    except Exception as e:
        raise ValueError(f"{pubkey_path}: invalid base64 in public key: {e}") from e
    if len(raw) < 10:
        raise ValueError(
            f"{pubkey_path}: decoded key too short ({len(raw)} bytes, expected >= 10)"
        )
    key_id_bytes = raw[2:10]
    key_id_hex = key_id_bytes.hex().upper()
    return key_id_hex, b64_line


# ---------------------------------------------------------------------------
# Tarball operations
# ---------------------------------------------------------------------------

def build_tarball(skill_dir: Path, output_dir: Path) -> Path:
    """Build a deterministic tar.gz from a skill directory.

    Properties (from ageless_publish.py pattern):
    - mtime=0 for all entries
    - uid/gid=0, uname/gname=""
    - Sorted traversal order
    - gzip mtime=0

    Returns path to the tarball.
    """
    skill_name = skill_dir.name
    tarball_name = f"{skill_name}.tar.gz"
    tarball_path = output_dir / tarball_name

    entries: list[tuple[str, Path | None, bool]] = []

    # Root directory
    entries.append((skill_name, None, True))

    # Sorted walk of the skill directory
    for root, dirs, files in os.walk(skill_dir):
        dirs.sort()
        rel = Path(root).relative_to(skill_dir)

        if rel != Path("."):
            entries.append((f"{skill_name}/{rel}", None, True))

        for fname in sorted(files):
            src = Path(root) / fname
            if rel == Path("."):
                archive_name = f"{skill_name}/{fname}"
            else:
                archive_name = f"{skill_name}/{rel}/{fname}"
            entries.append((archive_name, src, False))

    # Build tar into BytesIO, then gzip with mtime=0
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for archive_name, src_path, is_dir in entries:
            info = tarfile.TarInfo(name=archive_name)
            info.mtime = TARBALL_MTIME
            info.uid = TARBALL_UID
            info.gid = TARBALL_GID
            info.uname = ""
            info.gname = ""

            if is_dir:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            else:
                data = src_path.read_bytes()
                info.size = len(data)
                info.mode = 0o755 if os.access(src_path, os.X_OK) else 0o644
                tar.addfile(info, io.BytesIO(data))

    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    tar_bytes = tar_buffer.getvalue()
    with open(tarball_path, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            gz.write(tar_bytes)

    return tarball_path


# ---------------------------------------------------------------------------
# Catalog operations
# ---------------------------------------------------------------------------

def build_catalog(
    analysis_reports: list[AnalysisReport],
    signers: dict[str, str],
    base_version: int = 0,
) -> Catalog:
    """Assemble a Catalog from analysis reports.

    Args:
        analysis_reports: List of completed analysis reports.
        signers: Dict mapping signer_id -> signer description/name.
        base_version: Previous catalog version (new version = base + 1).

    Returns:
        A Catalog ready to be signed.
    """
    entries = []
    for report in analysis_reports:
        entry = CatalogEntry(
            skill_name=report.skill_name,
            source_url=report.source_url,
            pinned_commit=report.pinned_commit,
            content_sha256=report.content_sha256,
            upstream_author="",  # filled from frontmatter if available
            signer=list(signers.keys())[0] if signers else "",
            catalog_version=base_version + 1,
            analyzed_at=report.analyzed_at,
            static_verdict=report.static.verdict if report.static else AnalysisVerdict.ERROR,
            summarization_verdict=report.summarization.verdict if report.summarization else AnalysisVerdict.ERROR,
            crossref_verdict=report.crossref.verdict if report.crossref else AnalysisVerdict.ERROR,
            sandbox_verdict=report.sandbox.verdict if report.sandbox else AnalysisVerdict.ERROR,
        )
        entries.append(entry)

    return Catalog(
        version=base_version + 1,
        entries=entries,
        signers=signers,
    )


def sign_catalog(
    catalog: Catalog,
    output_dir: Path,
    secret_key: Path,
) -> Path:
    """Write catalog.json and sign it.

    Trusted comment: catalog-version:{catalog.version}
    Returns path to catalog.json.
    """
    catalog_path = output_dir / "catalog.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(catalog.model_dump_json())
    _write_json(data, catalog_path)

    sign_file(catalog_path, secret_key, f"catalog-version:{catalog.version}")

    return catalog_path


def verify_catalog(
    catalog_path: Path,
    pubkey_path: Path,
) -> Catalog:
    """Verify catalog signature, load and return catalog.

    Raises RuntimeError if verification fails.
    """
    verify_file(catalog_path, pubkey_path)
    data = _load_json(catalog_path)
    return Catalog.model_validate(data)


def sign_skill(
    skill_tarball: Path,
    secret_key: Path,
    skill_name: str,
    commit_sha: str,
) -> Path:
    """Sign a skill tarball with a reviewer key.

    Trusted comment: skill:{skill_name}:{commit_sha[:12]}
    Returns path to .minisig file.
    """
    trusted = f"skill:{skill_name}:{commit_sha[:12]}"
    return sign_file(skill_tarball, secret_key, trusted)


def verify_skill(
    skill_tarball: Path,
    pubkey_path: Path,
    expected_sha256: str,
) -> None:
    """Verify skill tarball signature and content hash.

    Raises RuntimeError if either check fails.
    """
    verify_file(skill_tarball, pubkey_path)

    actual_hash = sha256_file(skill_tarball)
    if actual_hash != expected_sha256:
        raise RuntimeError(
            f"Content hash mismatch for {skill_tarball}:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual_hash}"
        )


# ---------------------------------------------------------------------------
# Signer management
# ---------------------------------------------------------------------------

def add_signer(
    catalog: Catalog,
    signer_id: str,
    pubkey_path: Path,
) -> None:
    """Register a reviewer key in the catalog."""
    key_id_hex, b64_key = parse_minisign_pubkey(pubkey_path)
    catalog.signers[signer_id] = b64_key
    logger.info("Registered signer %s (key ID: %s)", signer_id, key_id_hex)


# ---------------------------------------------------------------------------
# Revocations
# ---------------------------------------------------------------------------

def load_revocations(output_dir: Path) -> dict:
    """Load revocations.json or return empty skeleton."""
    rev_path = output_dir / "revocations.json"
    if rev_path.exists():
        return _load_json(rev_path)
    return {"revoked": []}


def revoke_signer(
    output_dir: Path,
    key_id: str,
    reason: str,
    secret_key: Path,
) -> None:
    """Add key to revocations.json, re-sign."""
    revocations = load_revocations(output_dir)

    for entry in revocations["revoked"]:
        if entry["key_id"] == key_id:
            raise ValueError(f"Key '{key_id}' is already revoked")

    revocations["revoked"].append({
        "key_id": key_id,
        "revoked_at": _now_utc(),
        "reason": reason,
    })

    rev_path = output_dir / "revocations.json"
    _write_json(revocations, rev_path)
    sign_file(rev_path, secret_key, f"revocations:{_now_utc()}")


def check_revocations(
    catalog: Catalog,
    revocations: dict,
) -> list[str]:
    """Check catalog entries against revocation list.

    Returns list of warning strings for entries signed by revoked keys.
    """
    revoked_ids = {e["key_id"] for e in revocations.get("revoked", [])}
    warnings = []

    for entry in catalog.entries:
        if entry.signer in revoked_ids:
            warnings.append(
                f"Skill '{entry.skill_name}' is signed by revoked key '{entry.signer}'"
            )

    return warnings


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

def verify_all(
    output_dir: Path,
    pubkey_path: Path,
) -> None:
    """Full verification chain.

    1. Verify catalog signature
    2. Load catalog
    3. Check revocations
    4. Verify each skill tarball signature + hash

    Raises RuntimeError on first failure.
    """
    catalog_path = output_dir / "catalog.json"
    catalog = verify_catalog(catalog_path, pubkey_path)

    revocations = load_revocations(output_dir)
    warnings = check_revocations(catalog, revocations)
    if warnings:
        raise RuntimeError(
            "Revoked signer detected:\n" + "\n".join(f"  - {w}" for w in warnings)
        )

    for entry in catalog.entries:
        tarball_name = f"{entry.skill_name}.tar.gz"
        tarball_path = output_dir / "skills" / tarball_name
        if tarball_path.exists():
            verify_skill(tarball_path, pubkey_path, entry.content_sha256)
            logger.info("Verified: %s", entry.skill_name)
        else:
            logger.warning("Tarball not found for %s (expected %s)", entry.skill_name, tarball_path)
