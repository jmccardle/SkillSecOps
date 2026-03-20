"""Tests for catalog signing operations."""

from __future__ import annotations

import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillsecops.catalog import (
    build_catalog,
    build_tarball,
    parse_minisign_pubkey,
    revoke_signer,
    sha256_file,
    sign_catalog,
    sign_file,
    sign_skill,
    verify_catalog,
    verify_file,
    verify_skill,
    load_revocations,
    check_revocations,
)
from skillsecops.models import (
    AnalysisReport,
    AnalysisVerdict,
    Catalog,
    CatalogEntry,
    StaticAnalysisResult,
)


needs_minisign = pytest.mark.skipif(
    not shutil.which("minisign"),
    reason="minisign not available",
)


# ---------------------------------------------------------------------------
# Deterministic tarball
# ---------------------------------------------------------------------------

class TestDeterministicTarball:
    def test_tarball_reproducible(self, tmp_path):
        """Same input produces identical tarballs."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test\n\nContent.")
        (skill_dir / "README.md").write_text("Readme.")

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        t1 = build_tarball(skill_dir, out1)
        t2 = build_tarball(skill_dir, out2)

        assert sha256_file(t1) == sha256_file(t2)

    def test_tarball_mtime_zero(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")

        tarball = build_tarball(skill_dir, tmp_path / "out")
        with tarfile.open(tarball, "r:gz") as tar:
            for member in tar.getmembers():
                assert member.mtime == 0

    def test_tarball_sorted_entries(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        # Create files in reverse alphabetical order
        (skill_dir / "z_file.txt").write_text("z")
        (skill_dir / "a_file.txt").write_text("a")
        (skill_dir / "m_file.txt").write_text("m")

        tarball = build_tarball(skill_dir, tmp_path / "out")
        with tarfile.open(tarball, "r:gz") as tar:
            names = [m.name for m in tar.getmembers() if not m.isdir()]
            # Should be sorted
            basenames = [n.split("/")[-1] for n in names]
            assert basenames == sorted(basenames)

    def test_tarball_uid_gid_zero(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")

        tarball = build_tarball(skill_dir, tmp_path / "out")
        with tarfile.open(tarball, "r:gz") as tar:
            for member in tar.getmembers():
                assert member.uid == 0
                assert member.gid == 0
                assert member.uname == ""
                assert member.gname == ""


# ---------------------------------------------------------------------------
# Minisign sign/verify
# ---------------------------------------------------------------------------

@needs_minisign
class TestSignVerify:
    def test_sign_verify_roundtrip(self, minisign_keypair, tmp_path):
        secret_key, pubkey = minisign_keypair
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        sig_path = sign_file(test_file, secret_key, "test-comment")
        assert sig_path.exists()
        assert sig_path.suffix == ".minisig"

        # Should not raise
        verify_file(test_file, pubkey)

    def test_verify_tampered_fails(self, minisign_keypair, tmp_path):
        secret_key, pubkey = minisign_keypair
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        sign_file(test_file, secret_key, "test")

        # Tamper with the file
        test_file.write_text("Tampered content")

        with pytest.raises(RuntimeError, match="verification failed"):
            verify_file(test_file, pubkey)

    def test_missing_signature_fails(self, minisign_keypair, tmp_path):
        _, pubkey = minisign_keypair
        test_file = tmp_path / "unsigned.txt"
        test_file.write_text("No signature")

        with pytest.raises(RuntimeError, match="Signature file not found"):
            verify_file(test_file, pubkey)


@needs_minisign
class TestPubkeyParsing:
    def test_parse_minisign_pubkey(self, minisign_keypair):
        _, pubkey = minisign_keypair
        key_id, b64 = parse_minisign_pubkey(pubkey)

        assert len(key_id) == 16  # 8 bytes = 16 hex chars
        assert len(b64) > 20


# ---------------------------------------------------------------------------
# Catalog assembly and signing
# ---------------------------------------------------------------------------

class TestCatalogAssembly:
    def test_build_catalog_from_reports(self):
        report = AnalysisReport(
            skill_name="csv-tool",
            source_url="https://github.com/owner/repo/tree/main/skills/csv",
            pinned_commit="abc123def456",
            content_sha256="deadbeef" * 8,
            static=StaticAnalysisResult(
                verdict=AnalysisVerdict.PASS, patterns_checked=11,
            ),
        )
        catalog = build_catalog(
            [report],
            signers={"reviewer-2026": "Test Reviewer"},
            base_version=0,
        )

        assert catalog.version == 1
        assert len(catalog.entries) == 1
        assert catalog.entries[0].skill_name == "csv-tool"
        assert catalog.entries[0].pinned_commit == "abc123def456"

    def test_catalog_version_monotonic(self):
        report = AnalysisReport(
            skill_name="test",
            source_url="https://github.com/test",
            pinned_commit="abc123",
            content_sha256="aabb" * 16,
        )
        cat1 = build_catalog([report], {"s": "Signer"}, base_version=0)
        cat2 = build_catalog([report], {"s": "Signer"}, base_version=cat1.version)

        assert cat2.version == 2
        assert cat2.version > cat1.version


@needs_minisign
class TestCatalogSigning:
    def test_sign_and_verify_catalog(self, minisign_keypair, tmp_path):
        secret_key, pubkey = minisign_keypair

        catalog = Catalog(version=1, signers={"test": "test-key"})
        catalog_path = sign_catalog(catalog, tmp_path, secret_key)

        loaded = verify_catalog(catalog_path, pubkey)
        assert loaded.version == 1


# ---------------------------------------------------------------------------
# Skill signing
# ---------------------------------------------------------------------------

@needs_minisign
class TestSkillSigning:
    def test_sign_and_verify_skill(self, minisign_keypair, tmp_path):
        secret_key, pubkey = minisign_keypair

        # Create a test skill tarball
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")
        tarball = build_tarball(skill_dir, tmp_path / "out")

        expected_hash = sha256_file(tarball)
        sign_skill(tarball, secret_key, "test-skill", "abc123def456")

        # Should not raise
        verify_skill(tarball, pubkey, expected_hash)

    def test_verify_wrong_hash_fails(self, minisign_keypair, tmp_path):
        secret_key, pubkey = minisign_keypair

        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")
        tarball = build_tarball(skill_dir, tmp_path / "out")

        sign_skill(tarball, secret_key, "test-skill", "abc123")

        with pytest.raises(RuntimeError, match="Content hash mismatch"):
            verify_skill(tarball, pubkey, "wrong_hash")


# ---------------------------------------------------------------------------
# Revocations
# ---------------------------------------------------------------------------

@needs_minisign
class TestRevocations:
    def test_revoke_signer(self, minisign_keypair, tmp_path):
        secret_key, _ = minisign_keypair

        revoke_signer(tmp_path, "BADKEY123", "Compromised", secret_key)

        revocations = load_revocations(tmp_path)
        assert len(revocations["revoked"]) == 1
        assert revocations["revoked"][0]["key_id"] == "BADKEY123"
        assert revocations["revoked"][0]["reason"] == "Compromised"

    def test_double_revoke_raises(self, minisign_keypair, tmp_path):
        secret_key, _ = minisign_keypair

        revoke_signer(tmp_path, "BADKEY123", "Compromised", secret_key)
        with pytest.raises(ValueError, match="already revoked"):
            revoke_signer(tmp_path, "BADKEY123", "Again", secret_key)

    def test_check_revocations_warns(self):
        catalog = Catalog(
            version=1,
            entries=[
                CatalogEntry(
                    skill_name="evil-skill",
                    source_url="https://github.com/test",
                    pinned_commit="abc",
                    content_sha256="dead" * 16,
                    upstream_author="attacker",
                    signer="REVOKED_KEY",
                    catalog_version=1,
                    analyzed_at=datetime.now(timezone.utc),
                ),
            ],
        )
        revocations = {"revoked": [{"key_id": "REVOKED_KEY", "revoked_at": "2026-01-01", "reason": "bad"}]}

        warnings = check_revocations(catalog, revocations)
        assert len(warnings) == 1
        assert "evil-skill" in warnings[0]

    def test_check_revocations_clean(self):
        catalog = Catalog(
            version=1,
            entries=[
                CatalogEntry(
                    skill_name="good-skill",
                    source_url="https://github.com/test",
                    pinned_commit="abc",
                    content_sha256="beef" * 16,
                    upstream_author="author",
                    signer="GOOD_KEY",
                    catalog_version=1,
                    analyzed_at=datetime.now(timezone.utc),
                ),
            ],
        )
        warnings = check_revocations(catalog, {"revoked": []})
        assert warnings == []
