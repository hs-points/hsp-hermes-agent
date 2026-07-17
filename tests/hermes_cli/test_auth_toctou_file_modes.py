"""Regression tests for TOCTOU-safe credential file writers in ``hermes_cli.auth``.

Background
==========
The three writers below used to create a temp file via ``Path.write_text`` /
``Path.open('w')`` and only ``chmod``'d it to ``0o600`` afterward. Between
create and chmod the file existed at the process umask (typically ``0o644``),
briefly exposing OAuth tokens to other local users on multi-user hosts. The
fix switches them to ``os.open(O_EXCL, mode=0o600)`` + ``os.fdopen`` +
``fsync`` so the file is atomic at ``0o600`` on creation. Mirrors the fixes
shipped for ``agent/google_oauth.py`` (#19673) and ``tools/mcp_oauth.py``
(#21148).

These tests stay green only while the token file and its parent directory
end up at ``0o600`` / ``0o700`` after every write. POSIX-only — the mode-bit
enforcement does not exist on Windows.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import subprocess
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)


# ---------------------------------------------------------------------------
# _save_auth_store  (~/.hermes/auth.json — every native OAuth provider)
# ---------------------------------------------------------------------------


def test_save_auth_store_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """``_save_auth_store`` must land ``auth.json`` at 0o600 and parent at 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old_umask = os.umask(0o022)  # make the race observable if it regresses
    try:
        from hermes_cli import auth as auth_mod

        auth_store = {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {"openai-codex": {"tokens": {"access_token": "secret-x"}}},
            "active_provider": "openai-codex",
        }
        auth_path = auth_mod._save_auth_store(auth_store)
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(auth_path.stat().st_mode)
    parent_mode = stat.S_IMODE(auth_path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"auth.json mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"auth.json parent dir mode 0o{parent_mode:o} != 0o700 — siblings can traverse"
    )

    # Content survived the rewrite
    data = json.loads(auth_path.read_text())
    assert data["providers"]["openai-codex"]["tokens"]["access_token"] == "secret-x"


# ---------------------------------------------------------------------------
# _save_qwen_cli_tokens  (Qwen CLI OAuth tokens)
# ---------------------------------------------------------------------------


def test_save_qwen_cli_tokens_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """``_save_qwen_cli_tokens`` must land the token file at 0o600 and parent at 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The Qwen CLI auth path lives under $HOME/.qwen by default — isolate it.
    monkeypatch.setenv("HOME", str(tmp_path))
    old_umask = os.umask(0o022)
    try:
        from hermes_cli import auth as auth_mod

        tokens = {
            "access_token": "qwen-secret",
            "refresh_token": "qwen-refresh",
            "token_type": "Bearer",
            "expiry_date": 123,
        }
        auth_path = auth_mod._save_qwen_cli_tokens(tokens)
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(auth_path.stat().st_mode)
    parent_mode = stat.S_IMODE(auth_path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"Qwen token file mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"Qwen token parent dir mode 0o{parent_mode:o} != 0o700"
    )

    data = json.loads(auth_path.read_text())
    assert data["access_token"] == "qwen-secret"


# ---------------------------------------------------------------------------
# Nous shared-credential store write (inside _write_shared_nous_state)
# ---------------------------------------------------------------------------


def test_shared_nous_store_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """The Nous shared-credential store must land at 0o600 / parent 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # _nous_shared_store_path() refuses to touch the real shared store during
    # pytest runs; redirect it into tmp_path explicitly. Use a distinct
    # subdirectory name (``shared_override``) so the guard's "real user
    # home" reference — which currently tracks HERMES_HOME via
    # get_default_hermes_root() — can't collide with our override and
    # falsely claim we're writing to the real user's shared store.
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared_override"))
    old_umask = os.umask(0o022)
    try:
        from hermes_cli import auth as auth_mod

        state = {
            "access_token": "nous-access-xxx",
            "refresh_token": "nous-refresh-xxx",
            "token_type": "Bearer",
            "scope": "openid profile",
            "client_id": "test-client",
            "obtained_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-01-01T01:00:00Z",
        }
        auth_mod._write_shared_nous_state(state)
        path = auth_mod._nous_shared_store_path()
    finally:
        os.umask(old_umask)

    assert path.exists(), "shared Nous store was not written"
    mode = stat.S_IMODE(path.stat().st_mode)
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"Nous shared store mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"Nous shared store parent dir mode 0o{parent_mode:o} != 0o700"
    )

    data = json.loads(path.read_text())
    assert data["refresh_token"] == "nous-refresh-xxx"


# ---------------------------------------------------------------------------
# Atomicity: verify ``os.open`` is called with an explicit 0o600 mode.
# ---------------------------------------------------------------------------


def test_save_auth_store_uses_os_open_with_0o600_mode(tmp_path, monkeypatch):
    """Regression: the writer must call ``os.open`` with an explicit restricted
    mode so the file is created at 0o600 atomically — closing the TOCTOU
    window the previous ``Path.open('w')`` left open (fd inherited process
    umask and was briefly 0o644 before post-write chmod)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    observed_opens: list[tuple[str, int, int]] = []
    real_os_open = os.open

    def spying_os_open(path, flags, mode=0o777, *args, **kwargs):
        observed_opens.append((str(path), flags, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    with patch.object(os, "open", spying_os_open):
        from hermes_cli import auth as auth_mod

        auth_mod._save_auth_store(
            {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
        )

    auth_tmp_opens = [
        (p, fl, m) for (p, fl, m) in observed_opens if "auth.json.tmp" in p
    ]
    assert auth_tmp_opens, (
        f"os.open was never called for the auth.json temp file; "
        f"observed={observed_opens!r}"
    )
    for path, flags, mode in auth_tmp_opens:
        assert flags & os.O_CREAT, f"auth.json temp open missing O_CREAT: path={path}"
        assert flags & os.O_EXCL, (
            f"auth.json temp open missing O_EXCL — TOCTOU-safe pattern regressed: "
            f"path={path}, flags={flags}"
        )
        # Must be exactly S_IRUSR | S_IWUSR (0o600) — no group/other bits.
        expected = stat.S_IRUSR | stat.S_IWUSR
        assert mode == expected, (
            f"auth.json temp open mode 0o{mode:o} != 0o{expected:o} — "
            f"umask would apply and potentially expose tokens"
        )

# ---------------------------------------------------------------------------
# Shared symlink auth stores: cross-user lock + metadata-preserving writes.
# ---------------------------------------------------------------------------


def _make_shared_auth_store(tmp_path, monkeypatch):
    home = tmp_path / "consumer-home"
    home.mkdir()
    shared_dir = tmp_path / "shared-auth"
    shared_dir.mkdir()
    shared_dir.chmod(0o2770)
    target = shared_dir / "hermes-auth.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {"openai-codex": []},
            }
        )
        + "\n"
    )
    target.chmod(0o660)
    auth_link = home / "auth.json"
    auth_link.symlink_to(target)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, auth_link, target


def _metadata(path):
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
    )


def test_shared_auth_save_preserves_symlink_inode_owner_group_and_mode(
    tmp_path, monkeypatch
):
    _, auth_link, target = _make_shared_auth_store(tmp_path, monkeypatch)
    before = _metadata(target)

    from hermes_cli import auth as auth_mod

    saved = auth_mod._save_auth_store(
        {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {},
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "shared",
                        "auth_type": "oauth",
                        "access_token": "rotated-secret",
                    }
                ]
            },
        }
    )

    assert saved == auth_link
    assert auth_link.is_symlink()
    assert auth_link.resolve() == target
    assert _metadata(target) == before
    assert json.loads(target.read_text())["credential_pool"]["openai-codex"][0][
        "access_token"
    ] == "rotated-secret"

    journal = auth_mod._shared_auth_journal_path(target)
    assert journal.exists()
    assert journal.stat().st_gid == target.stat().st_gid
    assert stat.S_IMODE(journal.stat().st_mode) == 0o660
    assert auth_mod._auth_lock_path(auth_link) == target


def test_shared_auth_lock_blocks_a_second_process(tmp_path, monkeypatch):
    home, auth_link, _ = _make_shared_auth_store(tmp_path, monkeypatch)

    from hermes_cli import auth as auth_mod

    script = """
from hermes_cli import auth
try:
    with auth._auth_store_lock(timeout_seconds=0.2):
        print("acquired")
except TimeoutError:
    print("blocked")
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    with auth_mod._auth_store_lock(auth_file=auth_link):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

    assert result.stdout.strip() == "blocked"


def test_shared_auth_load_recovers_interrupted_write_without_metadata_change(
    tmp_path, monkeypatch
):
    _, auth_link, target = _make_shared_auth_store(tmp_path, monkeypatch)

    from hermes_cli import auth as auth_mod

    expected = {
        "version": auth_mod.AUTH_STORE_VERSION,
        "providers": {"openai-codex": {"tokens": {"access_token": "last-good"}}},
        "active_provider": "openai-codex",
    }
    payload = (json.dumps(expected, indent=2) + "\n").encode()
    auth_mod._write_shared_auth_journal(target, payload)
    before = _metadata(target)

    # Simulate a process dying after truncation while holding the shared lock.
    target.write_text('{"version": 1, "providers":')
    target.chmod(before[-1])

    loaded = auth_mod._load_auth_store(auth_link)

    assert loaded["providers"]["openai-codex"]["tokens"]["access_token"] == "last-good"
    assert json.loads(target.read_text()) == expected
    assert _metadata(target) == before



def test_shared_auth_save_uses_group_journal_when_target_directory_is_read_only(
    tmp_path, monkeypatch
):
    _, auth_link, target = _make_shared_auth_store(tmp_path, monkeypatch)

    from hermes_cli import auth as auth_mod

    fallback_root = tmp_path / "var-tmp"
    fallback_root.mkdir()
    monkeypatch.setattr(auth_mod, "_SHARED_AUTH_JOURNAL_ROOT", fallback_root)
    target.parent.chmod(0o2550)
    before = _metadata(target)

    auth_mod._save_auth_store(
        {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {},
            "credential_pool": {"openai-codex": []},
        }
    )

    journal = auth_mod._shared_auth_journal_path(target)
    assert journal.parent.parent == fallback_root
    assert stat.S_IMODE(journal.parent.stat().st_mode) == 0o2770
    assert journal.parent.stat().st_gid == target.stat().st_gid
    assert json.loads(journal.read_text())["credential_pool"]["openai-codex"] == []
    assert _metadata(target) == before
