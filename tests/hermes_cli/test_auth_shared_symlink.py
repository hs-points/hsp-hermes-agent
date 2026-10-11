"""HSP: shared auth.json symlink safety (ported from the live install's shared-auth fix).

Several HSP profiles symlink ``auth.json`` to one group-shared target (0660). Saving must
rewrite that target in place (same inode, owner, group and mode) and every profile must lock
the canonical target, so single-use OAuth refresh tokens are never consumed concurrently.
Tests 1-7 are copied verbatim from the live install; the last one guards upstream's global
auth cache invalidation on explicit write-throughs. POSIX-only.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared-inode semantics")


def test_save_auth_store_preserves_shared_symlink_target_inode(tmp_path, monkeypatch):
    """Refreshing a shared auth link must not replace the managed target."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    shared_auth.write_text(
        json.dumps({"version": 1, "providers": {}}) + "\n",
        encoding="utf-8",
    )
    shared_auth.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    before = shared_auth.stat()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    auth_mod._save_auth_store(
        {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {
                "openai-codex": {"tokens": {"access_token": "refreshed-secret"}}
            },
        }
    )

    after = shared_auth.stat()
    assert auth_link.is_symlink()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    data = json.loads(shared_auth.read_text(encoding="utf-8"))
    assert data["providers"]["openai-codex"]["tokens"]["access_token"] == "refreshed-secret"


def test_auth_store_lock_for_shared_symlink_uses_canonical_target_lock(tmp_path, monkeypatch):
    """Profiles sharing auth.json must serialize refreshes through one lock."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    shared_auth.write_text(
        json.dumps({"version": 1, "providers": {}}) + "\n",
        encoding="utf-8",
    )
    shared_auth.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    with auth_mod._auth_store_lock():
        assert shared_auth.with_suffix(".lock").exists()
        assert not auth_link.with_suffix(".lock").exists()

    assert auth_link.is_symlink()


def test_save_auth_store_target_path_shared_symlink_uses_target_lock_and_preserves_metadata(
    tmp_path,
    monkeypatch,
):
    """Explicit target_path writes need the same shared-symlink safety as active auth.json."""
    profile_home = tmp_path / "profile"
    other_profile_home = tmp_path / "other-profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    other_profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    shared_auth.write_text(
        json.dumps({"version": 1, "providers": {}}) + "\n",
        encoding="utf-8",
    )
    shared_auth.chmod(0o660)
    target_link = other_profile_home / "auth.json"
    target_link.symlink_to(shared_auth)
    before = shared_auth.stat()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    with auth_mod._auth_store_lock(target_path=target_link):
        assert shared_auth.with_suffix(".lock").exists()
        assert not target_link.with_suffix(".lock").exists()

    auth_mod._save_auth_store(
        {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {"xai": {"tokens": {"access_token": "target-secret"}}},
            "active_provider": "xai",
        },
        target_path=target_link,
    )

    after = shared_auth.stat()
    assert target_link.is_symlink()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    data = json.loads(shared_auth.read_text(encoding="utf-8"))
    assert data["providers"]["xai"]["tokens"]["access_token"] == "target-secret"


def test_shared_auth_inner_target_flock_times_out_without_corrupting_target(
    tmp_path,
    monkeypatch,
):
    """A peer holding the target inode cannot hang this writer indefinitely."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    original_payload = json.dumps({"version": 1, "providers": {}}) + "\n"
    shared_auth.write_text(original_payload, encoding="utf-8")
    shared_auth.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    real_flock = auth_mod.fcntl.flock

    def blocking_target_flock(fd, operation):
        if operation & auth_mod.fcntl.LOCK_EX:
            try:
                fd_path = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                fd_path = ""
            if fd_path == str(shared_auth):
                raise BlockingIOError("target locked by peer")
        return real_flock(fd, operation)

    monkeypatch.setattr(auth_mod.fcntl, "flock", blocking_target_flock)
    monkeypatch.setattr(auth_mod, "AUTH_LOCK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(auth_mod.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="shared auth target lock"):
        auth_mod._save_auth_store(
            {
                "version": auth_mod.AUTH_STORE_VERSION,
                "providers": {"xai": {"tokens": {"access_token": "new-secret"}}},
                "active_provider": "xai",
            }
        )

    after = shared_auth.stat()
    assert auth_link.is_symlink()
    assert stat.S_IMODE(after.st_mode) == 0o660
    assert shared_auth.read_text(encoding="utf-8") == original_payload


def test_shared_auth_target_swap_between_detection_and_open_fails_closed(
    tmp_path,
    monkeypatch,
):
    """A swapped shared target path must not redirect writes to another file."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    original_payload = json.dumps({"version": 1, "providers": {}}) + "\n"
    shared_auth.write_text(original_payload, encoding="utf-8")
    shared_auth.chmod(0o660)
    attacker_target = shared_dir / "attacker-auth.json"
    attacker_payload = json.dumps(
        {"version": 1, "providers": {"openai": {"tokens": {"access_token": "attacker"}}}},
        sort_keys=True,
    ) + "\n"
    attacker_target.write_text(attacker_payload, encoding="utf-8")
    attacker_target.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    real_open = auth_mod.os.open
    swapped = {"done": False}

    def swap_target_before_open(path, flags, mode=0o777, *args, **kwargs):
        if not swapped["done"] and os.fspath(path) == str(shared_auth):
            swapped["done"] = True
            shared_auth.unlink()
            shared_auth.symlink_to(attacker_target)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(auth_mod.os, "open", swap_target_before_open)

    with pytest.raises(OSError, match="shared auth target"):
        auth_mod._save_auth_store(
            {
                "version": auth_mod.AUTH_STORE_VERSION,
                "providers": {"xai": {"tokens": {"access_token": "new-secret"}}},
                "active_provider": "xai",
            }
        )

    assert swapped["done"]
    assert auth_link.is_symlink()
    assert shared_auth.is_symlink()
    assert shared_auth.resolve(strict=True) == attacker_target
    assert attacker_target.read_text(encoding="utf-8") == attacker_payload


def test_shared_auth_target_rebind_after_open_fails_closed(tmp_path, monkeypatch):
    """A target path swapped after open must not produce a successful save."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    original_payload = json.dumps({"version": 1, "providers": {}}) + "\n"
    shared_auth.write_text(original_payload, encoding="utf-8")
    shared_auth.chmod(0o660)
    attacker_target = shared_dir / "attacker-auth.json"
    attacker_payload = json.dumps(
        {"version": 1, "providers": {"openai": {"tokens": {"access_token": "attacker"}}}},
        sort_keys=True,
    ) + "\n"
    attacker_target.write_text(attacker_payload, encoding="utf-8")
    attacker_target.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    real_open = auth_mod.os.open
    swapped = {"done": False}

    def swap_target_after_open(path, flags, mode=0o777, *args, **kwargs):
        fd = real_open(path, flags, mode, *args, **kwargs)
        if not swapped["done"] and os.fspath(path) == str(shared_auth):
            swapped["done"] = True
            shared_auth.unlink()
            shared_auth.symlink_to(attacker_target)
        return fd

    monkeypatch.setattr(auth_mod.os, "open", swap_target_after_open)

    with pytest.raises(OSError, match="shared auth target"):
        auth_mod._save_auth_store(
            {
                "version": auth_mod.AUTH_STORE_VERSION,
                "providers": {"xai": {"tokens": {"access_token": "new-secret"}}},
                "active_provider": "xai",
            }
        )

    assert swapped["done"]
    assert auth_link.is_symlink()
    assert shared_auth.is_symlink()
    assert shared_auth.resolve(strict=True) == attacker_target
    assert attacker_target.read_text(encoding="utf-8") == attacker_payload


def test_shared_auth_partial_write_failure_restores_original_payload_and_metadata(
    tmp_path,
    monkeypatch,
):
    """Recoverable write failures restore the prior shared auth payload."""
    profile_home = tmp_path / "profile"
    shared_dir = tmp_path / "shared"
    profile_home.mkdir()
    shared_dir.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    original_payload = json.dumps(
        {
            "version": 1,
            "providers": {"openai": {"tokens": {"access_token": "old-secret"}}},
            "active_provider": "openai",
        },
        sort_keys=True,
    ) + "\n"
    shared_auth.write_text(original_payload, encoding="utf-8")
    shared_auth.chmod(0o660)
    auth_link = profile_home / "auth.json"
    auth_link.symlink_to(shared_auth)
    before = shared_auth.stat()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    real_write_all = auth_mod._write_all
    calls = {"count": 0}

    def fail_after_partial_payload_write(fd, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            os.write(fd, b'{"partial":')
            raise OSError("simulated partial write failure")
        return real_write_all(fd, payload)

    monkeypatch.setattr(auth_mod, "_write_all", fail_after_partial_payload_write)

    with pytest.raises(OSError, match="simulated partial write failure"):
        auth_mod._save_auth_store(
            {
                "version": auth_mod.AUTH_STORE_VERSION,
                "providers": {"xai": {"tokens": {"access_token": "new-secret"}}},
                "active_provider": "xai",
            }
        )

    after = shared_auth.stat()
    assert auth_link.is_symlink()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert shared_auth.read_text(encoding="utf-8") == original_payload


def test_shared_target_path_write_still_invalidates_global_auth_cache(tmp_path, monkeypatch):
    """The shared in-place branch must not skip upstream's explicit write-through cache reset."""
    profile_home = tmp_path / "profile"
    other_profile_home = tmp_path / "other-profile"
    shared_dir = tmp_path / "shared"
    for d in (profile_home, other_profile_home, shared_dir):
        d.mkdir()
    shared_auth = shared_dir / "hermes-auth.json"
    shared_auth.write_text(json.dumps({"version": 1, "providers": {}}) + "\n", encoding="utf-8")
    shared_auth.chmod(0o660)
    target_link = other_profile_home / "auth.json"
    target_link.symlink_to(shared_auth)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from hermes_cli import auth as auth_mod

    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", object())
    auth_mod._save_auth_store(
        {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}, target_path=target_link)
    assert auth_mod._global_auth_store_cache is None
