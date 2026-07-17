"""Tests for the narrow ``hermes gsd`` CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from hermes_cli import gsd_cmd


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeCodexGsdRunner:
    instances: ClassVar[list["_FakeCodexGsdRunner"]] = []
    real_cls: ClassVar[Any] = None

    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self._real = self.real_cls(self.workspace)
        self.calls = []
        self.instances.append(self)

    def run_gsd_command(self, command):
        argv = self._real.build_gsd_command(command)
        self.calls.append(("run", command, argv))
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"test-thread"}\n'
            '{"type":"turn.completed"}\n',
            stderr="",
            command=argv,
        )

    def resume(self, session_id, answer):
        argv = self._real.build_resume_command(session_id, answer)
        self.calls.append(("resume", session_id, answer, argv))
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"test-thread"}\n'
            '{"type":"turn.completed"}\n',
            stderr="",
            command=argv,
        )


def _run(argv, monkeypatch):
    import agent.transports.codex_app_server as codex_mod

    _FakeCodexGsdRunner.instances = []
    _FakeCodexGsdRunner.real_cls = codex_mod.CodexGsdRunner
    monkeypatch.setattr(codex_mod, "CodexGsdRunner", _FakeCodexGsdRunner)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    p = gsd_cmd.build_parser(sub)
    p.set_defaults(func=gsd_cmd.gsd_command)
    args = parser.parse_args(["gsd", *argv])
    rc = gsd_cmd.gsd_command(args)
    return rc, _FakeCodexGsdRunner.instances


def test_gsd_help_with_workspace_reaches_runner_and_prints_jsonl(monkeypatch, tmp_path, capsys):
    rc, instances = _run(["help", "--workspace", str(tmp_path)], monkeypatch)

    assert rc == 0
    assert len(instances) == 1
    runner = instances[0]
    assert runner.workspace == tmp_path.resolve()
    kind, command, argv = runner.calls[0]
    assert kind == "run"
    assert command == "help"
    assert argv == [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "-C",
        str(tmp_path.resolve()),
        "$gsd-help",
    ]
    assert '{"type":"thread.started"' in capsys.readouterr().out
    assert not (REPO_ROOT / ".planning").exists()


def test_gsd_defaults_workspace_to_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    rc, instances = _run(["help"], monkeypatch)

    assert rc == 0
    assert instances[0].workspace == tmp_path.resolve()
    assert instances[0].calls[0][2][6] == str(tmp_path.resolve())
    assert not (REPO_ROOT / ".planning").exists()


def test_gsd_resume_with_workspace_reaches_runner_resume(monkeypatch, tmp_path, capsys):
    rc, instances = _run(
        ["resume", "session-123", "dummy answer", "--workspace", str(tmp_path)],
        monkeypatch,
    )

    assert rc == 0
    assert len(instances) == 1
    runner = instances[0]
    kind, session_id, answer, argv = runner.calls[0]
    assert kind == "resume"
    assert session_id == "session-123"
    assert answer == "dummy answer"
    assert argv == [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "-C",
        str(tmp_path.resolve()),
        "resume",
        "session-123",
        "dummy answer",
    ]
    assert '{"type":"turn.completed"}' in capsys.readouterr().out
    assert not (REPO_ROOT / ".planning").exists()


def test_gsd_command_argv_contains_required_exec_flags(monkeypatch, tmp_path):
    rc, instances = _run(["plan-phase", "--workspace", str(tmp_path)], monkeypatch)

    assert rc == 0
    argv = instances[0].calls[0][2]
    assert "--json" in argv
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert argv[argv.index("-C") + 1] == str(tmp_path.resolve())
    assert argv[-1] == "$gsd-plan-phase"
    assert not (REPO_ROOT / ".planning").exists()
