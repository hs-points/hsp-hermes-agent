"""``hermes gsd`` CLI — drive GSD Core through Codex from a workspace.

This is intentionally a narrow CLI surface over ``CodexGsdRunner``. It does
not add a model tool, config key, slash command, or runner registry.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RAW_JSONL_LOCATION_HINT = (
    "codex emits JSONL on stdout; redirect this command if you want to save the raw stream"
)


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``gsd`` subcommand. Returns the top parser."""
    parser = parent_subparsers.add_parser(
        "gsd",
        help="Run GSD Core skills through Codex in a target workspace",
        description=(
            "Run a GSD Core command through Codex from a target application "
            "workspace. Defaults --workspace to the current working directory. "
            "Examples:\n"
            "  hermes gsd help\n"
            "  hermes gsd new-project --workspace /path/to/app\n"
            "  hermes gsd resume <SESSION_ID> \"<answer>\" --workspace /path/to/app"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gsd_args",
        nargs="+",
        metavar="COMMAND|resume",
        help=(
            "GSD command name (help, new-project, gsd-plan-phase, $gsd-help) "
            "or: resume <SESSION_ID> <answer>"
        ),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        metavar="PATH",
        help="Target application workspace path (default: current directory)",
    )
    parser.set_defaults(_gsd_parser=parser)
    return parser


def gsd_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes gsd …`` argparse dispatch."""
    gsd_args = list(getattr(args, "gsd_args", []) or [])
    if not gsd_args:
        parser = getattr(args, "_gsd_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print("usage: hermes gsd <command> [--workspace PATH]", file=sys.stderr)
        return 2

    workspace = Path(getattr(args, "workspace", None) or os.getcwd()).expanduser().resolve()

    try:
        from agent.transports.codex_app_server import CodexGsdRunner  # type: ignore[import-not-found]

        runner = CodexGsdRunner(workspace)
        if gsd_args[0] == "resume":
            if len(gsd_args) < 3:
                print(
                    "usage: hermes gsd resume <SESSION_ID> <answer> [--workspace PATH]",
                    file=sys.stderr,
                )
                return 2
            session_id = gsd_args[1]
            answer = " ".join(gsd_args[2:])
            result = runner.resume(session_id, answer)
        else:
            if len(gsd_args) != 1:
                print(
                    "usage: hermes gsd <command> [--workspace PATH]",
                    file=sys.stderr,
                )
                return 2
            result = runner.run_gsd_command(gsd_args[0])
    except Exception as exc:
        print(f"hermes gsd: {exc}", file=sys.stderr)
        return 1

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return int(getattr(result, "returncode", 0) or 0)
