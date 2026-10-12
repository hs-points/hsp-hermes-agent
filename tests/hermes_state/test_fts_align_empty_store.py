"""HSP regression: an EMPTY store (no messages) whose ``messages_fts`` still reads raw
``messages`` (pre-v0.21 shape) must open cleanly on the first try.

Upstream's empty-store realign wraps ``do_align`` in ``SAVEPOINT fts_align_empty``,
but ``_ensure_fts_schema`` runs the FTS DDL with ``executescript``, which COMMITs any
open transaction first and so discards the savepoint. The following
``RELEASE SAVEPOINT`` then raised ``no such savepoint: fts_align_empty`` and the first
``SessionDB`` open failed (a gateway/dashboard start crash on upgrade for every
profile with an empty history). The realign work itself is idempotent, so a second
open succeeded, which hid the bug.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION


def _rewind_to_raw_messages_index(db: SessionDB) -> None:
    db._conn.execute("DROP TABLE messages_fts")
    db._conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id')"
    )
    db._conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )


@pytest.mark.parametrize("with_session", [False, True])
def test_empty_store_with_raw_messages_index_opens_first_try(tmp_path, with_session):
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    if with_session:
        first.create_session("session", source="cli")  # a session, but still no messages
    _rewind_to_raw_messages_index(first)
    first.close()

    migrated = SessionDB(db_path=path)  # raised OperationalError before the fix
    try:
        assert migrated.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        index_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
        assert migrated._conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        # The store is fully usable afterwards: write, then find it via the index.
        if not with_session:
            migrated.create_session("session", source="cli")
        migrated.append_message("session", role="user", content="hello aligned index")
        assert migrated.search_messages("aligned")
    finally:
        migrated.close()

    again = SessionDB(db_path=path)  # and stays openable
    again.close()
