"""Regression: OAuth errors wrapped in an anyio TaskGroup must be recognised
as auth errors so the initial-connect retry loop does NOT relaunch a fresh
OAuth authorization (new ``state``) while a human is completing the previous
one.

Root cause of the interactive ``hermes mcp login`` failure where a pasted /
callback redirect produced ``OAuthFlowError: State parameter mismatch``: the
MCP SDK runs its HTTP transport inside an anyio TaskGroup, so an
``OAuthFlowError`` raised mid-connect reaches ``MCPServerTask.run`` wrapped in
a ``BaseExceptionGroup`` ("unhandled errors in a TaskGroup"). The flat
``isinstance`` check in ``_is_auth_error`` missed it, the loop treated it as a
transient failure, and each retry (1s/2s/4s backoff) rebuilt the transport and
started a brand-new OAuth flow with a new ``state`` — impossibly faster than a
browser round-trip, so whichever flow the human completed carried a stale
state and never persisted tokens.
"""
import pytest

pytest.importorskip("mcp.client.auth.oauth2")


def test_is_auth_error_unwraps_taskgroup_oauth_flow_error():
    """A TaskGroup-wrapped OAuthFlowError (state mismatch) is an auth error."""
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    inner = OAuthFlowError(
        "State parameter mismatch: Ts17iOOl != -c5taf_l"
    )
    group = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    assert _is_auth_error(group) is True


def test_is_auth_error_unwraps_taskgroup_non_interactive():
    """A TaskGroup-wrapped OAuthNonInteractiveError is an auth error."""
    from tools.mcp_tool import _is_auth_error
    from tools.mcp_oauth import OAuthNonInteractiveError

    group = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [OAuthNonInteractiveError("no browser")],
    )
    assert _is_auth_error(group) is True


def test_is_auth_error_unwraps_nested_group():
    """Auth errors nested more than one group deep are still detected."""
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    nested = ExceptionGroup(
        "outer",
        [ExceptionGroup("inner", [OAuthFlowError("mismatch")])],
    )
    assert _is_auth_error(nested) is True


def test_is_auth_error_finds_auth_among_siblings():
    """A group mixing a non-auth error with an auth error is auth-related."""
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    group = ExceptionGroup("tg", [ValueError("noise"), OAuthFlowError("m")])
    assert _is_auth_error(group) is True


def test_is_auth_error_grouped_non_auth_stays_false():
    """Unwrapping must not turn a non-auth group into a false positive."""
    from tools.mcp_tool import _is_auth_error

    group = ExceptionGroup("tg", [ValueError("noise"), RuntimeError("x")])
    assert _is_auth_error(group) is False


def test_is_auth_error_grouped_httpx_500_stays_false():
    """A grouped non-401 HTTP error is still not an auth error."""
    from tools.mcp_tool import _is_auth_error
    import httpx
    from unittest.mock import MagicMock

    response = MagicMock()
    response.status_code = 500
    exc = httpx.HTTPStatusError("oops", request=MagicMock(), response=response)
    group = ExceptionGroup("tg", [exc])
    assert _is_auth_error(group) is False


def test_is_auth_error_grouped_httpx_401_true():
    """A grouped 401 HTTP error is an auth error."""
    from tools.mcp_tool import _is_auth_error
    import httpx
    from unittest.mock import MagicMock

    response = MagicMock()
    response.status_code = 401
    exc = httpx.HTTPStatusError("unauth", request=MagicMock(), response=response)
    group = ExceptionGroup("tg", [exc])
    assert _is_auth_error(group) is True
