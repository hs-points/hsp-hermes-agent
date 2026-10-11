"""Classification exactly as tools/mcp_tool_server_run.py applies it to grouped errors."""
from tools import mcp_tool_errors as _errors


def classify(exc: BaseException) -> bool:
    return _errors._is_auth_error(_errors._unwrap_exception_group(exc))
