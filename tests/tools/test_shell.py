"""Tests for stock_analysis_agent.tools.shell: the run_command tool.

The shell tool lets the LLM invoke arbitrary CLI programs (most importantly
``lark-cli`` to publish the analysis report as a Lark cloud document).
``argv`` is passed as a list (no shell expansion) to keep the API safe.
"""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.shell import (
    MAX_OUTPUT_BYTES,
    _run_subprocess,
    run_command,
)


# ---------------------------------------------------------------------------
# _run_subprocess — the pure function under @tool
# ---------------------------------------------------------------------------


def test_run_subprocess_invokes_echo_with_argv() -> None:
    """``echo`` writes its argv joined by spaces — no shell interpretation."""
    result = _run_subprocess("echo", ["hello", "world"], cwd=None, timeout=10)
    assert "$ echo hello world" in result
    assert "--- stdout ---" in result
    assert "hello world" in result
    assert "=== exit=0 ===" in result


def test_run_subprocess_captures_exit_code() -> None:
    """Non-zero exit codes are reflected in the formatted output."""
    # `false` is a POSIX builtin/external that always exits 1.
    result = _run_subprocess("false", [], cwd=None, timeout=10)
    assert "=== exit=1 ===" in result


def test_run_subprocess_captures_stderr() -> None:
    """Stderr is reported separately from stdout in the output."""
    # Use python instead of sh to avoid a shell.
    result = _run_subprocess(
        "python3", ["-c", "import sys; print('OUT'); sys.stderr.write('ERR\\n')"],
        cwd=None, timeout=10,
    )
    assert "OUT" in result
    assert "--- stdout ---" in result
    assert "--- stderr ---" in result
    assert "ERR" in result


def test_run_subprocess_respects_cwd(tmp_path) -> None:
    """A relative path run via ``cwd`` resolves under the given directory."""
    # `python3 -c "import os; print(os.getcwd())"` — print cwd
    result = _run_subprocess(
        "python3", ["-c", "import os; print(os.getcwd())"],
        cwd=str(tmp_path), timeout=10,
    )
    assert str(tmp_path) in result


def test_run_subprocess_timeout_returns_timeout_marker() -> None:
    """A subprocess that exceeds the timeout is killed and the result is
    marked ``=== TIMEOUT (after <N>s) ===`` — the LLM gets a clean signal.
    """
    result = _run_subprocess(
        "python3", ["-c", "import time; time.sleep(10)"],
        cwd=None, timeout=1,
    )
    assert "=== TIMEOUT (after 1s) ===" in result


def test_run_subprocess_truncates_large_output() -> None:
    """Stdout > ``MAX_OUTPUT_BYTES`` is truncated with a marker so the
    LLM context is bounded. The truncation removes bytes from the END of
    the stream, not the beginning — so the LLM still sees the start of
    whatever was produced.
    """
    # Distinct head/tail chars so we can check the tail is gone:
    # ``big[:MAX_OUTPUT_BYTES]`` is all ``x``, ``big[MAX_OUTPUT_BYTES:]``
    # is all ``y``. If the tail chars (``y``) survive in the truncated
    # stdout block, truncation did NOT take effect.
    head = "x" * MAX_OUTPUT_BYTES
    tail = "y" * 500
    big = head + tail
    result = _run_subprocess(
        "python3", ["-c", f"print({big!r})"],
        cwd=None, timeout=10,
    )
    # The truncation marker appears somewhere in the result.
    assert "[truncated, stdout > " in result
    # The TAIL characters (``y``) must be missing from the stdout block.
    # We isolate the stdout body — between the "--- stdout ---" header
    # and the "--- stderr ---" header — so the ``$ <cmd>`` echo line at
    # the top (which still contains the full literal command, including
    # the un-truncated arg) does not fool the assertion. The marker text
    # contains the word ``bytes`` which has a single ``y``, so we check
    # for 100 consecutive ``y`` chars (a substring that can only exist
    # if the tail survived truncation).
    stdout_start = result.index("--- stdout ---") + len("--- stdout ---\n")
    stdout_end = result.index("\n--- stderr ---")
    truncated_stdout = result[stdout_start:stdout_end]
    assert ("y" * 100) not in truncated_stdout, (
        "expected the y-tail to be removed by stdout truncation"
    )
    # The truncation marker lives inside the stdout block.
    assert "[truncated" in truncated_stdout


def test_run_subprocess_rejects_empty_command() -> None:
    """An empty command must raise ``ValueError`` — never call ``subprocess``."""
    with pytest.raises(ValueError, match="command"):
        _run_subprocess("", ["echo"], cwd=None, timeout=10)


def test_run_subprocess_rejects_command_with_whitespace() -> None:
    """The exact LLM failure mode from the run-logs:

    LLM called ``run_command(command='echo "triggering get_stock_snapshot..."',
    argv=[])`` — the entire shell line was stuffed into ``command``. Without
    this guard, ``subprocess.run`` would try to look up
    ``echo "triggering get_stock_snapshot..."`` on PATH and fail with a
    confusing ``FileNotFoundError``. The error message should instead name
    the actual mistake and hint at the correct split.
    """
    with pytest.raises(ValueError, match="command='echo'") as excinfo:
        _run_subprocess('echo "triggering get_stock_snapshot via skill tools"',
                        [], cwd=None, timeout=10)
    msg = str(excinfo.value)
    assert "argv" in msg, (
        "error message should hint at splitting into command + argv"
    )


def test_run_subprocess_rejects_command_with_quotes() -> None:
    """A bare quoted arg concatenated into ``command`` is the same mistake."""
    with pytest.raises(ValueError, match="shell command"):
        _run_subprocess("ls 'foo bar'", [], cwd=None, timeout=10)


def test_run_subprocess_rejects_non_list_argv() -> None:
    """A string for ``argv`` is a common LLM mistake — must raise clearly."""
    with pytest.raises(TypeError, match="argv"):
        _run_subprocess("echo", "hello world", cwd=None, timeout=10)


def test_run_subprocess_raises_filenotfound_for_missing_command() -> None:
    """An unknown command surfaces as ``FileNotFoundError`` so the LLM can recover."""
    with pytest.raises(FileNotFoundError):
        _run_subprocess(
            "definitely-not-installed-cli-xyz123", [], cwd=None, timeout=5,
        )


def test_run_subprocess_does_not_shell_interpret_argv() -> None:
    """Args are passed as argv — shell metacharacters are NOT interpreted.

    The LLM could legitimately pass glob characters, redirects, or quotes
    as literal strings (e.g. content blocks for ``lark-cli --content``)
    and we must not interpret them as shell syntax.
    """
    literal = "echo $HOME && cat /etc/passwd > /tmp/foo"
    result = _run_subprocess(
        "echo", [literal], cwd=None, timeout=10,
    )
    # echo writes the literal string to stdout — no shell expansion.
    assert literal in result
    # /tmp/foo should NOT have been created.
    import os
    assert not os.path.exists("/tmp/foo")


def test_run_subprocess_handles_non_utf8_bytes() -> None:
    """CLIs occasionally emit bytes that aren't valid UTF-8 (e.g. mixed GBK
    + UTF-8 output from ``lark-cli``). The tool must NOT crash with
    ``UnicodeDecodeError`` — bad bytes must be replaced with ``U+FFFD``
    so the LLM sees a clean, parseable response.
    """
    # Emit 0xd2 followed by an ASCII char — a clearly invalid UTF-8 byte.
    result = _run_subprocess(
        "python3", ["-c", "import sys; sys.stdout.buffer.write(b'\\xd2bad')"],
        cwd=None, timeout=10,
    )
    assert "=== exit=0 ===" in result
    # The U+FFFD replacement character (or '?') should appear in place
    # of the bad byte — the tool must not raise.
    assert "�" in result or "?" in result


# ---------------------------------------------------------------------------
# @tool run_command — schema and invocation
# ---------------------------------------------------------------------------


def test_tool_name_is_run_command() -> None:
    assert run_command.name == "run_command"


def test_tool_args_schema_has_command_argv_cwd_timeout() -> None:
    """The schema must declare the four parameters the tool accepts.

    For ``@tool``-decorated functions LangChain exposes the JSON schema
    as a flat dict whose keys are the parameter names (``"command"``,
    ``"argv"``, ``"cwd"``, ``"timeout"``) — not nested under a
    ``"properties"`` key as in Pydantic ``model_json_schema``.
    """
    schema = run_command.args
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    assert isinstance(schema, dict)
    assert "command" in schema
    assert "argv" in schema
    assert "cwd" in schema
    assert "timeout" in schema


def test_tool_description_mentions_lark_cli() -> None:
    """The tool description should advertise the lark-cli use case so the
    LLM has a strong retrieval hint when choosing tools.
    """
    desc = run_command.description
    if desc is None:
        # Tool description is on the inner function's docstring.
        desc = run_command.func.__doc__ or ""
    assert "lark-cli" in desc or "CLI" in desc


def test_tool_invoke_runs_command_and_returns_formatted_string() -> None:
    """End-to-end: ``tool.invoke`` runs the subprocess and returns text."""
    result = run_command.invoke({"command": "echo", "argv": ["hi"], "timeout": 10})
    assert isinstance(result, str)
    assert "hi" in result
