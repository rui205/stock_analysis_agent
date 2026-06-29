"""Subprocess runner tool — invoke CLI programs (lark-cli, git, curl, …).

The agent can call :func:`run_command` to shell out to any executable on
``PATH``. This is the bridge that lets a tool-using agent actually
**execute** CLIs (not just read about them via ``load_skill``) — the
canonical use case is ``lark-cli docs +create`` to publish a Lark
document.

``argv`` is passed as a list (no shell expansion) so the LLM cannot
inject shell metacharacters to escalate beyond what ``subprocess.run``
with ``shell=False`` allows. The tool is **opt-in** at the agent level
— see ``StockAnalysisAgent(include_shell_tool=...)``.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from langchain.tools import tool

logger = logging.getLogger(__name__)

#: Hard cap on stdout/stderr size returned to the LLM. Larger output is
#: truncated with a marker so the LLM context window is not consumed.
MAX_OUTPUT_BYTES = 30_000

#: Default timeout (seconds) for a subprocess invocation.
DEFAULT_TIMEOUT_SECONDS = 60


def _truncate(stream_name: str, text: str) -> str:
    """Truncate ``text`` to ``MAX_OUTPUT_BYTES`` if it exceeds the cap."""
    if len(text) <= MAX_OUTPUT_BYTES:
        return text
    head = text[:MAX_OUTPUT_BYTES]
    return f"{head}\n... [truncated, {stream_name} > {MAX_OUTPUT_BYTES} bytes]"


def _format_result(
    *,
    cmd: list[str],
    cwd: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    timeout: int,
) -> str:
    """Build the structured text response returned to the LLM."""
    parts: list[str] = [
        "$ " + " ".join(cmd),
        f"cwd: {cwd}",
    ]
    if timed_out:
        parts.append(f"=== TIMEOUT (after {timeout}s) ===")
    else:
        parts.append(f"=== exit={returncode} ===")
    parts.append(f"--- stdout ---\n{_truncate('stdout', stdout)}")
    parts.append(f"--- stderr ---\n{_truncate('stderr', stderr)}")
    return "\n".join(parts)


def _run_subprocess(
    command: str,
    argv: list[str],
    cwd: str | None,
    timeout: int,
) -> str:
    """Internal subprocess runner — exposed for testing without ``@tool``.

    Args:
        command: The executable name (e.g. ``"lark-cli"``) or absolute path.
        argv: Argument list. **No** shell expansion.
        cwd: Working directory. ``None`` means use the parent process cwd.
        timeout: Seconds before the subprocess is killed.

    Returns:
        Formatted text containing ``$ <cmd>``, ``cwd: ...``, the exit
        code or timeout marker, and the (possibly truncated) stdout and
        stderr streams.

    Raises:
        ValueError: ``command`` is empty.
        TypeError: ``argv`` is not a list of strings.
        FileNotFoundError: ``command`` is not on ``PATH`` (and not an
            absolute path to an existing executable).
    """
    if not command or not command.strip():
        raise ValueError("command cannot be empty")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise TypeError("argv must be a list of strings")

    work_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd()
    cmd: list[str] = [command, *argv]

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as e:
        logger.warning("subprocess timed out: %s after %ss", cmd, timeout)
        stdout = e.stdout if isinstance(e.stdout, str) else (
            e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        )
        stderr = e.stderr if isinstance(e.stderr, str) else (
            e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        )
        return _format_result(
            cmd=cmd, cwd=work_dir, returncode=-1,
            stdout=stdout, stderr=stderr,
            timed_out=True, timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"command not found on PATH: {command!r} "
            f"(cwd={work_dir}); check `which {command.split('/')[-1]}`"
        ) from None

    return _format_result(
        cmd=cmd, cwd=work_dir, returncode=proc.returncode,
        stdout=proc.stdout or "", stderr=proc.stderr or "",
        timed_out=False, timeout=timeout,
    )


@tool("run_command")
def run_command(
    command: str,
    argv: list[str],
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run a CLI subprocess and return its stdout / stderr / exit code as text.

    Use this to invoke CLI programs (lark-cli, git, curl, jq, ls, …).
    The canonical case for the stock-analyst agent is ``lark-cli docs
    +create --content <xml>...`` to publish the analysis report as a
    Lark cloud document.

    ``argv`` is passed as a list — **no** shell expansion. Quoting, glob
    characters, and newlines are preserved verbatim. Stdout and stderr
    are truncated if longer than 30 KB.

    Args:
        command: The program name or absolute path (e.g. ``"lark-cli"``).
        argv: Argument list as a list of strings. E.g.
            ``["docs", "+create", "--content", "<title>...</title>..."]``.
        cwd: Working directory. ``None`` means use the parent process cwd.
        timeout: Seconds before the subprocess is killed (default 60).

    Returns:
        A formatted text block::

            $ <command> <args...>
            cwd: <cwd>
            === exit=<N> ===          (or "=== TIMEOUT (after <N>s) ===")
            --- stdout ---
            <truncated stdout>
            --- stderr ---
            <truncated stderr>

    Raises:
        ValueError: ``command`` is empty.
        TypeError: ``argv`` is not a list of strings.
        FileNotFoundError: ``command`` is not on ``PATH``.
    """
    return _run_subprocess(command, argv, cwd, timeout)


__all__ = ["run_command", "_run_subprocess", "MAX_OUTPUT_BYTES", "DEFAULT_TIMEOUT_SECONDS"]
