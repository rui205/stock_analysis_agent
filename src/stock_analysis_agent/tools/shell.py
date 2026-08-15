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

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from stock_analysis_agent.tools.lark_errors import classify_lark_error

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
        ValueError: ``command`` is empty, or it contains whitespace /
            quotes / shell metacharacters — a common LLM mistake is
            to pass an entire shell command (e.g.
            ``'echo "..."'``) as the single ``command`` argument.
            Split it: ``command="echo"``,
            ``argv=['...']``.
        TypeError: ``argv`` is not a list of strings.
        FileNotFoundError: ``command`` is not on ``PATH`` (and not an
            absolute path to an existing executable).
    """
    if not command or not command.strip():
        raise ValueError("command cannot be empty")
    # A program name is a single token — spaces, quotes, or shell
    # metacharacters in ``command`` mean the LLM concatenated an entire
    # shell command into the field instead of splitting it into
    # ``command`` + ``argv``. We reject this with a precise hint so the
    # model can correct itself on the next turn.
    if any(ch.isspace() or ch in {'"', "'", '`', '&', '|', ';', '>', '<'} for ch in command):
        raise ValueError(
            f"command {command!r} looks like a shell command, not a program "
            "name — pass the executable in `command` and the rest as "
            "separate entries in `argv` (e.g. command='echo', "
            "argv=['hello']). The tool does NOT invoke a shell."
        )
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

    result = _format_result(
        cmd=cmd, cwd=work_dir, returncode=proc.returncode,
        stdout=proc.stdout or "", stderr=proc.stderr or "",
        timed_out=False, timeout=timeout,
    )
    # On non-zero exit, surface a structured ``[LARK_*]`` signal line
    # above the raw output so the agent can pattern-match lark-cli
    # failures (auth_required, confirmation_required, generic errors)
    # without parsing the JSON envelope in stderr itself. The classifier
    # is a no-op for non-lark commands and for envelopes it cannot
    # parse — see ``tools.lark_errors.classify_lark_error``.
    signal = classify_lark_error(
        command=command,
        stderr=proc.stderr or "",
        exit_code=proc.returncode,
    )
    if signal is not None:
        return f"{signal}\n{result}"
    return result


class RunCommandInput(BaseModel):
    """Input schema for the ``run_command`` tool.

    The four-field schema mirrors the function signature verbatim.
    The tool does NOT spawn a shell — ``command`` is a single program
    name (or absolute path), ``argv`` is the argument list, and
    timeout/cwd control execution.
    """

    command: str = Field(
        description=(
            "The executable name or absolute path (e.g. `\"lark-cli\"`, "
            "`\"ls\"`, `\"cat\"`, `\"python3\"`). Pass the program name "
            "ONLY — flags AND positional arguments (paths, queries, "
            "script files) ALL belong in `argv`. Never pass a shell "
            "pipeline (`a && b`, `a | b`, `;`, `>`, `<`, backticks) or "
            "a program+args concatenation (`\"ls /path\"`, "
            "`\"python3 script.py --foo\"`) here; the tool does not "
            "invoke a shell. To chain commands, make multiple calls or "
            "explicitly invoke `bash`."
        ),
        min_length=1,
    )
    argv: list[str] = Field(
        description=(
            "Argument list as a list of strings, passed without shell "
            "expansion. Quoting, glob characters, and newlines are "
            "preserved verbatim. Every flag and positional argument "
            "goes here — never concatenated into `command`. Example: "
            "to run `lark-cli docs +create --content <xml>...`, pass "
            "`[\"docs\", \"+create\", \"--content\", \"<xml>...\"]`. "
            "To list a directory, pass `[\"-la\", \"src/.../skill\"]`."
        ),
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for the subprocess. `None` (default) "
            "means use the parent process's current directory. "
            "Absolute paths and `~`-prefixed paths are expanded."
        ),
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        description=(
            "Seconds before the subprocess is killed. Default 60. "
            "On timeout the result carries a "
            "`=== TIMEOUT (after <N>s) ===` marker instead of an "
            "exit code."
        ),
    )


@tool(
    "run_command",
    description=(
        "Run a CLI subprocess and return its stdout, stderr, and exit "
        "code as a formatted text block. The canonical use cases are:\n"
        "  • invoking `lark-cli docs +create --content <xml>...` to "
        "publish an analysis report as a 飞书 cloud document;\n"
        "  • running skill scripts, e.g. `python3 <skill>/scripts/"
        "get_data.py --query \"...\" --indicators \"...\"`;\n"
        "  • browsing the local filesystem via `ls`, `cat`, `head`, "
        "`pwd`, `find` (this is the ONLY way to list a directory — "
        "there is no separate `list_dir` tool).\n\n"
        "`argv` is passed as a list — NO shell expansion, so quoting / "
        "glob / newlines are preserved verbatim. stdout/stderr are "
        "truncated at 30KB; default timeout is 60s (subprocess is "
        "killed on timeout). `command` must be on PATH or an absolute "
        "path.\n\n"
        "ARGUMENT SHAPE — split program from args, ALWAYS: `command` is "
        "the SINGLE executable name (e.g. `\"lark-cli\"`, `\"ls\"`, "
        "`\"python3\"`); EVERY flag AND positional argument (paths, "
        "queries, script names) goes into `argv` as a separate string. "
        "The tool does NOT invoke a shell — there is no `cd`, no `&&`, "
        "no glob expansion, no variable interpolation.\n\n"
        "Two common LLM mistakes to avoid:\n"
        "  1. Concatenating program + args into `command` — e.g. "
        "`command=\"ls /some/path\"` or `command=\"python3 "
        "scripts/get_data.py --query \\\"...\\\"\"`. Both are rejected "
        "(whitespace in `command`). Split them: `command=\"ls\", "
        "argv=[\"/some/path\"]`.\n"
        "  2. Shell pipelines in `command` — e.g. `command=\"pwd && "
        "ls src/...\"` or `command=\"cat foo | grep bar\"`. Rejected "
        "(shell metacharacters). Either make two separate calls or "
        "explicitly invoke bash: `command=\"bash\", argv=[\"-lc\", "
        "\"<full shell command>\"]`.\n\n"
        "Concrete examples (all correct shapes):\n"
        "  • list a directory: `command=\"ls\", "
        "argv=[\"-la\", \"src/stock_analysis_agent/skill\"]`\n"
        "  • read a small file: `command=\"cat\", argv=[\"path/to/"
        "file\"]`\n"
        "  • run a skill script: `command=\"python3\", "
        "argv=[\"<skill>/scripts/get_data.py\", \"--query\", \"...\", "
        "\"--indicators\", \"...\"]`\n"
        "  • publish to lark: `command=\"lark-cli\", argv=[\"docs\", "
        "\"+create\", \"--api-version\", \"v2\", \"--content\", "
        "\"<xml>...\"]`"
    ),
    args_schema=RunCommandInput,
)
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


__all__ = [
    "RunCommandInput",
    "run_command",
    "_run_subprocess",
    "MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
]