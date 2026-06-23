"""Subprocess wrapper around the ``lark-cli`` binary (Lark/Feishu CLI).

The wrapper class is named ``FeishuCli`` because the product brand is
Feishu (international: Lark). The actual binary it wraps is ``lark-cli``
v1.0.56+.

This is intentionally NOT a ``@tool`` — only ``script/analyze_stock.py``
calls into it; the LLM never sees these methods.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeishuDocRef:
    """Reference to a Feishu document returned by the CLI.

    ``doc_id`` holds whatever the CLI returned — a URL or a token. Both
    are accepted by ``lark-cli`` as ``--doc`` values.
    """

    doc_id: str
    url: str
    title: str


class FeishuCliError(RuntimeError):
    """Raised when ``lark-cli`` returns non-zero or emits non-empty stderr
    that the wrapper treats as an error."""

    def __init__(self, message: str, *, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


_Runner = Callable[..., subprocess.CompletedProcess]


def _default_runner() -> _Runner:
    return partial(
        subprocess.run,
        capture_output=True,
        text=True,
        check=False,
    )


def _coerce_doc_id(item: dict[str, Any]) -> str:
    for key in ("doc_id", "id", "token", "document_id"):
        if key in item and item[key] is not None:
            return str(item[key])
    raise FeishuCliError(
        f"lark-cli search result missing doc id; keys={list(item.keys())}",
        returncode=0,
        stderr="",
    )


def _coerce_url(item: dict[str, Any]) -> str:
    for key in ("url", "document_url", "link"):
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def _coerce_title(item: dict[str, Any]) -> str:
    for key in ("title", "name", "document_title"):
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    """Normalise the various JSON shapes ``+search`` may return.

    Tries, in order: ``{"items": [...]}``, ``{"data": [...]}``,
    ``{"docs": [...]}``, then the root if it's a list. Anything else
    is treated as "no matches".
    """
    if isinstance(payload, list):
        return [it for it in payload if isinstance(it, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "docs", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [it for it in value if isinstance(it, dict)]
    return []


class FeishuCli:
    """Thin wrapper around ``lark-cli docs`` subcommands.

    All methods run synchronously and block until the CLI exits. The
    ``runner`` parameter accepts any callable with the same signature as
    :func:`subprocess.run`; tests inject a fake.
    """

    def __init__(
        self,
        *,
        binary: str | Path = "lark-cli",
        timeout: float = 30.0,
        runner: _Runner | None = None,
    ) -> None:
        self._binary = str(binary)
        self._timeout = timeout
        self._runner: _Runner = runner if runner is not None else _default_runner()

    def _run(
        self, cmd: list[str], *, input: str | None = None
    ) -> subprocess.CompletedProcess:
        if input is None:
            return self._runner(cmd, timeout=self._timeout)
        return self._runner(cmd, input=input, timeout=self._timeout)

    def _check(self, result: subprocess.CompletedProcess, action: str) -> None:
        if result.returncode != 0 or (result.stderr or "").strip():
            raise FeishuCliError(
                f"lark-cli {action} failed (rc={result.returncode}): "
                f"{(result.stderr or '').strip()}",
                returncode=result.returncode,
                stderr=(result.stderr or "").strip(),
            )

    def list_matching_docs(self, query: str) -> list[FeishuDocRef]:
        """Search Feishu docs by ``query`` and return matching refs.

        Runs ``lark-cli docs +search --query <query> --json --as user``.
        The response envelope shape is not formally documented; the
        wrapper accepts several common variants — see
        :func:`_extract_items`.
        """
        cmd = [
            self._binary, "docs", "+search",
            "--query", query,
            "--json",
            "--as", "user",
        ]
        result = self._run(cmd)
        self._check(result, "docs +search")
        if not (result.stdout or "").strip():
            return []
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise FeishuCliError(
                f"lark-cli docs +search returned non-JSON: {e}",
                returncode=result.returncode,
                stderr=result.stderr,
            ) from e
        return [
            FeishuDocRef(
                doc_id=_coerce_doc_id(item),
                url=_coerce_url(item),
                title=_coerce_title(item),
            )
            for item in _extract_items(payload)
        ]

    def create_doc(self, *, title: str, content_file: Path) -> FeishuDocRef:
        """Create a new doc with ``title`` and the file's body as Markdown.

        ``+create`` has no ``--title`` flag, so the wrapper prepends a
        ``# {title}\\n\\n`` heading to the file body and pipes the result
        via stdin to ``--content -``.
        """
        body = content_file.read_text(encoding="utf-8")
        full = f"# {title}\n\n{body}"
        cmd = [
            self._binary, "docs", "+create",
            "--content", "-",
            "--doc-format", "markdown",
            "--as", "user",
            "--json",
        ]
        result = self._run(cmd, input=full)
        self._check(result, "docs +create")
        payload = json.loads(result.stdout)
        return FeishuDocRef(
            doc_id=_coerce_doc_id(payload),
            url=_coerce_url(payload),
            title=_coerce_title(payload) or title,
        )

    def append_to_doc(self, doc_id: str, *, content_file: Path) -> None:
        """Append the file's body as Markdown to the doc identified by
        ``doc_id`` (URL or token)."""
        cmd = [
            self._binary, "docs", "+update",
            "--command", "append",
            "--doc", doc_id,
            "--content", f"@{content_file}",
            "--doc-format", "markdown",
            "--as", "user",
        ]
        result = self._run(cmd)
        self._check(result, "docs +update --command append")


__all__ = ["FeishuCli", "FeishuCliError", "FeishuDocRef"]
