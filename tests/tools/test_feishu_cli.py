"""Tests for the FeishuCli subprocess wrapper around lark-cli."""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from stock_analysis_agent.tools.feishu_cli import (
    FeishuCli,
    FeishuCliError,
    FeishuDocRef,
)


class _FakeRunner:
    """Records calls and returns a preconfigured CompletedProcess."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _build_cli(runner: Callable[..., subprocess.CompletedProcess]) -> FeishuCli:
    return FeishuCli(binary="lark-cli", runner=runner)


# ---------------------------------------------------------------------------
# list_matching_docs
# ---------------------------------------------------------------------------


def test_list_matching_docs_invokes_search_with_query_and_json() -> None:
    runner = _FakeRunner(stdout='{"items": [{"doc_id": "d1", "url": "https://x", "title": "t"}]}')
    cli = _build_cli(runner)
    docs = cli.list_matching_docs("02319")
    assert docs == [FeishuDocRef(doc_id="d1", url="https://x", title="t")]
    cmd, _kwargs = runner.calls[0]
    cmd_list = cmd[0]
    assert cmd_list[:3] == ["lark-cli", "docs", "+search"]
    assert "--query" in cmd_list
    assert "02319" in cmd_list
    assert "--json" in cmd_list


def test_list_matching_docs_accepts_root_level_list_envelope() -> None:
    """Some lark-cli versions may return a top-level JSON array."""
    runner = _FakeRunner(stdout='[{"token": "x1", "document_url": "https://y", "name": "z"}]')
    cli = _build_cli(runner)
    docs = cli.list_matching_docs("02319")
    assert docs == [FeishuDocRef(doc_id="x1", url="https://y", title="z")]


def test_list_matching_docs_returns_empty_for_empty_stdout() -> None:
    runner = _FakeRunner(stdout="")
    cli = _build_cli(runner)
    assert cli.list_matching_docs("02319") == []


def test_list_matching_docs_raises_on_nonzero_return() -> None:
    runner = _FakeRunner(returncode=2, stderr="not found")
    cli = _build_cli(runner)
    with pytest.raises(FeishuCliError) as ei:
        cli.list_matching_docs("02319")
    assert "not found" in str(ei.value)


# ---------------------------------------------------------------------------
# create_doc
# ---------------------------------------------------------------------------


def test_create_doc_sends_stdin_with_title_prepended(tmp_path: Path) -> None:
    """Create prepends '# {title}\\n\\n' to the file body and sends via stdin."""
    src = tmp_path / "a.md"
    src.write_text("body contents\n", encoding="utf-8")
    runner = _FakeRunner(
        stdout='{"doc_id": "new", "url": "https://new", "title": "my title"}'
    )
    cli = _build_cli(runner)
    ref = cli.create_doc(title="my title", content_file=src)
    assert ref == FeishuDocRef(doc_id="new", url="https://new", title="my title")
    cmd, kwargs = runner.calls[0]
    cmd_list = cmd[0]
    assert cmd_list[:3] == ["lark-cli", "docs", "+create"]
    assert "--doc-format" in cmd_list
    assert "markdown" in cmd_list
    assert "--content" in cmd_list
    assert cmd_list[cmd_list.index("--content") + 1] == "-"  # stdin
    assert "--as" in cmd_list
    assert "user" in cmd_list
    # The stdin payload must begin with the Markdown title heading.
    assert kwargs["input"].startswith("# my title\n\n")


def test_create_doc_reads_content_from_file_for_stdin_payload() -> None:
    runner = _FakeRunner(stdout='{"doc_id": "n", "url": "u", "title": "t"}')

    import stock_analysis_agent.tools.feishu_cli as mod
    orig_read = mod.Path.read_text

    def _patched(self: Path, *, encoding: str = "utf-8") -> str:
        return "body line\n"

    mod.Path.read_text = _patched  # type: ignore[assignment]
    try:
        cli = _build_cli(runner)
        cli.create_doc(title="t", content_file=Path("/tmp/a.md"))
    finally:
        mod.Path.read_text = orig_read  # type: ignore[assignment]
    _, kwargs = runner.calls[0]
    assert kwargs["input"] == "# t\n\nbody line\n"


def test_create_doc_raises_on_nonzero_return(tmp_path: Path) -> None:
    src = tmp_path / "a.md"
    src.write_text("x", encoding="utf-8")
    runner = _FakeRunner(returncode=1, stderr="perm denied")
    cli = _build_cli(runner)
    with pytest.raises(FeishuCliError):
        cli.create_doc(title="t", content_file=src)


# ---------------------------------------------------------------------------
# append_to_doc
# ---------------------------------------------------------------------------


def test_append_to_doc_uses_update_command_append_with_doc_and_content_at_file() -> None:
    runner = _FakeRunner()
    cli = _build_cli(runner)
    cli.append_to_doc("d1", content_file=Path("/tmp/b.md"))
    cmd, _kwargs = runner.calls[0]
    cmd_list = cmd[0]
    assert cmd_list[:3] == ["lark-cli", "docs", "+update"]
    assert "--command" in cmd_list
    assert cmd_list[cmd_list.index("--command") + 1] == "append"
    assert "--doc" in cmd_list
    assert "d1" in cmd_list
    assert "--content" in cmd_list
    assert "@/tmp/b.md" in cmd_list
    assert "--doc-format" in cmd_list
    assert "markdown" in cmd_list
    assert "--as" in cmd_list
    assert "user" in cmd_list


def test_append_to_doc_raises_on_nonzero_return() -> None:
    runner = _FakeRunner(returncode=3, stderr="forbidden")
    cli = _build_cli(runner)
    with pytest.raises(FeishuCliError):
        cli.append_to_doc("d1", content_file=Path("/tmp/b.md"))


# ---------------------------------------------------------------------------
# FeishuDocRef
# ---------------------------------------------------------------------------


def test_feishu_doc_ref_is_frozen() -> None:
    """The ref dataclass must be hashable / immutable to be safe to pass around."""
    ref = FeishuDocRef(doc_id="d1", url="u", title="t")
    with pytest.raises(Exception):  # FrozenInstanceError, exact class is private
        ref.doc_id = "x"  # type: ignore[misc]
