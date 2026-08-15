"""Cross-agent tool catalog contract.

The sub-agent (``StockAnalysisAgent``) and the orchestrator
(``StrategyMatchAgent``) each have a deliberately partitioned tool
set — ``read_file`` is sub-agent-only, ``run_analyze_stock`` is
orchestrator-only. The system prompt that each script injects
**must** mirror the same split, or the model will see tools it
can't actually call (or won't see tools it can).

This module is the contract test. It renders both scripts' system
prompts with the bundled catalog helpers and asserts the boundary.
"""
from __future__ import annotations

import re

import pytest

from stock_analysis_agent.script import analyze_stock as sub
from stock_analysis_agent.script import evaluate_strategy as orch


def _tool_names_in(markdown: str) -> set[str]:
    """Extract the set of ``### `<tool>` `` headings from a rendered tool index."""
    return set(re.findall(r"^### `([a-z_]+)`", markdown, flags=re.MULTILINE))


class TestSubAgentToolCatalog:
    """Sub-agent's ``<!-- TOOL_INDEX -->`` advertises only its own tools."""

    def test_default_subagent_catalog_excludes_orchestrator_only_tools(
        self,
    ) -> None:
        """Without shell, sub-agent sees ``load_skill`` and ``read_file``."""
        prompt = sub._load_system_prompt(include_shell_tool=False)
        names = _tool_names_in(prompt)
        assert "load_skill" in names
        assert "read_file" in names
        # list_dir was removed from the project — confirm it doesn't sneak back in.
        assert "list_dir" not in names
        # Orchestrator-only tools must not leak in.
        assert "run_analyze_stock" not in names, (
            "run_analyze_stock is the orchestrator's surface; the sub-agent "
            "must never see it in its prompt or it will try to call itself"
        )
        assert "load_strategy" not in names, (
            "load_strategy is orchestrator-only; the sub-agent doesn't load strategies"
        )
        # Shell is opt-in and was not requested.
        assert "run_command" not in names

    def test_subagent_catalog_includes_run_command_when_shell_enabled(
        self,
    ) -> None:
        """``include_shell_tool=True`` adds ``run_command`` symmetrically."""
        prompt = sub._load_system_prompt(include_shell_tool=True)
        names = _tool_names_in(prompt)
        assert "run_command" in names
        # Still must not include orchestrator-only tools.
        assert "run_analyze_stock" not in names
        assert "load_strategy" not in names

    def test_subagent_catalog_helpers_dedupe_and_sort(self) -> None:
        """The Python helpers return a sorted, deduplicated allowlist."""
        # Default call — should already be sorted + deduped.
        names = sub._subagent_tool_names(include_shell_tool=False)
        assert names == sorted(set(names))
        assert "run_analyze_stock" not in names
        assert "load_strategy" not in names

    def test_subagent_catalog_helpers_add_run_command_when_enabled(self) -> None:
        """Opt-in shell surfaces ``run_command`` in the helper output."""
        default = sub._subagent_tool_names(include_shell_tool=False)
        enabled = sub._subagent_tool_names(include_shell_tool=True)
        assert "run_command" not in default
        assert "run_command" in enabled


class TestOrchestratorToolCatalog:
    """Orchestrator's ``<!-- TOOL_INDEX -->`` advertises only its own tools."""

    def test_default_orchestrator_catalog_excludes_subagent_only_tools(
        self,
    ) -> None:
        """Without shell, orchestrator sees workflow glue only — no file/dir primitives."""
        prompt = orch._load_system_prompt(include_shell_tool=False)
        names = _tool_names_in(prompt)
        # Orchestration surface:
        assert "load_skill" in names
        assert "load_strategy" in names
        assert "run_analyze_stock" in names
        assert "run_deepresearch" in names
        # Sub-agent's data-discovery primitives MUST NOT leak into the
        # orchestrator's prompt — that's how the LLM ends up trying
        # to do raw research instead of delegating.
        assert "read_file" not in names, (
            "read_file is the sub-agent's primitive; orchestrator must not see it"
        )
        # list_dir was removed from the project entirely.
        assert "list_dir" not in names
        # Shell is opt-in.
        assert "run_command" not in names

    def test_orchestrator_catalog_includes_run_command_when_shell_enabled(
        self,
    ) -> None:
        """``include_shell_tool=True`` adds ``run_command`` symmetrically."""
        prompt = orch._load_system_prompt(include_shell_tool=True)
        names = _tool_names_in(prompt)
        assert "run_command" in names
        # Still must not include sub-agent-only tools.
        assert "read_file" not in names
        assert "list_dir" not in names

    def test_orchestrator_catalog_helpers_dedupe_and_sort(self) -> None:
        """The Python helpers return a sorted, deduplicated allowlist."""
        names = orch._orchestrator_tool_names(include_shell_tool=False)
        assert names == sorted(set(names))
        # Workflow glue must be present.
        assert "load_strategy" in names
        assert "run_analyze_stock" in names
        assert "run_deepresearch" in names
        # And sub-agent-only tools must be absent.
        assert "read_file" not in names
        assert "list_dir" not in names


class TestCatalogInvariants:
    """Both catalogs must render without raising, and the boundary is asymmetric."""

    @pytest.mark.parametrize("include_shell_tool", [False, True])
    def test_both_prompts_render_for_every_shell_setting(
        self, include_shell_tool: bool
    ) -> None:
        """Smoke: both scripts produce a populated ``<!-- TOOL_INDEX -->``."""
        sub_md = sub._load_system_prompt(include_shell_tool=include_shell_tool)
        orch_md = orch._load_system_prompt(include_shell_tool=include_shell_tool)
        assert "<!-- TOOL_INDEX -->" not in sub_md, (
            "analyze_stock template still contains unresolved placeholder"
        )
        assert "<!-- TOOL_INDEX -->" not in orch_md, (
            "evaluate_strategy template still contains unresolved placeholder"
        )
        assert "### `" in sub_md
        assert "### `" in orch_md

    def test_catalog_boundary_is_asymmetric(self) -> None:
        """Documentary: capture the full boundary once so reviewers can see it.

        Not a pass/fail — the real assertions live in the per-agent
        classes above. This test exists to surface the partition when
        someone runs the file alone with ``-s``.
        """
        sub_names = _tool_names_in(
            sub._load_system_prompt(include_shell_tool=False)
        )
        orch_names = _tool_names_in(
            orch._load_system_prompt(include_shell_tool=False)
        )
        # The shared tool is ``load_skill``; everything else is partitioned.
        assert sub_names != orch_names, (
            "sub-agent and orchestrator must NOT advertise identical catalogs"
        )
        # Sub-agent-only: file discovery primitive.
        assert "read_file" in (sub_names - orch_names), (
            "read_file must remain a sub-agent-only primitive"
        )
        # Orchestrator-only: strategy glue + sub-agent runner.
        assert {"load_strategy", "run_analyze_stock", "run_deepresearch"}.issubset(
            orch_names - sub_names
        )
        # list_dir was removed from the project entirely.
        assert "list_dir" not in sub_names
        assert "list_dir" not in orch_names
