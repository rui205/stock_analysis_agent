"""Tests for stock_analysis_agent.tools.lark_errors.

``run_command`` returns raw stderr. When the command is ``lark-cli``,
errors arrive as JSON envelopes and the LLM can miss the ``error.type``
field inside long blobs. ``classify_lark_error`` extracts that field and
returns a single-line structured signal so the agent can pattern-match
reliably.

Signal sentinels (stable contract — LLM prompt + tests both match on
these substrings, do not rename without updating both):

* ``[LARK_AUTH_REQUIRED]``         — caller must run the split-flow from
                                      the ``lark-shared`` skill.
* ``[LARK_CONFIRMATION_REQUIRED]`` — caller must ask the user, then
                                      retry with ``--yes``.
* ``[LARK_ERROR]``                 — generic envelope error; caller
                                      should surface ``error.message``.
"""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.lark_errors import (
    AUTH_REQUIRED_PREFIX,
    CONFIRMATION_REQUIRED_PREFIX,
    LARK_ERROR_PREFIX,
    classify_lark_error,
    is_lark_cli_command,
)


# ---------------------------------------------------------------------------
# is_lark_cli_command
# ---------------------------------------------------------------------------


def test_is_lark_cli_command_matches_bare_name() -> None:
    assert is_lark_cli_command("lark-cli") is True


def test_is_lark_cli_command_matches_absolute_path() -> None:
    """The tool is often invoked with an absolute path like
    ``/Users/me/.local/bin/lark-cli``; the basename is what matters.
    """
    assert is_lark_cli_command("/usr/local/bin/lark-cli") is True


def test_is_lark_cli_command_is_case_insensitive() -> None:
    """Be lenient — the basename is compared case-insensitively so
    ``LARK-CLI`` / ``Lark-Cli`` on weird filesystems still match.
    """
    assert is_lark_cli_command("LARK-CLI") is True
    assert is_lark_cli_command("/opt/Lark-Cli") is True


def test_is_lark_cli_command_rejects_unrelated() -> None:
    assert is_lark_cli_command("lark") is False
    assert is_lark_cli_command("lark-cli-fork") is False
    assert is_lark_cli_command("git") is False
    assert is_lark_cli_command("echo") is False
    assert is_lark_cli_command("") is False


# ---------------------------------------------------------------------------
# classify_lark_error — non-applicable cases (must return None)
# ---------------------------------------------------------------------------


def test_classify_returns_none_for_non_lark_command() -> None:
    """git, ls, python3 etc. are not lark-cli — must short-circuit."""
    stderr = '{"ok": false, "error": {"type": "auth_required"}}'
    assert classify_lark_error(command="git", stderr=stderr, exit_code=1) is None


def test_classify_returns_none_when_stderr_has_no_json() -> None:
    """Plain-text stderr (e.g. lark-cli not installed) is not a signal."""
    assert classify_lark_error(
        command="lark-cli", stderr="lark-cli: command not found\n",
        exit_code=127,
    ) is None


def test_classify_returns_none_when_envelope_has_no_error_field() -> None:
    """A JSON envelope without ``error`` is not an error response — it's
    likely a success payload that the agent must parse differently.
    """
    stderr = '{"ok": true, "data": {"doc_id": "doxcnXXX"}}'
    assert classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=0,
    ) is None


def test_classify_returns_none_for_malformed_json() -> None:
    """A truncated/unbalanced JSON object must not crash the classifier."""
    assert classify_lark_error(
        command="lark-cli", stderr='{"ok": false, "error": {',
        exit_code=1,
    ) is None


# ---------------------------------------------------------------------------
# classify_lark_error — auth_required
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "err_type",
    [
        "auth_required",
        "not_authenticated",
        "token_expired",
        "auth.login_required",
    ],
)
def test_classify_maps_auth_related_types_to_auth_required(err_type: str) -> None:
    """Any error.type that contains "auth" (case-insensitive) or matches
    the known ``token_expired`` alias maps to ``[LARK_AUTH_REQUIRED]``.
    The signal line must mention the original type so the agent can log it.
    """
    stderr = (
        '{"ok": false, "error": {"type": "' + err_type + '", '
        '"message": "user not logged in", '
        '"hint": "run lark-cli auth login --scope docs:document:create"}}'
    )
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    assert signal.startswith(AUTH_REQUIRED_PREFIX)
    # The signal carries the original error.type so the LLM can log it
    # verbatim (useful when the user reports a bug).
    assert f"type={err_type!r}" in signal
    # The hint is the actionable part — surface it.
    assert "hint=" in signal
    assert "lark-cli auth login" in signal


def test_classify_auth_required_works_with_absolute_path_command() -> None:
    """The command-basename check must work for absolute paths too."""
    stderr = '{"ok": false, "error": {"type": "auth_required"}}'
    signal = classify_lark_error(
        command="/opt/lark-cli/bin/lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    assert signal.startswith(AUTH_REQUIRED_PREFIX)


def test_classify_auth_required_extracts_message_when_no_hint() -> None:
    """Some envelopes only carry ``message`` — surface it instead of hint."""
    stderr = '{"ok": false, "error": {"type": "auth_required", "message": "no token"}}'
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    assert "message=" in signal
    assert "no token" in signal


# ---------------------------------------------------------------------------
# classify_lark_error — confirmation_required (high-risk-write gate)
# ---------------------------------------------------------------------------


def test_classify_maps_confirmation_required_with_risk_action() -> None:
    """``confirmation_required`` envelopes carry ``risk.action`` — that
    is the field the agent must echo back to the user verbatim so they
    know which operation needs confirmation.
    """
    stderr = (
        '{"ok": false, "error": {'
        '"type": "confirmation_required", '
        '"message": "drive +delete requires confirmation", '
        '"hint": "add --yes to confirm", '
        '"risk": {"level": "high-risk-write", "action": "drive +delete"}}}'
    )
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=10,
    )
    assert signal is not None
    assert signal.startswith(CONFIRMATION_REQUIRED_PREFIX)
    # Action is the user-facing piece — must be in the signal.
    assert "action=" in signal
    assert "drive +delete" in signal
    # Hint is the recovery command — must be in the signal.
    assert "add --yes" in signal


def test_classify_confirmation_required_without_risk_field() -> None:
    """Older envelopes may not have ``risk`` — the classifier must still
    recognize ``confirmation_required`` and surface what's available.
    """
    stderr = '{"ok": false, "error": {"type": "confirmation_required", "hint": "add --yes"}}'
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=10,
    )
    assert signal is not None
    assert signal.startswith(CONFIRMATION_REQUIRED_PREFIX)
    assert "add --yes" in signal


# ---------------------------------------------------------------------------
# classify_lark_error — generic envelope error
# ---------------------------------------------------------------------------


def test_classify_generic_envelope_error() -> None:
    """An envelope with an unknown ``error.type`` still maps to a
    structured signal so the LLM at least sees ``[LARK_ERROR]`` and the
    type/message verbatim — instead of an unstructured stderr blob.
    """
    stderr = (
        '{"ok": false, "error": {'
        '"type": "permission_violations", '
        '"message": "missing scope: docs:document:create", '
        '"hint": "see https://open.feishu.cn/app/X/cli"}}'
    )
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    assert signal.startswith(LARK_ERROR_PREFIX)
    assert "permission_violations" in signal
    assert "missing scope" in signal
    # The console URL must survive into the signal — the user/agent may
    # need it to fix the scope in the developer console.
    assert "https://open.feishu.cn" in signal


def test_classify_generic_error_without_hint() -> None:
    stderr = '{"ok": false, "error": {"type": "rate_limited", "message": "too many requests"}}'
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=429,
    )
    assert signal is not None
    assert signal.startswith(LARK_ERROR_PREFIX)
    assert "rate_limited" in signal
    assert "too many requests" in signal


# ---------------------------------------------------------------------------
# classify_lark_error — envelope parsing edge cases
# ---------------------------------------------------------------------------


def test_classify_handles_envelope_embedded_in_larger_stderr() -> None:
    """Real lark-cli output sometimes wraps the envelope with a log
    prefix. The classifier must locate the first balanced JSON object,
    not require the envelope to be at byte 0.
    """
    stderr = (
        "2026-07-15 10:00:00 [lark-cli] request failed\n"
        '{"ok": false, "error": {"type": "auth_required", "message": "expired"}}\n'
        "request_id: abc123\n"
    )
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    assert signal.startswith(AUTH_REQUIRED_PREFIX)


def test_classify_ignores_non_dict_envelope() -> None:
    """Defensive: if the first JSON object is a list/string/number, the
    classifier must not crash and must not classify it as an error.
    """
    stderr = '[1, 2, 3]\n{"ok": false, "error": {"type": "auth_required"}}\n'
    signal = classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    )
    assert signal is not None
    # The dict envelope (second object) is what we care about.
    assert signal.startswith(AUTH_REQUIRED_PREFIX)


def test_classify_ignores_non_object_error_field() -> None:
    """Defensive: ``error`` is occasionally a string, not a dict.
    Must not crash.
    """
    stderr = '{"ok": false, "error": "something went wrong"}'
    assert classify_lark_error(
        command="lark-cli", stderr=stderr, exit_code=1,
    ) is None