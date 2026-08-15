"""Lark CLI error classification for ``run_command``.

``run_command`` returns raw ``stdout`` / ``stderr`` / exit code as text.
When the command is ``lark-cli``, errors arrive as JSON envelopes such
as::

    {"ok": false, "error": {"type": "auth_required", "message": "...",
                            "hint": "...", "risk": {...}}}

The LLM can miss the ``error.type`` field when the envelope is buried
in a long stderr blob or wrapped by a log prefix. This module extracts
the type and emits a single-line signal prefixed with a stable sentinel
that the agent can pattern-match reliably:

* :data:`AUTH_REQUIRED_PREFIX`         — caller must run the split-flow
  from the ``lark-shared`` skill (``lark-cli auth login --no-wait
  --json``, then ``--device-code`` after user confirms).
* :data:`CONFIRMATION_REQUIRED_PREFIX` — caller must ask the user,
  then retry with ``--yes`` appended.
* :data:`LARK_ERROR_PREFIX`            — generic envelope error;
  caller surfaces ``error.message`` / ``error.hint`` to the user.

The signal is **only** emitted when:

1. the command basename is ``lark-cli`` (matches ``lark-cli``,
   ``/abs/path/to/lark-cli``, case-insensitive), AND
2. the exit code is non-zero, AND
3. stderr contains a parseable JSON envelope with an ``error`` dict.

Otherwise the classifier returns ``None`` and the caller behaves as if
no enrichment is needed — non-lark commands and non-JSON stderr pass
through unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

#: Stable sentinel — matched by tests AND by the agent's prompt guidance.
#: Do not rename without updating both surfaces.
AUTH_REQUIRED_PREFIX: Final[str] = "[LARK_AUTH_REQUIRED]"

#: Stable sentinel — high-risk-write gate (exit code 10).
CONFIRMATION_REQUIRED_PREFIX: Final[str] = "[LARK_CONFIRMATION_REQUIRED]"

#: Stable sentinel — any other structured envelope error.
LARK_ERROR_PREFIX: Final[str] = "[LARK_ERROR]"


#: Error types whose substring ``"auth"`` (case-insensitive) flags an
#: authentication failure requiring the ``lark-shared`` split-flow.
#: Combined with the substring check, also covers aliases like
#: ``auth.login_required`` and ``token_expired``.
_AUTH_TYPE_KEYWORDS: Final[frozenset[str]] = frozenset({
    "auth_required",
    "not_authenticated",
    "token_expired",
})


def is_lark_cli_command(command: str) -> bool:
    """Return ``True`` if ``command`` resolves to the ``lark-cli`` binary.

    Matches the bare name (``"lark-cli"``), absolute paths
    (``"/usr/local/bin/lark-cli"``), and is case-insensitive on the
    basename. Does NOT match unrelated names like ``"lark"`` or
    ``"lark-cli-fork"`` — only the exact basename ``lark-cli``.
    """
    if not command:
        return False
    return Path(command).name.lower() == "lark-cli"


def _parse_envelope(stderr: str) -> dict | None:
    """Find the first balanced JSON dict in ``stderr`` and return it.

    Stops at the first parseable **dict** — lark-cli writes one
    JSON envelope per stderr line, and the first one with ``error`` is
    what we care about. Non-object JSON (lists, scalars) is skipped.

    Args:
        stderr: Raw stderr text from the subprocess.

    Returns:
        The first parseable dict, or ``None`` if no balanced JSON
        object exists in the text.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = stderr.find("{", idx)
        if start < 0:
            return None
        try:
            obj, end = decoder.raw_decode(stderr, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = end


def _format_signal(prefix: str, *, error_type: str, message: str = "",
                   hint: str = "", action: str = "") -> str:
    """Render the structured signal line.

    Args:
        prefix: One of the :data:`*_PREFIX` sentinels.
        error_type: The original ``error.type`` from the envelope.
        message: Optional ``error.message``.
        hint: Optional ``error.hint`` — the actionable recovery hint.
        action: Optional ``error.risk.action`` — the high-risk operation
            name for ``confirmation_required`` envelopes.

    Returns:
        A single-line string of the form
        ``<prefix> type='...' [hint='...'] [message='...'] [action='...']``.
        Trailing fields are omitted when empty so the signal stays tight.
    """
    parts: list[str] = [prefix, f"type={error_type!r}"]
    if action:
        parts.append(f"action={action!r}")
    if hint:
        parts.append(f"hint={hint!r}")
    if message:
        parts.append(f"message={message!r}")
    return " ".join(parts)


def classify_lark_error(
    *, command: str, stderr: str, exit_code: int,
) -> str | None:
    """Return a structured signal line for a failed ``lark-cli`` run.

    The function is a pure classifier — it does not mutate, log, or
    call any subprocess. ``run_command`` invokes it after a non-zero
    exit and prepends the returned line (if any) to the formatted
    output so the LLM sees a deterministic, pattern-matchable marker
    on top of the raw stderr blob.

    Args:
        command: The executable name passed to ``subprocess.run`` —
            either a bare program name or an absolute path.
        stderr: The captured stderr text from the subprocess (decoded
            with ``errors="replace"`` so it is always a valid string).
        exit_code: The subprocess return code. Zero is treated as
            "not an error" and short-circuits.

    Returns:
        A single-line structured signal beginning with one of the
        :data:`*_PREFIX` sentinels, or ``None`` if no enrichment
        applies (non-lark command, zero exit, no parseable envelope,
        or envelope with no ``error`` dict).
    """
    if exit_code == 0:
        return None
    if not is_lark_cli_command(command):
        return None
    envelope = _parse_envelope(stderr)
    if envelope is None:
        return None
    err = envelope.get("error")
    if not isinstance(err, dict):
        return None

    err_type = str(err.get("type", "")).strip()
    message = str(err.get("message", "")).strip()
    hint = str(err.get("hint", "")).strip()
    if not err_type:
        # An envelope with no ``error.type`` is malformed — let the
        # caller surface the raw stderr rather than guessing.
        return None

    # --- confirmation_required: high-risk-write gate (exit 10) -----------
    if err_type == "confirmation_required":
        risk = err.get("risk")
        action = ""
        if isinstance(risk, dict):
            action = str(risk.get("action", "")).strip()
        return _format_signal(
            CONFIRMATION_REQUIRED_PREFIX,
            error_type=err_type, message=message, hint=hint, action=action,
        )

    # --- auth_required: must trigger the lark-shared split-flow ----------
    type_lower = err_type.lower()
    if (
        err_type in _AUTH_TYPE_KEYWORDS
        or "auth" in type_lower
    ):
        return _format_signal(
            AUTH_REQUIRED_PREFIX,
            error_type=err_type, message=message, hint=hint,
        )

    # --- generic envelope error ------------------------------------------
    return _format_signal(
        LARK_ERROR_PREFIX,
        error_type=err_type, message=message, hint=hint,
    )


__all__ = [
    "AUTH_REQUIRED_PREFIX",
    "CONFIRMATION_REQUIRED_PREFIX",
    "LARK_ERROR_PREFIX",
    "classify_lark_error",
    "is_lark_cli_command",
]