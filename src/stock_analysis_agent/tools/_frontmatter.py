"""Shared YAML frontmatter parser used by tools.skill and tools.strategy.

Two callers with different needs:
- ``tools/skill.py`` parses skill SKILL.md frontmatter; needs ``name``
  and ``description`` keys.
- ``tools/strategy.py`` parses strategy .md frontmatter; needs
  ``name``, ``version``, ``description`` keys (others filtered).

Both share the same parsing rules: top-level ``key: value`` pairs,
optional ``description: |`` multi-line literal blocks (joined with
spaces), and surrounding double-quote stripping on single-line
values.
"""
from __future__ import annotations


def parse_yaml_frontmatter(
    text: str,
    *,
    allow: frozenset[str] | None = None,
) -> dict[str, str]:
    """Extract simple ``key: value`` pairs from a YAML frontmatter block.

    Args:
        text: Full Markdown text starting with the ``---`` fence.
        allow: If given, only keys in this set are returned. If
            ``None``, all parsed keys are returned.

    Returns:
        Dict of frontmatter keys. Missing/unparseable input yields
        ``{}``. The caller is responsible for pre-seeding defaults
        if it needs ``{"name": "", "description": ""}``-style empty
        values.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = 1
    while end < len(lines) and lines[end].strip() != "---":
        end += 1
    fm_lines = lines[1:end]
    result: dict[str, str] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if ":" not in line or line.startswith((" ", "\t")):
            i += 1
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip()
        if rest == "|":
            i += 1
            block: list[str] = []
            while i < len(fm_lines):
                nxt = fm_lines[i]
                if nxt.startswith((" ", "\t")) and nxt.strip():
                    block.append(nxt.strip())
                    i += 1
                else:
                    break
            result[key.strip()] = " ".join(block)
            continue
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] == '"':
            rest = rest[1:-1]
        result[key.strip()] = rest
        i += 1
    if allow is None:
        return result
    return {k: v for k, v in result.items() if k in allow}


__all__ = ["parse_yaml_frontmatter"]
