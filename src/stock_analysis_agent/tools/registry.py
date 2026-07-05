"""Tool registry — the catalog of self-built @tool callables.

Mirrors :mod:`stock_analysis_agent.tools.skill`, which catalogs skill
names from SKILL.md frontmatter. Here we catalog the self-built
``@tool`` callables so the system prompt can carry a one-line summary
per tool (name + description + inputs + output) without paying the
token cost of each tool's full docstring.

Two-tier loading model:

1. The **name + description + inputs + output** of every bundled tool
   is rendered into the system prompt (see :func:`get_tool_index` and
   :func:`format_tool_index_markdown`). This gives the model a
   structured catalog of what each tool does, the parameters it
   accepts, and the shape of its return value.
2. Once the model decides it needs a specific tool's full procedure,
   it invokes the tool itself — LangChain's @tool schema exposes the
   full Field descriptions (set via ``args_schema=...``) on the
   ``args`` attribute.

Adding a new tool requires:

1. Decorate the function with ``@tool("name", args_schema=InputModel)``.
2. Add the @tool object to :func:`list_tools`.
3. Add a ``ToolOutputSpec`` entry under :data:`_TOOL_OUTPUTS` keyed by
   the tool name.

The input columns are introspected from the @tool's ``args_schema``
Pydantic model (via ``model_json_schema()``), so adding a new field
to the Pydantic model automatically extends the catalog — no
second-place edit needed.
"""
from __future__ import annotations

from typing import TypedDict, cast

from langchain_core.tools import BaseTool

from stock_analysis_agent.tools.market_data import _get_stock_snapshot
from stock_analysis_agent.tools.read_file import read_file
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import load_skill
from stock_analysis_agent.tools.web_search import _web_search


class ToolParamSpec(TypedDict):
    """One row in a tool's input table.

    Attributes:
        name: Parameter name (matches the field name on the
            ``args_schema`` Pydantic model).
        type: Human-readable type signature (e.g. ``"str"``,
            ``"list[str]"``, ``"int | None"``).
        required: ``True`` if the field is required, ``False`` if it
            has a default value.
        description: Short description of what the parameter means.
    """

    name: str
    type: str
    required: bool
    description: str


class ToolIndexEntry(TypedDict):
    """One tool catalog row.

    Attributes:
        name: Tool name as exposed to the LLM (matches ``@tool("...")``).
        description: One-line purpose statement (mirrors the
            ``@tool(description=...)`` text).
        inputs: Per-parameter table; the registry introspects each
            tool's ``args_schema`` to build this list.
        output: Human-readable description of the return value shape,
            plus any documented raises. Pulled from
            :data:`_TOOL_OUTPUTS` (hand-curated, since the return
            type isn't carried on the @tool object).
    """

    name: str
    description: str
    inputs: list[ToolParamSpec]
    output: str


class ToolOutputSpec(TypedDict):
    """Hand-curated return-shape metadata for a tool.

    The ``@tool`` decorator does not expose a return-type schema
    (LangChain picks the return type from the function annotation,
    but a string description isn't available). These specs live next
    to the tool in code review so updates to behaviour land in the
    same diff.
    """

    output: str


#: Per-tool return-shape metadata. Keyed by the @tool name (matches
#: ``@tool("...")`` and ``Tool.name``).
_TOOL_OUTPUTS: dict[str, ToolOutputSpec] = {
    "get_stock_snapshot": {
        "output": (
            "`dict[str, Any]` — nested dict with: top-level `<symbol>` → "
            "per-source `{data, row_index}` or `{error: {type, message}}` "
            "blocks; `fetched_at` (ISO 8601 in Asia/Shanghai); `peers` "
            "(when `include_peers=True`) → dict keyed by peer symbol. "
            "LangChain serializes this to JSON before reaching the LLM. "
            "Raises `ValueError` on unknown market suffix; raises "
            "`ToolExecutionError` when every primary source errored."
        ),
    },
    "web_search": {
        "output": (
            "`str` — plain-text concatenation of `[<site>]\\n<text>` "
            "blocks separated by blank lines. Per-site failures surface "
            "as `[error: <ExceptionClass>: <msg>]` segments; CAPTCHA "
            "responses as `[error: captcha page returned]`. Raises "
            "`ToolExecutionError` when every site failed (caught by "
            "the retry middleware)."
        ),
    },
    "load_skill": {
        "output": (
            "`str` — full Markdown content of the skill's `SKILL.md` "
            "file. The LLM should follow these instructions to produce "
            "the formatted output for the user. Raises "
            "`FileNotFoundError` for unknown skill names — the error "
            "message lists the available skills."
        ),
    },
    "read_file": {
        "output": (
            "`str` — UTF-8 text content of the file. Binary inputs "
            "may raise `UnicodeDecodeError`. Raises `ValueError` on "
            "empty path or path-traversal; `IsADirectoryError` when "
            "the path is a directory; `FileNotFoundError` when the "
            "file does not exist."
        ),
    },
    "run_command": {
        "output": (
            "`str` — formatted text block of the form::\n\n"
            "    $ <command> <args...>\n"
            "    cwd: <cwd>\n"
            "    === exit=<N> ===          (or `=== TIMEOUT (after <N>s) ===`)\n"
            "    --- stdout ---\n"
            "    <truncated stdout — capped at 30 KB>\n"
            "    --- stderr ---\n"
            "    <truncated stderr — capped at 30 KB>\n\n"
            "Raises `ValueError` on empty `command` or when `command` "
            "contains whitespace (LLM accidentally pasted a shell "
            "command instead of splitting into `command` + `argv`); "
            "`TypeError` when `argv` isn't a list of strings; "
            "`FileNotFoundError` when `command` is not on `PATH`."
        ),
    },
}


#: Parameter type names that ``Pydantic v2`` emits in JSON-schema
#: ``"type"`` fields for the primitive cases we use. Reference for
#: the introspection below.
_PYDANTIC_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _render_type_fragment(part: dict[str, object]) -> str:
    """Render one ``anyOf`` part (or a bare field schema) as a type string.

    Args:
        part: One branch of ``anyOf``, or a top-level field schema.

    Returns:
        A short type signature such as ``"str"`` or ``"list[str]"``.
    """
    if part.get("type") == "array":
        items = part.get("items")
        if isinstance(items, dict):
            inner = _PYDANTIC_TYPE_MAP.get(
                str(items.get("type")), str(items.get("type"))
            )
            return f"list[{inner}]"
        return "list"
    if "enum" in part:
        values = cast(list[object], part.get("enum") or [])
        return "Literal[" + ", ".join(repr(v) for v in values) + "]"
    type_name = str(part.get("type", "Any"))
    return _PYDANTIC_TYPE_MAP.get(type_name, type_name)


def _human_type(field_schema: dict[str, object]) -> str:
    """Convert a JSON-schema fragment into a short Python-style signature.

    Handles the three shapes Pydantic emits:

    * ``{"type": "string"}`` → ``"str"``
    * ``{"type": "array", "items": {"type": "string"}}`` → ``"list[str]"``
    * ``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` → ``"str | None"``
    * ``{"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}``
      → ``"list[str] | None"``

    Args:
        field_schema: One field's JSON-schema fragment.

    Returns:
        A short type signature, e.g. ``"str"``, ``"list[str]"``,
        ``"str | None"``.
    """
    if "anyOf" in field_schema:
        parts = cast(list[object], field_schema.get("anyOf") or [])
        rendered: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "null":
                rendered.append("None")
                continue
            if isinstance(part, dict):
                rendered.append(_render_type_fragment(part))
        return " | ".join(rendered) if rendered else "Any"
    return _render_type_fragment(field_schema)


def _extract_inputs(tool_obj: BaseTool) -> list[ToolParamSpec]:
    """Introspect ``tool_obj``'s ``args_schema`` and return one row per field.

    LangChain exposes ``tool.args`` in one of two shapes:

    1. **Pydantic model** (LangChain v1+) — has ``model_json_schema()``
       returning a nested ``{"properties": {...}, "required": [...]}``.
    2. **Flat dict** (current LangChain) — top-level keys are parameter
       names; ``"required"`` is implied by the **absence** of a
       ``"default"`` field.

    This helper accepts both shapes.

    Args:
        tool_obj: The ``@tool``-wrapped callable.

    Returns:
        One :class:`ToolParamSpec` per input parameter, in declaration
        order. Fields with ``"default"`` set are reported as
        ``required=False``; the rest are required.
    """
    raw_schema: object = tool_obj.args
    properties: dict[str, object]
    required: set[str]

    json_schema_callable = getattr(raw_schema, "model_json_schema", None)
    if callable(json_schema_callable):
        schema = cast(dict[str, object], json_schema_callable())
        properties = cast(dict[str, object], schema.get("properties", {}))
        required_list = schema.get("required", []) or []
        required = set(cast(list[str], required_list))
    elif isinstance(raw_schema, dict):
        # Flat LangChain schema: top-level keys ARE the parameter names.
        # Filter out bookkeeping keys like ``"title"`` / ``"description"``.
        bookkeeping = {"title", "description"}
        raw_dict = cast(dict[str, object], raw_schema)
        properties = {
            k: v for k, v in raw_dict.items()
            if k not in bookkeeping and isinstance(v, dict)
        }
        required = set()
    else:
        return []

    rows: list[ToolParamSpec] = []
    for name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        # Flat schemas lack ``required`` — derive from ``default``.
        has_default = "default" in field_schema
        field_required = (name in required) if required else not has_default
        rows.append(
            ToolParamSpec(
                name=name,
                type=_human_type(field_schema),
                required=bool(field_required),
                description=str(field_schema.get("description", "")).strip(),
            )
        )
    return rows


def _output_for(name: str) -> str:
    """Return the hand-curated output spec for tool ``name``.

    Tools missing from :data:`_TOOL_OUTPUTS` get a fallback placeholder
    so the catalog still renders — operators should add the entry.
    """
    spec = _TOOL_OUTPUTS.get(name)
    if spec is None:
        return (
            "_(no output description registered — add a `_TOOL_OUTPUTS` "
            f"entry for `{name}` in `tools/registry.py`)_"
        )
    return spec["output"]


def list_tools() -> list[BaseTool]:
    """Return every self-built ``@tool`` exposed to the LLM catalog.

    The list is ordered alphabetically by tool name for stable
    rendering. Adding a new tool requires appending here **and** to
    :data:`_TOOL_OUTPUTS`.
    """
    all_tools: list[BaseTool] = [
        _get_stock_snapshot,
        _web_search,
        load_skill,
        read_file,
        run_command,
    ]
    return sorted(all_tools, key=lambda t: t.name)


def get_tool_index() -> list[ToolIndexEntry]:
    """Build the tool catalog injected into the system prompt.

    For each tool returned by :func:`list_tools`, introspect the
    ``args_schema`` (Pydantic model) for input rows and look up the
    hand-curated :class:`ToolOutputSpec` for the return shape.

    Returns:
        One :class:`ToolIndexEntry` per registered tool, in the same
        alphabetical order as :func:`list_tools`.
    """
    index: list[ToolIndexEntry] = []
    for tool_obj in list_tools():
        description = tool_obj.description or ""
        index.append(
            ToolIndexEntry(
                name=tool_obj.name,
                description=description.strip(),
                inputs=_extract_inputs(tool_obj),
                output=_output_for(tool_obj.name),
            )
        )
    return index


def format_tool_index_markdown(index: list[ToolIndexEntry]) -> str:
    """Render ``index`` as a Markdown document, one section per tool.

    Each tool section uses three blocks:

    1. A ``### `<name>` `` heading.
    2. A **purpose** paragraph carrying the one-line description.
    3. An **inputs** table (parameter name + type + required + description).
    4. An **output** paragraph carrying the hand-curated return spec.

    Long output specs with embedded code fences are passed through
    verbatim so the LLM can read them as-is.

    Args:
        index: Catalog from :func:`get_tool_index`.

    Returns:
        Markdown document. The placeholder ``<!-- TOOL_INDEX -->`` in
        the system prompt template is replaced by this string at load
        time.
    """
    if not index:
        return "_(no tools registered)_\n"

    blocks: list[str] = []
    for entry in index:
        header = f"### `{entry['name']}`"
        purpose = f"**purpose**: {entry['description']}"

        if entry["inputs"]:
            input_header = "| name | type | required | description |\n|------|------|----------|-------------|"
            input_rows = [
                f"| `{p['name']}` | `{p['type']}` | "
                f"{'yes' if p['required'] else 'no'} | {p['description']} |"
                for p in entry["inputs"]
            ]
            inputs_block = "**inputs**:\n\n" + input_header + "\n" + "\n".join(input_rows)
        else:
            inputs_block = "**inputs**: _(none)_"

        output_block = f"**output**: {entry['output']}"

        blocks.append("\n\n".join([header, purpose, inputs_block, output_block]))

    return "\n\n".join(blocks) + "\n"


__all__ = [
    "ToolIndexEntry",
    "ToolOutputSpec",
    "ToolParamSpec",
    "_TOOL_OUTPUTS",
    "format_tool_index_markdown",
    "get_tool_index",
    "list_tools",
]