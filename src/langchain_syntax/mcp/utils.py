from typing import Any

from langchain_core.tools import StructuredTool


def _sanitize_tool_output(raw: Any) -> Any:
    """Чистит ответ инструмента от мета-информации."""
    if isinstance(raw, list):
        return [_sanitize_tool_output(r) for r in raw]
    elif isinstance(raw, dict):
        required_fields = {"type", "text", "content", "url"} | {raw.get("type", "text")}
        keys = set(raw.keys())
        for k in keys - required_fields:
            raw.pop(k)
        return raw
    else:
        return raw


def sanitize_mcp_tool(original_tool: StructuredTool) -> StructuredTool:
    """Оборачивает MCP-инструмент так, чтобы Mistral ToolMessage был корректным."""

    async def wrapped_func(**kwargs) -> str:
        raw_result = await original_tool.ainvoke(kwargs)
        return _sanitize_tool_output(raw_result)

    return StructuredTool.from_function(
        coroutine=wrapped_func,
        name=original_tool.name,
        description=original_tool.description,
        args_schema=original_tool.args_schema,
    )
