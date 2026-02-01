from typing import Callable, Literal, Any

from langchain.agents.middleware import ModelRequest


def request_reader(
    field_storage: Literal["state", "context"], 
    on_match_field: str,
) -> Callable[[ModelRequest], Any]:
    match field_storage:
        case "state":
            return lambda request: request.state.get(on_match_field)
        case "context":
            return lambda request: getattr(request.runtime.context, on_match_field)
        case _:
            return lambda request: getattr(getattr(request, field_storage), on_match_field)
