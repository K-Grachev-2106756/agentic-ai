from langchain.agents.middleware import (
    dynamic_prompt, 
    AgentMiddleware, 
    ModelRequest,
)


def dynamic_prompt_factory(
    base_prompt: str, 
    on_match_prompt: str, 
    on_match_context_field: str,
    on_match_value: str,
) -> AgentMiddleware:

    @dynamic_prompt
    def prompt_changer(request: ModelRequest) -> str:
        field_value = getattr(request.runtime.context, on_match_context_field)
        if field_value == on_match_value:
            return on_match_prompt
        
        return base_prompt

    return prompt_changer
