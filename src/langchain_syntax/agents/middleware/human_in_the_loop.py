from typing import Dict
import json

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langchain_core.messages.base import BaseMessage
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.tools import BaseTool


def hitl_factory(
        tool_interrupt_mapping: Dict[str, bool],
        interrupt_description: str = "Tool execution requires approval",
    ) -> HumanInTheLoopMiddleware:
    """
    Creates [human in the loop middleware].
    
    :param tool_interrupt_mapping: {tool_name: true/false}
    :param interrupt_description: the message that the agent will see in the event of an interruption
    """
    if any(tool_interrupt_mapping.values()):
        return HumanInTheLoopMiddleware(
            interrupt_on=tool_interrupt_mapping,
            description_prefix=interrupt_description,
        )

    raise ValueError("No instructions for interruption were passed")


def human_response_interpretation(llm: BaseChatModel, human_msg: str, msg_history: list[BaseMessage]):
    interrupt_meta = msg_history['__interrupt__'][-1].value["action_requests"][-1]

    role_prompt = "You are a human-in-the-loop controller."

    tool_info = f"""The execution of the tool was interrupted. Interruption meta:
    {interrupt_meta}"""

    instructions = """Based on the user's message, return ONE of the following:
- approve
- reject
- edit

Rules:
- If the user agrees: approve
- If the user refuses: reject
- If the user suggests changes: edit

If your answer is "approve" or "reject", output format - json:
{{"answer": str}}

If your answer is "edit", add to json arguments for calling tool:
{{"answer": str, "args": dict}}

Return only the correct json. No explanations."""

    prompt = [
        SystemMessage(content="\n".join([role_prompt, tool_info, instructions])),
        HumanMessage(content=human_msg),
    ]

    decision = {"answer": "reject"}
    for _ in range(3):  # retry
        response = llm.invoke(prompt)
        
        try:
            decision = json.loads(response.content)
        except:
            continue
        break
    
    match decision["answer"]:
        case "edit":
            return Command(        
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": interrupt_meta["name"],
                                "args": decision["args"],
                            }
                        }
                    ]
                }
            )
        case "approve":
            return Command(resume={"decisions": [{"type": "approve"}]})
        case "reject":
            return Command(        
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": human_msg,
                        }
                    ]
                }
            )
    