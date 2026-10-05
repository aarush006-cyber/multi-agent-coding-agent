from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # Original user request
    task: str

    # Conversation / tool messages
    messages: Annotated[list, add_messages]

    # Planning
    plan: str
    current_step: str

    # Multi-agent outputs
    coder_output: str
    test_output: str

    # Files modified by coder
    files_changed: list[str]

    # Long-running task control
    iteration: int
    max_iterations: int

    # Overall task state
    task_status: str