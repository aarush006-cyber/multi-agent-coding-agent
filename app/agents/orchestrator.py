import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.state import AgentState
from app.status import TaskStatus


load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is not set")


orchestrator = ChatOpenAI(
    model="z-ai/glm-5.3-flash",
    api_key=API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0,
)


ORCHESTRATOR_PROMPT = """
You are the orchestrator of a long-running software engineering agent.

Your job is to analyze the user's software engineering task and create
a clear implementation plan for the coding agent.

You are NOT the coder.

Responsibilities:

1. Understand the user's task.
2. Break the task into logical implementation steps.
3. Identify the files/components that need to be created or modified.
4. Identify important requirements and edge cases.
5. Tell the coder what needs to be implemented.
6. Keep the plan concrete and actionable.

Do not write the implementation code yourself.

Do not claim that the task is complete.

Your output should be a concise implementation plan.
"""


def orchestrator_node(state: AgentState):

    task = state.get("task", "")

    if not task:
        raise ValueError("No task provided to orchestrator.")

    prompt = f"""
{ORCHESTRATOR_PROMPT}

USER TASK:
{task}
"""

    response = orchestrator.invoke(prompt)

    return {
        "plan": response.content,
        "current_step": "coding",
        "task_status": TaskStatus.CODING,
        "iteration": state.get("iteration", 0),
    }