import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from app.state import AgentState
from app.status import TaskStatus

from app.tools import (
    list_files,
    read_file,
    run_command,
    search_files,
)


load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is not set")


tester = ChatOpenAI(
    model="z-ai/glm-5.3-flash",
    api_key=API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0,
)


tester_tools = [
    list_files,
    read_file,
    run_command,
    search_files,
]

tester_with_tools = tester.bind_tools(tester_tools)


TESTER_PROMPT = """
You are the testing and review agent of a long-running coding agent.

Your job is to independently verify whether the coder actually completed
the user's task correctly.

You are NOT the coder.

You are NOT the orchestrator.

Do not blindly trust the coder's output.

You MUST inspect the actual project using your tools.

Your workflow:

1. Understand the original user task.
2. Inspect the project structure.
3. Find the files relevant to the task.
4. Read the important implementation files.
5. Read the relevant tests.
6. Run the appropriate test suite using run_command.
7. Carefully inspect the test output.
8. If tests fail, determine the actual cause.
9. Verify that the implementation satisfies the original requirements.
10. Only then produce a verdict.

IMPORTANT:

Do not declare PASS merely because the coder says the task is complete.

PASS requires actual evidence from the project and tests.

Your final response MUST contain exactly one of:

VERDICT: PASS

or

VERDICT: FAIL

If FAIL:
- Explain what is wrong.
- Identify the file/function involved.
- Explain what the coder needs to fix.

If PASS:
- Explain what was verified.
- Mention the relevant test results.

Do not modify project files.
You are a reviewer/tester only.
"""


def tester_node(state: AgentState):

    task = state.get("task", "")
    coder_output = state.get("coder_output", "")

    messages = [
        SystemMessage(content=TESTER_PROMPT),
        HumanMessage(
            content=f"""
ORIGINAL USER TASK:

{task}


CODER'S LAST OUTPUT:

{coder_output}


Now independently inspect the project and verify the implementation.
Do not trust the coder's claims without checking the actual files and
running the appropriate tests.
"""
        ),
    ]

    response = tester_with_tools.invoke(messages)

    if response.tool_calls:
        status = TaskStatus.TESTING
    else:
        status = TaskStatus.TEST_DONE

    return {
        "messages": [response],
        "test_output": response.content,
        "task_status": status,
    }