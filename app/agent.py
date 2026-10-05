from app.status import TaskStatus
import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.orchestrator import orchestrator_node
from app.agents.tester import tester_node,tester_tools

from app.tools import (
    create_directory,
    delete_file,
    list_files,
    move_file,
    read_file,
    run_command,
    search_files,
    write_file,
)

from app.state import AgentState


load_dotenv()


# ============================================================
# CODER MODEL
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is not set")


coder_llm = ChatOpenAI(
    model="meta/muse-glimmer-30b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


# ============================================================
# CODER TOOLS
# ============================================================

tools = [
    read_file,
    write_file,
    list_files,
    run_command,
    search_files,
    delete_file,
    create_directory,
    move_file,
]

llm_with_tools = coder_llm.bind_tools(tools)
tool_node = ToolNode(tools)
tester_tool_node = ToolNode(tester_tools)


# ============================================================
# CODER SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an autonomous coding agent.

Your job is to complete coding tasks using the tools available to you.

Follow this workflow:

1. Understand the implementation plan from the orchestrator.
2. Inspect the project structure when necessary.
3. Find relevant files using list_files or search_files.
4. Read relevant files before modifying them.
5. Make the required changes using write_file.
6. Run the code or appropriate tests using run_command.
7. Carefully inspect the command output and exit code.
8. If the code fails, diagnose the error, modify the code, and run it again.
9. Continue until the implementation is working.
10. Do not claim a task is complete without verification.

IMPORTANT:

Do not repeatedly inspect the same files or project structure.

Once you have enough information to implement the task,
start making the required changes.

Do not spend all iterations planning or inspecting.

If the task requires creating new files and they do not exist,
create them using write_file.

Prioritize completing the implementation over excessive investigation.

When the task is complete, DO NOT call any more tools.

Instead, provide a short final response explaining:
- what you changed
- how you verified it

Do not guess about file contents.
Use tools whenever necessary.
"""


# ============================================================
# CODER NODE
# ============================================================

def coder(state: AgentState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),

        SystemMessage(
            content=f"""
The orchestrator created the following implementation plan:

{state.get('plan', '')}

Your job is to execute this plan.

Do not create a new plan.
Follow the orchestrator's instructions.
"""
        ),
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    print("\n🤖 CODER RESPONSE:")
    print(response.content)

    if response.tool_calls:
        print("\n🔧 CODER TOOL CALLS:")
        print(response.tool_calls)

    iteration = state.get("iteration", 0) + 1

    if response.tool_calls:
        status = TaskStatus.CODING
    else:
        status = TaskStatus.CODER_DONE

    return {
        "messages": [response],
        "coder_output": response.content,
        "task_status": status,
        "iteration": iteration,
    }


# ============================================================
# CODER TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# TESTER TOOL NODE
# ============================================================

tester_tool_node = ToolNode(tester_tools)


# ============================================================
# CODER ROUTER
# ============================================================

def route_coder(state: AgentState):

    # 1. If coder has finished, send to tester
    if state.get("task_status") == TaskStatus.CODER_DONE:
        return "tester"

    # 2. If coder requested tools, execute them
    if state.get("messages"):
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

    # 3. Safety limit
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        print("\n⚠️ Maximum coder iterations reached.")
        return END

    return END
# ============================================================
# TESTER ROUTER
# ============================================================

def route_tester(state: AgentState):

    # Tester still needs to use tools
    if state.get("messages"):
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tester_tools"

    test_output = state.get("test_output", "")

    if "VERDICT: PASS" in test_output:
        return END

    if "VERDICT: FAIL" in test_output:
        return "orchestrator"

    return END

# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(AgentState)


graph.add_node(
    "orchestrator",
    orchestrator_node
)

graph.add_node(
    "coder",
    coder
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "tester",
    tester_node
)

graph.add_node(
    "tester_tools",
    tester_tool_node
)


# ============================================================
# ORCHESTRATOR → CODER
# ============================================================

graph.add_edge(
    START,
    "orchestrator"
)

graph.add_edge(
    "orchestrator",
    "coder"
)


# ============================================================
# CODER ROUTING
# ============================================================

graph.add_conditional_edges(
    "coder",
    route_coder,
    {
        "tools": "tools",
        "tester": "tester",
        END: END,
    },
)


# ============================================================
# CODER TOOLS → CODER
# ============================================================

graph.add_edge(
    "tools",
    "coder"
)


# ============================================================
# TESTER ROUTING
# ============================================================

graph.add_conditional_edges(
    "tester",
    route_tester,
    {
        "tester_tools": "tester_tools",
        "orchestrator": "orchestrator",
        END: END,
    },
)


# ============================================================
# TESTER TOOLS → TESTER
# ============================================================

graph.add_edge(
    "tester_tools",
    "tester"
)


# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = MemorySaver()


agent = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    task = input(
        "What should the coding agent build?\n> "
    )

    initial_state = {

        "messages": [
            HumanMessage(content=task)
        ],

        "task": task,

        "plan": "",

        "current_step": "",

        "coder_output": "",

        "test_output": "",

        "files_changed": [],

        "iteration": 0,

        "max_iterations": 10,

        "task_status": TaskStatus.STARTING,
    }


    config = {
        "configurable": {
            "thread_id": "conversation_1"
        }
    }


    max_loops = 60

    loop_count = 0

    inputs = initial_state


    while loop_count < max_loops:

        loop_count += 1

        for step in agent.stream(
            inputs,
            config=config
        ):

            for node_name, state_update in step.items():

                print(
                    f"\n✅ --- FINISHED RUNNING NODE: "
                    f"{node_name.upper()} ---"
                )


                if "plan" in state_update:

                    print(
                        "\n📋 PLAN:\n"
                        + str(state_update["plan"])
                    )


                if (
                    "messages" in state_update
                    and state_update["messages"]
                ):

                    content = (
                        state_update["messages"][-1].content
                    )

                    if content:

                        print(
                            "\n💬 MESSAGE:\n"
                            + str(content)
                        )


                if "test_output" in state_update:

                    if state_update["test_output"]:

                        print(
                            "\n🧪 TESTER OUTPUT:\n"
                            + str(
                                state_update["test_output"]
                            )
                        )


                if "task_status" in state_update:

                    print(
                        "\n📌 TASK STATUS:\n"
                        + str(
                            state_update["task_status"]
                        )
                    )


        # ====================================================
        # CHECK GRAPH STATE
        # ====================================================

        state = agent.get_state(config)


        if not state.next:

            print(
                "\n✅ Task Completed!"
            )

            break


        print(
            "\n🔍 NEXT NODE:",
            state.next
        )

        print(
            "🔍 TASK STATUS:",
            state.values.get(
                "task_status"
            )
        )


        # ====================================================
        # CODER TOOL INTERRUPT
        # ====================================================

        if state.next == ("tools",):

            last_message = (
                state.values["messages"][-1]
            )

            tool_calls = getattr(
                last_message,
                "tool_calls",
                []
            )


            dangerous_tools_requested = [
                tool["name"]
                for tool in tool_calls
                if tool["name"] == "delete_file"
            ]


            if dangerous_tools_requested:

                print(
                    "\n⚠️ The agent wants to run "
                    f"dangerous tool(s): "
                    f"{', '.join(dangerous_tools_requested)}"
                )


                user_input = input(
                    "Type 'y' to approve, "
                    "or anything else to quit: "
                )


                if user_input.lower().strip() != "y":

                    print(
                        "Execution cancelled by user."
                    )

                    break


            # Resume from interrupted tools node
            inputs = None

            continue


        # ====================================================
        # OTHER NODES
        # ====================================================

        else:

            inputs = None

            continue


    if loop_count >= max_loops:

        print(
            "\n⚠️ Stopped because maximum "
            "loop limit was reached."
        )