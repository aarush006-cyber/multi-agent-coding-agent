# 🤖 Multi-Agent Coding Agent

A long-running multi-agent software engineering system built with **LangGraph, LangChain, and NVIDIA NIM**.

The system takes a software development task from the user, plans the work, delegates implementation to a coding agent, executes tools inside the project environment, and uses a separate testing agent to review the implementation.

If the tester detects a failure, the workflow can route the task back to the coding agent for another iteration.

---

## 🧠 Architecture

```text
                    ┌─────────────────┐
                    │      User       │
                    │   Coding Task   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │   / Planner     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Coding Agent  │
                    │      Muse       │
                    └────────┬────────┘
                             │
                       Tool Calls
                             │
                             ▼
                    ┌─────────────────┐
                    │  Tool Execution │
                    │                 │
                    │ Read / Write    │
                    │ Search / Run    │
                    │ Delete / Move   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Tester      │
                    │   / Reviewer    │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                 PASS                 FAIL
                   │                   │
                   ▼                   ▼
                  END             Orchestrator
                                      │
                                      ▼
                                    Coder
✨ Features
Multi-agent software engineering workflow
Task planning and orchestration
Dedicated coding agent
Dedicated testing/review agent
Tool calling through LangChain
Long-running iterative workflow using LangGraph
File system interaction
Command execution
Project-wide file searching
Human-in-the-loop approval for potentially destructive operations
Iterative coder → tester → coder workflow
State persistence using LangGraph checkpointing
🧩 Agents
1. Orchestrator

The orchestrator is responsible for managing the overall software engineering task.

Its responsibilities include:

Understanding the user's task
Breaking the task into smaller steps
Deciding what the coding agent should work on
Reviewing progress
Analyzing testing results
Deciding whether more work is required

The orchestrator acts as the manager of the coding workflow rather than directly implementing the code.

2. Coding Agent

The coding agent is responsible for implementing the planned work.

The project currently uses:

Muse (meta/muse-glimmer-30b)

The coding agent can:

Inspect existing files
Create new files
Modify existing files
Search through the project
Run commands and tests
Debug implementation issues
Iterate based on test results

The coding agent interacts with the project through structured tools rather than directly manipulating the environment.

3. Tester / Reviewer

The tester is responsible for reviewing the implementation after the coding phase.

It can:

Inspect the generated implementation
Run or inspect tests
Analyze failures
Determine whether the task requirements were satisfied
Return a final verdict

The tester produces either:

VERDICT: PASS

or

VERDICT: FAIL

A failed result can send the workflow back to the orchestrator and coding agent for another iteration.

🛠️ Available Tools

The coding agent currently has access to tools for interacting with the project environment:

Tool	Purpose
read_file	Read file contents
write_file	Create or modify files
list_files	Inspect project files
search_files	Search through project files
run_command	Execute terminal commands
delete_file	Delete files
create_directory	Create directories
move_file	Move files

The tool interface allows the language model to interact with the software project through structured tool calls.

🔄 Workflow

The main workflow is implemented using LangGraph.

A simplified execution flow is:

START
  │
  ▼
Orchestrator
  │
  ▼
Coder
  │
  ├───────────────┐
  │               │
  │ Tool Calls    │ Coding Complete
  ▼               ▼
Tools           Tester
  │               │
  └───► Coder     │
                  │
            ┌─────┴─────┐
            │           │
          PASS         FAIL
            │           │
            ▼           ▼
           END     Orchestrator

This allows the system to perform multiple coding and testing iterations instead of stopping after a single LLM response.

🧠 State Management

The system uses a shared AgentState to maintain information throughout the workflow.

The state contains information such as:

Current task
Conversation messages
Plan
Current step
Coder output
Tester output
Files changed
Current iteration
Maximum iterations
Task status

LangGraph's message state handling allows messages from different stages of the workflow to be maintained throughout the execution.

🔐 Human-in-the-Loop

Potentially destructive operations can be paused before execution.

For example, file deletion can require human approval before the tool is executed.

This provides an additional safety layer when an autonomous coding agent interacts with the local development environment.

🧰 Tech Stack
Python
LangChain
LangGraph
LangChain OpenAI integration
NVIDIA NIM
Muse / GLM models
python-dotenv
📁 Project Structure
multi-agent-coding-agent/
│
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   └── tester.py
│   │
│   ├── agent.py
│   ├── state.py
│   ├── status.py
│   └── tools.py
│
├── .gitignore
├── requirements.txt
└── README.md
🚀 Setup

Clone the repository:

git clone https://github.com/aarush006-cyber/multi-agent-coding-agent.git
cd multi-agent-coding-agent

Install dependencies:

pip install -r requirements.txt

Create a .env file:

NVIDIA_API_KEY=your_nvidia_api_key

Never commit your .env file or expose your API key.

▶️ Running the Agent

Run the agent from the project root:

python -m app.agent

The agent will ask for a software development task.

For example:

What should the coding agent build?

> Build a Python calculator with tests

The orchestrator will then plan the task and delegate implementation to the coding agent.

🎯 Project Goal

The goal of this project is to explore how large language models can be combined with:

Agent orchestration
Tool calling
State management
Software engineering workflows
Automated testing
Iterative debugging
Human-in-the-loop safety

Instead of treating an LLM as a simple code generator, this project experiments with building an AI software engineering workflow that can plan, execute, test, and iterate.

🔮 Future Improvements

Planned improvements include:

Better automated evaluation of coding agents
More robust error recovery
Improved test generation
Better task planning
Persistent long-term project memory
More specialized coding agents
Stronger sandboxing for tool execution
Observability and tracing
More advanced multi-agent collaboration
👨‍💻 Author

Aarush