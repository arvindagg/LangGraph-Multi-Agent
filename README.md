# LangGraph Multi-Agent Framework

This repository provides a modular, LangGraph-based multi-agent framework designed for building sophisticated conversational AI applications. It leverages LangChain's capabilities for LLM integration and tool utilization, orchestrated by a central supervisor agent. The framework is designed to be easily extensible, allowing you to add new agents and tools to handle diverse user requests.

## Features

* **Modular Architecture**: Easily add or remove agents and tools without significantly refactoring the core framework.

* **Supervisor-Driven Routing**: A central LLM-powered supervisor intelligently routes user requests to the most appropriate worker agent.

* **Tool-Calling Agents**: Agents can dynamically decide to use predefined tools to perform specific actions (e.g., calculations, fetching real-world data).

* **Conversation History Management**: The system maintains and utilizes full conversation history for informed decision-making by both the supervisor and worker agents.

* **Ollama Integration**: Configured to work seamlessly with local Ollama LLMs, making it accessible for development and experimentation.

* **Extensible**: Clear patterns for adding new agents, tools, and modifying routing logic.

## Project Structure
Here is the complete README.md content, provided directly in this chat message. Please copy all the text below and paste it into a new file named README.md in the root of your project using a plain text editor (like VS Code, Sublime Text, or even Notepad on Windows).

Markdown

# LangGraph Multi-Agent Framework

This repository provides a modular, LangGraph-based multi-agent framework designed for building sophisticated conversational AI applications. It leverages LangChain's capabilities for LLM integration and tool utilization, orchestrated by a central supervisor agent. The framework is designed to be easily extensible, allowing you to add new agents and tools to handle diverse user requests.

## Features

* **Modular Architecture**: Easily add or remove agents and tools without significantly refactoring the core framework.

* **Supervisor-Driven Routing**: A central LLM-powered supervisor intelligently routes user requests to the most appropriate worker agent.

* **Tool-Calling Agents**: Agents can dynamically decide to use predefined tools to perform specific actions (e.g., calculations, fetching real-world data).

* **Conversation History Management**: The system maintains and utilizes full conversation history for informed decision-making by both the supervisor and worker agents.

* **Ollama Integration**: Configured to work seamlessly with local Ollama LLMs, making it accessible for development and experimentation.

* **Extensible**: Clear patterns for adding new agents, tools, and modifying routing logic.

## Project Structure

langgraph-multi-agent-framework/
├── agents/
│   ├── calculator_agent.py      # Agent for mathematical calculations
│   ├── data_analysis_agent.py   # Placeholder agent for data processing tasks
│   ├── stock_news_agent.py      # Agent for fetching and summarizing stock news
│   └── text_processing_agent.py # General-purpose agent for text queries
├── llms/
│   └── ollama_llms.py           # LLM configurations (Supervisor and Agent LLMs)
├── supervisor/
│   ├── prompts.py               # Prompts for the supervisor agent
│   └── supervisor_node.py       # Logic for the supervisor agent's routing decisions
├── tools/
│   └── tools.py                 # Definitions of all callable tools
├── main.py                      # Main application entry point; defines and runs the LangGraph
├── state.py                     # Defines the shared AgentState for the graph
└── README.md                    # This file

## Getting Started

Follow these steps to set up and run the multi-agent framework locally.

### Prerequisites

* Python 3.9+

* Poetry (recommended for dependency management) or pip

* Ollama installed and running with the specified models.

### 1. Install Ollama & Download Models

Ensure you have Ollama installed and running on your machine.
Download the `qwen2.5-coder:14b` model (or your preferred compatible model) which is used for both supervisor and agent LLMs in this setup:

```bash
ollama pull qwen2.5-coder:14b
# If you prefer a different model, update model_name in llms/ollama_llms.py accordingly.

Verify Ollama is running by visiting http://localhost:11434 in your browser. You should see a simple "Ollama is running" message.

2. Clone the Repository

git clone [https://github.com/your-username/langgraph-multi-agent-framework.git](https://github.com/your-username/langgraph-multi-agent-framework.git)
cd langgraph-multi-agent-framework
