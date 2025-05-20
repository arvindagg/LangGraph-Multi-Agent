from langchain_openai import ChatOpenAI
from tools.tools import tools # Import the tools to bind to agents

# Configuration for the Supervisor LLM
# This LLM is used by the supervisor agent for routing decisions.
# It does not need tool binding as its primary role is to select the next agent.
llm_supervisor = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",  # Base URL for the Ollama API
    openai_api_key="ollama",  # Placeholder key for local Ollama instances
    model_name="qwen2.5-coder:14b",  # Specify the Ollama model for supervisor tasks
    temperature=0  # Lower temperature for more deterministic routing decisions
)

# Configuration for a generic Agent LLM
# This LLM is used by various worker agents (e.g., Calculator, Stock News)
# and is bound to the defined tools, allowing it to perform tool calls.
llm_agent = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",  # Base URL for the Ollama API
    openai_api_key="ollama",  # Placeholder key for local Ollama instances
    model_name="qwen2.5-coder:14b",  # Specify the Ollama model for agent tasks (tool-calling capable)
    temperature=0  # Lower temperature for reliable tool calling and responses
).bind_tools(tools) # Bind the imported tools to this LLM instance