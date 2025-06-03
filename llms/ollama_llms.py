# llms/ollama_llms.py
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from tools.tools import tools

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = "qwen2.5-coder:14b" # Or your preferred Ollama model for general agents
OLLAMA_SUPERVISOR_MODEL_NAME = "codellama:34b" # Supervisor model (often a larger, more reasoning-capable model)
OLLAMA_CODE_GEN_MODEL_NAME = "qwen2.5-coder:14b" # NEW: Dedicated model for code generation. Can be different.

# LLM for agents that might call tools (e.g., calculator, web search, stock news)
llm_agent_with_tools = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    format="json", # Important for tool calling with Ollama
    temperature=0 # Added temperature for more deterministic behavior
).bind_tools(tools=tools) # This should now work correctly

# LLM for the supervisor (typically doesn't call tools, just routes)
llm_supervisor = ChatOllama(
    model=OLLAMA_SUPERVISOR_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

# NEW: LLM specifically for code generation.
# IMPORTANT: No format="json" here, as we want raw code output.
llm_code_generator = ChatOllama(
    model=OLLAMA_CODE_GEN_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2 # A slightly higher temperature might encourage creativity in code, but keep it low for reliability
)
