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
Bash

git clone [https://github.com/your-username/langgraph-multi-agent-framework.git](https://github.com/your-username/langgraph-multi-agent-framework.git)
cd langgraph-multi-agent-framework
3. Set Up Python Environment and Install Dependencies
Using Poetry (Recommended):

Bash

poetry install
poetry shell
Using pip:

Bash

python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt # You'll need to create a requirements.txt first
If you don't have requirements.txt, you can generate it using pip freeze > requirements.txt after installing dependencies, or manually install them:

Bash

pip install langchain langchain-openai langgraph yfinance
4. Run the Application
Execute the main.py script:

Bash

python main.py
The system will start, and you will be prompted to enter your requests.

--- Script started ---
Welcome to the Multi-Agent System!
Type your request below, or type 'quit' to exit.

You: What is 15 plus 27?
---Running agent with input: 'What is 15 plus 27?'---
{'supervisor': {'next': 'calculator_agent'}}
{'calculator_agent': {'messages': [HumanMessage(content='What is 15 plus 27?'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, "operation": "add"}', 'name': 'perform_calculation'}}]}, tool_calls=[{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, "operation": "add"}', 'name': 'perform_calculation'}}]), ToolMessage(content='42.0', tool_call_id='call_YFhL...')]}
---Executing Calculator Agent---
---Calculator Agent received tool calls: [{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, 'operation': 'add'}', 'name': 'perform_calculation'}}]---
---Executing perform_calculation tool with 15.0 add 27.0---
---Tool 'perform_calculation' executed, result: 42.0---
---Tool messages added to state---
{'supervisor': {'next': 'END'}}
---Executing Supervisor Node---
---Supervisor decided next action: END---
---Agent execution finished---

--- Agent's Final Output ---
--- Debugging State with Messages Found: {'messages': [HumanMessage(content='What is 15 plus 27?'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, "operation": "add"}', 'name': 'perform_calculation'}}]}, tool_calls=[{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, 'operation': 'add'}', 'name': 'perform_calculation'}}]), ToolMessage(content='42.0', tool_call_id='call_YFhL...')]} ---
Tool Result: 42.0
----------------------------

You: Get me the latest news for MSFT.
---Running agent with input: 'Get me the latest news for MSFT.'---
{'supervisor': {'next': 'stock_news_agent'}}
---Executing Supervisor Node---
---Supervisor decided next action: stock_news_agent---
{'stock_news_agent': {'messages': [HumanMessage(content='Get me the latest news for MSFT.'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_V81h...', 'function': {'arguments': '{"ticker": "MSFT"}', 'name': 'get_stock_news'}}]}, tool_calls=[{'id': 'call_V81h...', 'function': {'arguments': '{"ticker": "MSFT"}', 'name': 'get_stock_news'}}])]}
---Executing Stock News Agent---
---Stock News Agent received tool calls: [{'id': 'call_V81h...', 'function': {'arguments': '{"ticker": "MSFT"}', 'name': 'get_stock_news'}}]---
---Executing get_stock_news tool for ticker: MSFT using yfinance---
---Tool 'get_stock_news' executed, result: [{'title': 'Microsoft’s New Copilot+ PCs Threaten Windows on Arm’s Last Chance', 'summary': 'Microsoft is facing an uphill battle getting Windows on Arm to take off, but its newest Copilot+ PCs might give it the shot in the arm it needs.'}, ...] --- (truncated for brevity)
---Tool messages added to state---
---Performing sentiment analysis on news data---
---Sentiment analysis performed and final response generated---
{'supervisor': {'next': 'END'}}
---Executing Supervisor Node---
---Supervisor decided next action: END---
---Agent execution finished---

--- Agent's Final Output ---
--- Debugging State with Messages Found: {'messages': [HumanMessage(content='Get me the latest news for MSFT.'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_V81h...', 'function': {'arguments': '{"ticker": "MSFT"}', 'name': 'get_stock_news'}}]}, tool_calls=[{'id': 'call_V81h...', 'function': {'arguments': '{"ticker": "MSFT"}', 'name': 'get_stock_news'}}]), ToolMessage(content="[{'title': 'Microsoft’s New Copilot+ PCs Threaten Windows on Arm’s Last Chance', 'summary': 'Microsoft is facing an uphill battle getting Windows on Arm to take off, but its newest Copilot+ PCs might give it the shot in the arm it needs.'}, {'title': 'Microsoft’s new AI PCs can now process 40 trillion operations per second', 'summary': 'These new AI PCs are coming out this June.'}, {'title': 'Microsoft Introduces Copilot+ PCs: What to Know', 'summary': 'Microsoft is introducing new Copilot+ PCs at its Redmond campus on May 20, a significant step into the AI PC category.'}, {'title': 'Microsoft (MSFT) Stock: Should You Buy, Sell, or Hold Now?', 'summary': 'Microsoft (MSFT) stock has shown a strong performance recently, making it a topic of interest for many investors.'}, {'title': 'Why Is Everyone Talking About Microsoft Stock?', 'summary': 'Microsoft, a leading software company, continues to gain investor attention due to its strong financial performance and focus on AI technology.'}]", tool_call_id='call_V81h...'), AIMessage(content='Here is the latest news for MSFT:\n- Microsoft’s New Copilot+ PCs Threaten Windows on Arm’s Last Chance\n- Microsoft’s new AI PCs can now process 40 trillion operations per second\n- Microsoft Introduces Copilot+ PCs: What to Know\n- Microsoft (MSFT) Stock: Should You Buy, Sell, or Hold Now?\n- Why Is Everyone Talking About Microsoft Stock?\n\nOverall Sentiment & Themes:\nThe news for Microsoft appears to be generally positive, with a strong focus on their new "Copilot+ PCs" and advancements in AI technology. Key themes include the launch of new AI-powered personal computers, their processing capabilities, and ongoing investor interest in Microsoft stock due to its financial financial and AI focus.', response_metadata={'model_name': 'qwen2.5-coder:14b', 'token_usage': {'completion_tokens': 161, 'prompt_tokens': 604, 'total_tokens': 765}, 'finish_reason': 'stop'}, id='run-50a...-0', usage_metadata={'input_tokens': 604, 'output_tokens': 161})]} ---
AI Response: Here is the latest news for MSFT:
- Microsoft’s New Copilot+ PCs Threaten Windows on Arm’s Last Chance
- Microsoft’s new AI PCs can now process 40 trillion operations per second
- Microsoft Introduces Copilot+ PCs: What to Know
- Microsoft (MSFT) Stock: Should You Buy, Sell, or Hold Now?
- Why Is Everyone Talking About Microsoft Stock?

Overall Sentiment & Themes:
The news for Microsoft appears to be generally positive, with a strong focus on their new "Copilot+ PCs" and advancements in AI technology. Key themes include the launch of new AI-powered personal computers, their processing capabilities, and ongoing investor interest in Microsoft stock due to its financial financial and AI focus.
----------------------------

You: quit
Exiting Multi-Agent System. Goodbye!
How to Extend the Framework
This framework is designed for easy expansion. Here's how you can add new capabilities:

1. Create a New Tool
If your new agent needs to interact with external systems or perform specific, deterministic actions, create a new tool.

Create/Modify tools/tools.py:

Define a Python function with the @tool decorator.

Provide a clear docstring explaining its purpose, arguments, and return values. This docstring is crucial for the LLM to understand when and how to use the tool.

Add the new function to the tools list at the end of the file.

Example tools/tools.py addition:

Python

# ... existing tools ...

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Fetches the current weather for a specified city.
    Args:
        city (str): The name of the city.
        unit (str, optional): The unit for temperature ('celsius' or 'fahrenheit'). Defaults to 'celsius'.
    Returns:
        str: A description of the current weather, e.g., "The weather in London is 15°C and sunny."
    """
    print(f"---Executing get_weather tool for {city} in {unit}---")
    # In a real scenario, you'd call a weather API here
    if city.lower() == "london":
        return f"The weather in London is 15°C and partly cloudy." if unit == "celsius" else "The weather in London is 59°F and partly cloudy."
    return "Weather data not available for this city."

tools = [perform_calculation, get_stock_news, get_weather] # Add your new tool here
2. Create a New Agent
An agent encapsulates specific logic, potentially using an LLM and/or tools.

Create a new file in agents/: E.g., agents/new_agent.py.

Define the agent function: It must accept state: AgentState and return AgentState (or dict compatible with AgentState).

Implement agent logic:

Simple Agent: If it doesn't need an LLM or tools, just process the messages in state and add new AIMessage or HumanMessage to the list.

LLM-Powered Agent: Import llm_agent from llms.ollama_llms. Use llm_agent.invoke(messages) to get an LLM response.

Tool-Calling Agent: Follow the pattern in calculator_agent.py or stock_news_agent.py. The LLM will suggest tool calls, which you then manually execute and add the ToolMessage results back to the state. Remember to import the specific tool(s) if you're executing them directly.

Example agents/weather_agent.py:

Python

from state import AgentState
from llms.ollama_llms import llm_agent
from tools.tools import get_weather # Import the specific tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def weather_agent(state: AgentState) -> AgentState:
    print("---Executing Weather Agent---")
    messages = state['messages']
    response = llm_agent.invoke(messages) # LLM decides to call get_weather
    messages.append(response)

    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call['name'] == "get_weather":
                try:
                    result = get_weather.invoke(tool_call['args']) # Execute the tool
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                    # Optionally, invoke LLM again to summarize result for user
                    final_response = llm_agent.invoke(messages)
                    messages.append(final_response)
                except Exception as e:
                    messages.append(ToolMessage(content=f"Error getting weather: {e}", tool_call_id=tool_call['id']))
    return {"messages": messages}
3. Update the Supervisor Prompt
The supervisor LLM needs to know about your new agent to route requests to it.

Modify supervisor/prompts.py:

Update the SUPERVISOR_PROMPT to include your new agent in the "Available agents" list.

Provide a concise description of when to route to this new agent.

Example supervisor/prompts.py update:

Python

# ... existing prompt ...
Available agents:
# ... existing agents ...
- 'weather_agent': Use this for requests asking about current weather conditions for a city.
# ...
4. Integrate the New Agent into main.py
Import the new agent function:

Python

from agents.weather_agent import weather_agent
Add a new node to the workflow:

Python

workflow.add_node("weather_agent", weather_agent)
Add a conditional edge from the supervisor: Map the supervisor's output to your new agent's node name.

Python

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state['next'],
    {
        # ... existing mappings ...
        "weather_agent": "weather_agent", # New mapping
        "END": END,
    }
)
Add an edge from the new agent back to the supervisor:

Python

workflow.add_edge("weather_agent", "supervisor") # New edge
5. Test Your New Agent
Run python main.py and test with a query relevant to your new agent (e.g., "What's the weather in London?"). Observe the console output to ensure the supervisor routes correctly and your agent executes as expected.

Contributing
Feel free to fork this repository, open issues, and submit pull requests.

Acknowledgements
This project was developed with the assistance of an AI large language model.

License
MIT License

Copyright (c) 2025 Arvind Aggarwal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.