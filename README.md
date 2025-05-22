# LangGraph Multi-Agent Framework

This repository presents a modular, LangGraph-based multi-agent framework engineered for the development of sophisticated conversational AI applications. It leverages the capabilities of LangChain for seamless integration with Large Language Models (LLMs) and facilitates the utilization of specialized tools, all orchestrated by a central supervisor agent. The framework's design prioritizes extensibility, thereby enabling the straightforward addition of new agents and tools to address a diverse range of user requests.

## Key Features

-   **Modular Architecture**: The framework's design permits the facile incorporation or removal of agents and tools, minimizing the need for extensive refactoring of the core system.
    
-   **Supervisor-Driven Routing**: A central supervisor agent, powered by an LLM, intelligently directs user inquiries to the most appropriate worker agent, optimizing task allocation.
    
-   **Tool-Calling Agents**: Individual agents possess the capacity to dynamically invoke predefined tools, thereby executing specific, deterministic actions such as performing calculations or retrieving real-time data from external sources.
    
-   **Conversation History Management**: The system meticulously maintains and leverages comprehensive conversation history, which informs the decision-making processes of both the supervisor and the various worker agents.
    
-   **Ollama Integration**: The framework is configured for seamless operation with local Ollama LLMs, enhancing accessibility for development and experimental endeavors.
    
-   **Extensibility**: The architecture incorporates clear and well-defined patterns for the integration of new agents, the definition of novel tools, and the modification of routing logic.
    

## Project Structure

```
langgraph-multi-agent-framework/
├── agents/
│   ├── calculator_agent.py      # Agent responsible for mathematical computations
│   ├── data_analysis_agent.py   # Placeholder agent for data processing and analytical tasks
│   ├── stock_news_agent.py      # Agent dedicated to fetching and summarizing financial news for stock tickers
│   └── text_processing_agent.py # General-purpose agent for handling textual queries and basic information processing
├── llms/
│   └── ollama_llms.py           # Configuration files for Large Language Models (Supervisor and Agent LLMs)
├── supervisor/
│   ├── prompts.py               # Prompt definitions utilized by the supervisor agent
│   └── supervisor_node.py       # Implementation of the supervisor agent's routing decision logic
├── tools/
│   └── tools.py                 # Centralized definitions of all callable utility functions
├── main.py                      # The primary application entry point; responsible for defining and executing the LangGraph workflow
├── state.py                     # Defines the shared AgentState, which represents the system's state across agents
└── README.md                    # This documentation file


```

## Getting Started

To initiate and operate the multi-agent framework within a local environment, please adhere to the following procedural steps.

### Prerequisites

-   Python version 3.9 or higher is required.
    
-   Poetry (recommended for robust dependency management) or pip.
    
-   Ollama must be installed and actively running, with the specified LLM models downloaded.
    

### 1. Ollama Installation and Model Acquisition

Ensure that Ollama is installed and operational on your machine. Subsequently, download the `qwen2.5-coder:14b` model (or an alternative compatible model of your preference), which is employed for both the supervisor and worker agent LLMs in this configuration:

```
ollama pull qwen2.5-coder:14b
# Should an alternative model be preferred, please update the 'model_name' parameter within 'llms/ollama_llms.py' accordingly.


```

Verification of Ollama's operational status can be achieved by navigating to `http://localhost:11434` in your web browser. A confirmation message, typically "Ollama is running," should be displayed.

### 2. Repository Cloning

To obtain a local copy of the framework, execute the following commands:

```
git clone https://github.com/your-username/langgraph-multi-agent-framework.git
cd langgraph-multi-agent-framework


```

### 3. Python Environment Setup and Dependency Installation

**For Poetry Users (Recommended Approach):**

```
poetry install
poetry shell


```

**For pip Users:**

```
python -m venv venv
source venv/bin/activate # For Windows environments: .\venv\Scripts\activate
pip install -r requirements.txt # Generation of a 'requirements.txt' file may be necessary prior to this step.


```

In the absence of a `requirements.txt` file, it can be generated via `pip freeze > requirements.txt` subsequent to dependency installation, or dependencies may be installed individually:

```
pip install langchain langchain-openai langgraph yfinance


```

### 4. Application Execution

To launch the application, execute the `main.py` script:

```
python main.py


```

Upon initiation, the system will prompt for user requests.

```
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
--- Debugging State with Messages Found: {'messages': [HumanMessage(content='What is 15 plus 27?'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, "operation": "add"}', 'name': 'perform_calculation'}}]}, tool_calls=[{'id': 'call_YFhL...', 'function': {'arguments': '{"a": 15.0, "b": 27.0, "operation": "add"}', 'name': 'perform_calculation'}}]), ToolMessage(content='42.0', tool_call_id='call_YFhL...')]} ---
Tool Result: 42.0
----------------------------


```

## How to Extend the Framework

This framework is engineered for straightforward expansion. The following sections detail the methodology for incorporating new capabilities:

### 1. Create a New Tool

Should a new agent necessitate interaction with external systems or the execution of specific, deterministic actions, the creation of a new tool is requisite.

-   **Create/Modify `tools/tools.py`**:
    
    -   Define a Python function with the `@tool` decorator.
        
    -   Provide a clear docstring explaining its purpose, arguments, and return values. This docstring is crucial for the LLM to understand when and how to use the tool.
        
    -   Add the new function to the `tools` list at the end of the file.
        
    
    **Example `tools/tools.py` addition:**
    
    ```
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
    
    
    ```
    

### 2. Create a New Agent

An agent encapsulates specific logic, potentially using an LLM and/or tools.

-   **Create a new file in `agents/`**: E.g., `agents/new_agent.py`.
    
-   **Define the agent function**: It must accept `state: AgentState` and return `AgentState` (or `dict` compatible with `AgentState`).
    
-   **Implement agent logic**:
    
    -   **Simple Agent**: For agents not requiring an LLM or external tools, the procedure involves processing the `messages` within the `state` and appending new `AIMessage` or `HumanMessage` objects to the list.
        
    -   **LLM-Powered Agent**: Import `llm_agent` from `llms.ollama_llms`. Use `llm_agent.invoke(messages)` to get an LLM response.
        
    -   **Tool-Calling Agent**: Emulate the established pattern observed in `calculator_agent.py` or `stock_news_agent.py`. The LLM will propose tool invocations, which subsequently require manual execution and the integration of `ToolMessage` results into the state. It is imperative to import the requisite tool(s) for direct execution.
        
    
    ```
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
    
    
    ```
    

### 3. Update the Supervisor Prompt

The supervisor LLM requires explicit awareness of any newly integrated agent to facilitate appropriate request routing.

-   **Modify `supervisor/prompts.py`**:
    
    -   Update the `SUPERVISOR_PROMPT` to include your new agent in the "Available agents" list.
        
    -   Provide a concise description of when to route to this new agent.
        
    
    **Example `supervisor/prompts.py` update:**
    
    ```
    # ... existing prompt ...
    Available agents:
    # ... existing agents ...
    - 'weather_agent': Use this for requests asking about current weather conditions for a city.
    # ...
    
    
    ```
    

### 4. Integrate the New Agent into `main.py`

-   **Import the new agent function**:
    
    ```
    from agents.weather_agent import weather_agent
    
    
    ```
    
-   **Add a new node to the workflow**:
    
    ```
    workflow.add_node("weather_agent", weather_agent)
    
    
    ```
    
-   **Add a conditional edge from the supervisor**: Map the supervisor's output to your new agent's node name.
    
    ```
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state['next'],
        {
            # ... existing mappings ...
            "weather_agent": "weather_agent", # New mapping
            "END": END,
        }
    )
    
    
    ```
    
-   **Add an edge from the new agent back to the supervisor**:
    
    ```
    workflow.add_edge("weather_agent", "supervisor") # New edge
    
    
    ```
    

### 5. Test Your New Agent

Execute `python main.py` and validate the new agent's functionality by submitting a relevant query (e.g., "What is the current weather in London?"). Scrutinize the console output to confirm accurate routing by the supervisor and the agent's expected execution.

## Contributing

Users are encouraged to fork this repository, report issues, and submit pull requests.

## Acknowledgements

This project was developed with the assistance of an AI large language model.

## License

MIT License

Copyright (c) 2025 Arvind Aggarwal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
