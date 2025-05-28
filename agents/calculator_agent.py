# agents/calculator_agent.py

from state import AgentState
from llms.ollama_llms import llm_agent_with_tools # <<< CHANGE THIS LINE
from tools.tools import perform_calculation
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def calculator_agent(state: AgentState) -> AgentState:
    print("---Executing Calculator Agent---")
    messages = state['messages']
    
    # Use the LLM that is bound to tools
    response = llm_agent_with_tools.invoke(messages) # <<< CHANGE THIS LINE
    messages.append(response)

    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call['name'] == "perform_calculation":
                try:
                    result = perform_calculation.invoke(tool_call['args'])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                except Exception as e:
                    messages.append(ToolMessage(content=f"Error performing calculation: {e}", tool_call_id=tool_call['id']))
            else:
                messages.append(AIMessage(content=f"Calculator Agent received an unexpected tool call: {tool_call['name']}"))
    else:
        # If LLM didn't call a tool, it might have responded directly or asked clarifying question
        print("---Calculator Agent: LLM responded directly (no tool call)---")

    return {"messages": messages}