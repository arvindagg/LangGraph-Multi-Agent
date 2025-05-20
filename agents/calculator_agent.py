from state import AgentState
from llms.ollama_llms import llm_agent # LLM specifically bound to tools
from tools.tools import tools # All available tools
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def calculator_agent(state: AgentState) -> AgentState:
    """
    An agent designed to handle mathematical calculation requests.
    It uses a tool-calling LLM to decide if and how to use the 'perform_calculation' tool,
    and then manually executes the tool.
    """
    print("---Executing Calculator Agent---")
    messages = state['messages']

    # Invoke the tool-calling LLM.
    # The LLM analyzes the messages and decides if a tool call (specifically 'perform_calculation')
    # is needed to fulfill the request.
    response = llm_agent.invoke(messages)

    # Add the LLM's response (which may contain tool calls or a direct answer) to the state.
    messages.append(response)

    # --- Manual Tool Calling Execution Logic ---
    # Check if the LLM's response includes any tool calls.
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

    if tool_calls:
        print(f"---Calculator Agent received tool calls: {tool_calls}---")
        tool_messages = [] # To store the results of tool executions

        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            # Locate the specific tool function from the imported 'tools' list.
            executed_tool = next((t for t in tools if t.name == tool_name), None)

            if executed_tool:
                try:
                    # Execute the tool with the arguments provided by the LLM.
                    tool_result = executed_tool.invoke(tool_args)
                    print(f"---Tool '{tool_name}' executed, result: {tool_result}---")
                    # Append a ToolMessage with the result to be added to the state.
                    tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
                except Exception as e:
                    print(f"---Error executing tool '{tool_name}': {e}---")
                    tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call_id))
            else:
                print(f"---Tool '{tool_name}' not found by Calculator Agent---")
                tool_messages.append(ToolMessage(content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call_id))

        # Add all generated ToolMessages (results) back to the conversation history in the state.
        messages.extend(tool_messages)
        print("---Tool messages added to state---")

        # After tool execution, the flow typically returns to the supervisor to decide the next step.
        # The supervisor might route back to the calculator agent or to a different agent
        # to process the tool results or generate a final user-facing response.

    return {"messages": messages}