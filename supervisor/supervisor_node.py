from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from state import AgentState # Import the shared state definition
from llms.ollama_llms import llm_supervisor # Import the supervisor-specific LLM
from supervisor.prompts import SUPERVISOR_PROMPT, format_chat_history # Import prompt and formatter

def supervisor_node(state: AgentState) -> dict:
    """
    The supervisor node in the LangGraph.
    This node is responsible for analyzing the conversation history and the latest user request,
    then routing the request to the most appropriate worker agent or deciding to end the conversation.

    Args:
        state (AgentState): The current state of the multi-agent system, containing the message history.

    Returns:
        dict: A dictionary containing the 'next' key, whose value is the name of the next agent node
              to execute, or 'END' to terminate the graph.
    """
    print("---Executing Supervisor Node---")
    messages = state['messages']

    # Extract the latest message (user's most recent input)
    latest_message = messages[-1].content if messages else ""
    # Extract the conversation history (all messages except the latest one)
    chat_history = messages[:-1]

    # Format the extracted chat history into a string for inclusion in the prompt
    formatted_history_str = format_chat_history(chat_history)

    # Construct the full prompt for the supervisor LLM
    # The prompt includes detailed instructions, available agents, and the conversation context.
    prompt = SUPERVISOR_PROMPT.format(
        chat_history=formatted_history_str,
        latest_message=latest_message
    )

    # Invoke the supervisor LLM with the formatted prompt.
    # The LLM's response will be the name of the next agent or 'END'.
    response = llm_supervisor.invoke([HumanMessage(content=prompt)])
    next_action = response.content.strip() # Extract and clean the LLM's decision

    print(f"---Supervisor decided next action: {next_action}---")

    # Return the chosen next action, which LangGraph's conditional edge will use for routing.
    return {"next": next_action}