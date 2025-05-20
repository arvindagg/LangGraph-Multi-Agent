from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import ast # Used for safely evaluating string representations of lists (e.g., tool outputs)

# Import components from their respective modules
from state import AgentState
from supervisor.supervisor_node import supervisor_node
from agents.text_processing_agent import text_processing_agent
from agents.data_analysis_agent import data_analysis_agent
from agents.calculator_agent import calculator_agent
from agents.stock_news_agent import stock_news_agent # New agent for stock news

# --- Build the LangGraph Application ---

# 1. Initialize the StateGraph with the defined AgentState.
# The StateGraph manages how the AgentState is passed and modified between nodes.
workflow = StateGraph(AgentState)

# 2. Add all worker agent nodes and the supervisor node to the workflow.
# Each node corresponds to a Python function that takes and returns the AgentState.
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("text_processing_agent", text_processing_agent)
workflow.add_node("data_analysis_agent", data_analysis_agent)
workflow.add_node("calculator_agent", calculator_agent)
workflow.add_node("stock_news_agent", stock_news_agent) # Add the new stock news agent

# 3. Set the entry point of the graph.
# The execution of the graph always begins at the supervisor node.
workflow.set_entry_point("supervisor")

# 4. Define the conditional edges for routing from the supervisor.
# The supervisor's output (stored in the 'next' field of the state) determines the next node.
workflow.add_conditional_edges(
    "supervisor",
    # The condition function reads the 'next' field from the state and maps it to a node name.
    lambda state: state['next'],
    {
        "text_processing_agent": "text_processing_agent",
        "data_analysis_agent": "data_analysis_agent",
        "calculator_agent": "calculator_agent",
        "stock_news_agent": "stock_news_agent",
        "END": END, # If the supervisor returns 'END', the graph terminates.
    }
)

# 5. Add edges from worker nodes back to the supervisor.
# After a worker agent completes its task (or a step within its task), it typically
# returns control to the supervisor to decide the next overall step in the workflow,
# based on the updated conversation history.
workflow.add_edge("text_processing_agent", "supervisor")
workflow.add_edge("data_analysis_agent", "supervisor")
workflow.add_edge("calculator_agent", "supervisor")
workflow.add_edge("stock_news_agent", "supervisor")

# 6. Compile the graph.
# Compiling finalizes the graph structure and prepares it for execution.
app = workflow.compile()

# --- Helper Function to Run the Agent System ---

def run_agent(input_message: str):
    """
    Helper function to run the multi-agent system with a given user input.
    It initializes the state, streams the execution, and prints the final output.

    Args:
        input_message (str): The user's input message to the agent system.
    """
    print(f"\n---Running agent with input: '{input_message}'---")
    # Initialize the agent state with the user's message as the starting point.
    initial_state = {"messages": [HumanMessage(content=input_message)]}

    all_states = [] # To collect all intermediate states for debugging and final output extraction

    try:
        # Invoke the graph and stream the results.
        # Streaming allows observing the state changes at each step of the graph's execution.
        # A recursion_limit is set to prevent infinite loops in complex graphs.
        for s in app.stream(initial_state, {"recursion_limit": 50}):
            print(s) # Print each state as it updates
            all_states.append(s) # Collect each state for post-execution analysis

        print("---Agent execution finished---")

        # --- Extract and Print Final Output ---
        print("\n--- Agent's Final Output ---")

        # Find the last state that contains the 'messages' key, as this holds the conversation history.
        state_with_messages = None
        for s in all_states:
            # Check if the state dict itself contains 'messages'
            if isinstance(s, dict) and 'messages' in s:
                state_with_messages = s
            # Also check for 'messages' nested under a node's output (e.g., {'agent_name': {'messages': [...]}})
            for key, value in (s.items() if isinstance(s, dict) else []):
                if isinstance(value, dict) and 'messages' in value:
                    state_with_messages = value # Prioritize messages from the latest agent's direct output

        if state_with_messages and 'messages' in state_with_messages:
            # Iterate through messages in reverse to find the last AI or Tool message with content.
            last_content_message = None
            for msg in reversed(state_with_messages.get('messages', [])):
                if isinstance(msg, (AIMessage, ToolMessage)) and msg.content and msg.content.strip():
                    last_content_message = msg
                    break

            if last_content_message:
                if isinstance(last_content_message, AIMessage):
                    print(f"AI Response: {last_content_message.content}")
                elif isinstance(last_content_message, ToolMessage):
                    # For ToolMessages, try to format the output if it's a list (e.g., news articles)
                    print(f"Tool Result: {last_content_message.content}")
                    try:
                        # Safely evaluate the string content as a Python literal (like a list of dicts)
                        tool_output_list = ast.literal_eval(last_content_message.content)
                        if isinstance(tool_output_list, list):
                            print("Formatted Tool Output:")
                            for item in tool_output_list:
                                if isinstance(item, dict):
                                    # Assuming news articles have 'title' and 'summary'
                                    title = item.get('title', 'No Title Found')
                                    summary = item.get('summary', 'No Summary Found')
                                    print(f"- Title: {title}")
                                    print(f"  Summary: {summary}")
                                else:
                                    print(f"- {item}") # For other list items
                    except (ValueError, SyntaxError):
                        # If not a valid Python literal, just print the raw content (already done above)
                        pass
            else:
                print("No final AI or Tool message with content found in the last state.")
        else:
            print("No state containing messages was found during execution.")
        print("----------------------------")

    except Exception as e:
        print(f"\n---An error occurred during agent execution: {e}---")

# --- Interactive Loop for Agent Interaction ---
if __name__ == "__main__":
    print("Welcome to the Multi-Agent System!")
    print("Type your request below, or type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "done"]:
            print("Exiting Multi-Agent System. Goodbye!")
            break
        if user_input.strip(): # Only process non-empty input
            run_agent(user_input)
        else:
            print("Please enter a request.")