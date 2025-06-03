from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import ast # Used for safely evaluating string representations of lists (e.g., tool outputs)

# Import components from their respective modules
from state import AgentState
from supervisor.supervisor_node import supervisor_node
from agents.text_processing_agent import text_processing_agent
from agents.data_analysis_agent import data_analysis_agent
from agents.calculator_agent import calculator_agent
from agents.stock_news_agent import stock_news_agent
from agents.web_search_agent import web_search_agent
from agents.code_generation_agent import code_generation_agent # NEW
from agents.code_review_agent import code_review_agent       # NEW

# --- Build the LangGraph Application ---

workflow = StateGraph(AgentState)

# 2. Add all worker agent nodes and the supervisor node to the workflow.
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("text_processing_agent", text_processing_agent)
workflow.add_node("data_analysis_agent", data_analysis_agent)
workflow.add_node("calculator_agent", calculator_agent)
workflow.add_node("stock_news_agent", stock_news_agent)
workflow.add_node("web_search_agent", web_search_agent)
workflow.add_node("code_generation_agent", code_generation_agent) # NEW
workflow.add_node("code_review_agent", code_review_agent)       # NEW

# 3. Set the entry point of the graph.
workflow.set_entry_point("supervisor")

# 4. Define the conditional edges for routing from the supervisor.
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state['next'],
    {
        "text_processing_agent": "text_processing_agent",
        "data_analysis_agent": "data_analysis_agent",
        "calculator_agent": "calculator_agent",
        "stock_news_agent": "stock_news_agent",
        "web_search_agent": "web_search_agent",
        "code_generation_agent": "code_generation_agent", # NEW
        "END": END,
    }
)

# 5. Add edges from worker nodes back to the supervisor.
workflow.add_edge("text_processing_agent", "supervisor")
workflow.add_edge("data_analysis_agent", "supervisor")
workflow.add_edge("calculator_agent", "supervisor")
workflow.add_edge("stock_news_agent", "supervisor")
workflow.add_edge("web_search_agent", "supervisor")

# NEW: Define edges for the code generation/review loop
workflow.add_edge("code_generation_agent", "code_review_agent") # Code gen always leads to review
workflow.add_conditional_edges(
    "code_review_agent",
    lambda state: state['next'], # This 'next' is set by the code_review_agent itself
    {
        "code_generation_agent": "code_generation_agent", # Loop back for revision
        "supervisor": "supervisor", # Code is acceptable, go to supervisor
    }
)


# 6. Compile the graph.
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
    # Initialize new state variables here
    initial_state = {
        "messages": [HumanMessage(content=input_message)],
        "code_review_count": 0, # Initialize the counter
        "generated_code": "",
        "code_review_feedback": ""
    }

    all_states = [] # To collect all intermediate states for debugging and final output extraction

    try:
        for s in app.stream(initial_state, {"recursion_limit": 50}):
            print(s)
            all_states.append(s)

        print("---Agent execution finished---")

        print("\n--- Agent's Final Output ---")

        state_with_messages = None
        for s in all_states:
            if isinstance(s, dict) and 'messages' in s:
                state_with_messages = s
            for key, value in (s.items() if isinstance(s, dict) else []):
                if isinstance(value, dict) and 'messages' in value:
                    state_with_messages = value

        if state_with_messages and 'messages' in state_with_messages:
            last_content_message = None
            # Prioritize the explicitly generated code as final output if available
            final_generated_code = state_with_messages.get('generated_code', '').strip()
            if final_generated_code:
                print(f"Final Generated Code:\n```python\n{final_generated_code}\n```")
                # Also add it to messages for history if not already there as AIMessage
                # This check ensures we don't duplicate if the agent already added it
                if not any("Generated Code:" in msg.content for msg in reversed(state_with_messages.get('messages', [])) if isinstance(msg, AIMessage)):
                     state_with_messages['messages'].append(AIMessage(content=f"Final Code:\n```python\n{final_generated_code}\n```"))

            # Find the last actual AI/Tool response that is user-facing
            for msg in reversed(state_with_messages.get('messages', [])):
                if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                    # Filter out internal routing messages from supervisor if they end up here
                    if not msg.content.startswith("Supervisor routing to:"):
                        last_content_message = msg
                        break
                elif isinstance(msg, ToolMessage) and msg.content and msg.content.strip():
                    last_content_message = msg
                    break

            if last_content_message:
                print(f"AI Response: {last_content_message.content}")
            else:
                print("No clear final AI or Tool message with content found.")
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
        if user_input.strip():
            run_agent(user_input)
        else:
            print("Please enter a request.")