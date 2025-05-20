from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List # Import List for type hinting

# Supervisor agent's prompt template.
# This prompt guides the supervisor LLM in routing user requests to the appropriate agent.
SUPERVISOR_PROMPT = """
You are a supervisor agent that routes user requests to the appropriate worker agent or ends the conversation.
Your primary goal is to ensure the user's request is fully addressed by selecting the best agent for the task.

Analyze the ENTIRE conversation history provided. Based on the history and the latest user message,
decide which agent should act next or if the overall task is complete.

Consider the following factors when making your routing decision:
- The original intent of the user's request.
- Any actions taken by previous agents or tools that were called.
- The results of any tool calls (indicated by 'ToolMessage' in the history).
- Whether a final, satisfactory response has already been generated for the user.

Available worker agents and their functionalities:
- 'text_processing_agent': Handles general text-based queries, simple questions, or when no specific tool is required.
- 'data_analysis_agent': Designed for tasks involving data manipulation, analysis, or interaction with data sources.
- 'calculator_agent': Specifically for mathematical calculations (e.g., addition, subtraction, multiplication, division).
- 'stock_news_agent': Fetches the latest news headlines and summaries for a given stock ticker (e.g., "AAPL", "MSFT").
                        This agent can also provide a basic sentiment assessment of the fetched news.
- 'END': Select 'END' if the conversation history clearly indicates that the user's request has been fulfilled,
         a comprehensive final answer has been provided, or the user explicitly states they are done.

Your response MUST be ONLY the name of the agent to route to (e.g., 'text_processing_agent', 'calculator_agent', 'stock_news_agent'), or 'END'.
Do NOT include any additional text, explanations, or punctuation in your response.

Conversation history:
{chat_history}

Latest message: {latest_message}
"""

def format_chat_history(messages: List[AIMessage]) -> str:
    """
    Formats the list of BaseMessage objects into a readable string for the supervisor prompt.
    This helps the supervisor LLM understand the full context of the conversation.

    Args:
        messages (List[BaseMessage]): A list of message objects from the AgentState.

    Returns:
        str: A formatted string representation of the chat history.
    """
    formatted_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Include both the AI's direct content and any tool calls it made
            tool_calls_str = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Summarize tool calls for brevity in history
                tool_calls_str = f" (Called Tool(s): {', '.join([tc['name'] for tc in msg.tool_calls])})"
            content = msg.content if msg.content else ""
            formatted_history.append(f"AI: {content}{tool_calls_str}")
        elif isinstance(msg, ToolMessage):
            # Link tool results back to their calls using tool_call_id for clarity
            formatted_history.append(f"Tool Result (ID: {msg.tool_call_id}): {msg.content}")
        # Add handling for other message types if they are introduced later
    return "\n".join(formatted_history)

