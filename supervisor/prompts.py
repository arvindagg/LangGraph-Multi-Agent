# supervisor/prompts.py

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List

SUPERVISOR_PROMPT = """
You are an expert workflow supervisor. Your ONLY job is to determine the next action in a multi-agent conversation.

You must select one of the following options:
- 'text_processing_agent': For summarizing, rephrasing, or extracting information from text.
- 'data_analysis_agent': For analyzing structured data, complex numerical operations beyond simple arithmetic, or general data manipulation.
- 'calculator_agent': Specifically for straightforward mathematical calculations.
- 'stock_news_agent': For fetching and summarizing news for a stock ticker (e.g., "AAPL", "MSFT").
- 'web_search_agent': For general web searches (e.g., current events, facts, knowledge retrieval).
- 'END': Select 'END' if the user's original request has been fully addressed by the previous agent's response,
         or if the user explicitly indicates completion. This is the goal of every successful flow.

Analyze the ENTIRE conversation context, especially the latest message and any previous agent's output, to make your decision.

Your response MUST be a JSON object with a single key "next_action" and its value being one of the EXACT agent names listed above or "END".
DO NOT include any other text, explanation, or conversational remarks. STRICTLY adhere to this format.

Example of expected output:
{{"next_action": "stock_news_agent"}}

Example of expected output when the task is complete:
{{"next_action": "END"}}

Conversation History:
{chat_history}

Latest Message (from user or previous agent's output):
{latest_message}

Your Decision (JSON format only):
"""

def format_chat_history(messages: List[AIMessage]) -> str:
    """
    Formats the list of BaseMessage objects into a readable string for the supervisor prompt.
    Crucially, it now truncates the history to prevent context window issues.
    """
    formatted_history = []
    
    # Still keep truncation, but ensure crucial info from the *latest* part of the conversation is not missed.
    # The 'latest_message' parameter will handle the absolute last message.
    max_history_len = 6 # Reduce this slightly to ensure the 'latest_message' is distinct context
    
    # Iterate through the last `max_history_len` messages
    for msg in messages[-max_history_len:]:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            tool_calls_str = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Summarize tool calls for brevity
                tool_calls_str = f" (Called Tool(s): {', '.join([tc['name'] for tc in msg.tool_calls])})"
            content = msg.content if msg.content else ""
            formatted_history.append(f"AI: {content}{tool_calls_str}")
        elif isinstance(msg, ToolMessage):
            # Include tool results
            formatted_history.append(f"Tool Result: {msg.content}")
    return "\n".join(formatted_history) if formatted_history else "No significant chat history."