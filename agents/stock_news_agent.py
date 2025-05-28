# agents/stock_news_agent.py

from state import AgentState
from llms.ollama_llms import llm_agent_with_tools # <<< CHANGE THIS LINE TO llm_agent_with_tools
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from typing import List, Dict, Any
import json

from tools.tools import get_stock_news

# agents/stock_news_agent.py

# ... (existing imports)

def stock_news_agent(state: AgentState) -> AgentState:
    print("---Executing Stock News Agent---")
    messages = state['messages']

    # New: Add a system message to guide the LLM for tool calling
    messages_with_system_prompt = [
        SystemMessage(content="You are an expert at extracting stock tickers. When asked for stock news, ALWAYS identify the ticker from the user's request and use the 'get_stock_news' tool with that ticker. If no ticker is found, ask the user for clarification. Once news is fetched, summarize the sentiment and key themes concisely."),
        *messages
    ]

    response = llm_agent_with_tools.invoke(messages_with_system_prompt) # Use the new message list

    messages.append(response)

    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

    if tool_calls:
        print(f"---Stock News Agent received tool calls: {tool_calls}---")
        tool_messages = []
        fetched_news_data = None

        for tool_call in tool_calls:
             tool_name = tool_call['name']
             tool_args = tool_call['args']
             tool_call_id = tool_call['id']

             if tool_name == "get_stock_news":
                 try:
                     ticker = tool_args.get('ticker')
                     if ticker:
                         tool_result = get_stock_news.invoke(ticker)
                         print(f"---Tool '{tool_name}' executed, result: {tool_result}---")
                         fetched_news_data = tool_result
                         tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
                     else:
                         error_msg = "Missing 'ticker' argument for get_stock_news tool."
                         print(f"---Error executing tool '{tool_name}': {error_msg}---")
                         tool_messages.append(ToolMessage(content=f"Error: {error_msg}", tool_call_id=tool_call_id))

                 except Exception as e:
                     print(f"---Error executing tool '{tool_name}': {e}---")
                     tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call_id))
             else:
                 print(f"---Stock News Agent received call for unknown tool: {tool_name}---")
                 tool_messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call_id))

        messages.extend(tool_messages)
        print("---Tool messages added to state---")

        # --- Sentiment Analysis and Response Generation ---
        # Only proceed if fetched_news_data is a non-empty list of dicts (meaning actual news was returned)
        if fetched_news_data and isinstance(fetched_news_data, list) and fetched_news_data and isinstance(fetched_news_data[0], dict):
            print("---Performing sentiment analysis on news data---")
            news_summaries = [item.get('summary', '') for item in fetched_news_data if item.get('summary', '').strip()]
            ticker = tool_calls[0]['args'].get('ticker', 'the stock')

            if news_summaries:
                sentiment_prompt = f"""
                You are a sentiment analysis expert. Analyze the following news summaries for {ticker.upper()} and provide a brief overall sentiment assessment (e.g., generally positive, negative, mixed, neutral).
                Also, summarize the key themes from the news.
                DO NOT call any tools. Provide your response ONLY as a concise summary of sentiment and themes.

                News Summaries:
                {'- '.join(news_summaries)}

                Sentiment and Themes:
                """
                # Invoke the LLM with the sentiment prompt
                # IMPORTANT: For sentiment analysis, we want an LLM that *doesn't* try to call tools again.
                # If llm_agent_with_tools is too aggressive, you might need a separate llm_text_only from ollama_llms.py
                # For now, let's heavily constrain the prompt.
                sentiment_response = llm_agent_with_tools.invoke([HumanMessage(content=sentiment_prompt)])
                print(f"---Sentiment LLM Response: {sentiment_response}---")
                sentiment_summary = sentiment_response.content
                print(f"---Sentiment Summary Content: {sentiment_summary}---")

                headlines_text = "\n".join([f"- {item.get('title', 'No Title Found')}" for item in fetched_news_data])
                final_response_content = f"Here is the latest news for {ticker.upper()}:\n{headlines_text}\n\nOverall Sentiment & Themes:\n{sentiment_summary}"
                print(f"---Final Response Content (truncated to 500 chars): {final_response_content[:500]}...---")

                messages.append(AIMessage(content=final_response_content))
                print("---Sentiment analysis performed and final response generated---")

            else:
                 # If no summaries were found (but headlines might exist), just list headlines
                 headlines_text = "\n".join([f"- {item.get('title', 'No Title Found')}" for item in fetched_news_data])
                 final_response_content = f"Here is the latest news for {ticker.upper()}:\n{headlines_text}\n\nCould not perform detailed sentiment analysis as summaries were not available."
                 messages.append(AIMessage(content=final_response_content))
                 print("---No summaries found, listed headlines---")

        elif fetched_news_data == []: # <-- NEW: Handle empty list returned by tool
            ticker = tool_calls[0]['args'].get('ticker', 'the stock')
            final_response_content = f"Sorry, I could not find any recent news for {ticker.upper()}."
            messages.append(AIMessage(content=final_response_content))
            print("---No news fetched, informing user.---")

        # Removed the 'elif fetched_news_data and isinstance(fetched_news_data[0], str):'
        # because the tool will now return [] instead of error strings.

    print("---Stock News Agent returning state---")
    state['next'] = "supervisor" # <--- THIS IS CRUCIAL FOR ROUTING
    return state # Always return the updated state