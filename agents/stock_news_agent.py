from state import AgentState
from llms.ollama_llms import llm_agent # LLM bound to tools
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import List, Dict, Any
import json # Useful if tool output were stringified JSON, though not strictly needed here

# Import the specific tool this agent will use
from tools.tools import get_stock_news

def stock_news_agent(state: AgentState) -> AgentState:
    """
    An agent that specializes in fetching the latest stock news and performing
    a basic sentiment analysis on the retrieved information.
    """
    print("---Executing Stock News Agent---")
    messages = state['messages']

    # Invoke the tool-calling LLM with the current conversation history.
    # The LLM will decide if the 'get_stock_news' tool should be called based on the user's request.
    response = llm_agent.invoke(messages)

    # Append the LLM's response (which may contain tool calls) to the state.
    messages.append(response)

    # --- Tool Calling Execution Logic ---
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

    if tool_calls:
        print(f"---Stock News Agent received tool calls: {tool_calls}---")
        tool_messages = []
        fetched_news_data = None # Variable to store the news data fetched by the tool

        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            # Ensure the LLM called the expected tool
            if tool_name == "get_stock_news":
                try:
                    ticker = tool_args.get('ticker')
                    if ticker:
                        # Manually invoke the 'get_stock_news' tool with the extracted ticker.
                        tool_result = get_stock_news.invoke(ticker)
                        print(f"---Tool '{tool_name}' executed, result: {tool_result}---")
                        fetched_news_data = tool_result # Store the result for sentiment analysis
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

        # Add the results of the tool execution (ToolMessages) back to the conversation history.
        messages.extend(tool_messages)
        print("---Tool messages added to state---")

        # --- Sentiment Analysis and Response Generation ---
        # Proceed with sentiment analysis only if news data was successfully fetched and is in the expected format.
        # The tool returns a list of dictionaries (for news) or a list with a single string (for errors).
        if fetched_news_data and isinstance(fetched_news_data, list) and fetched_news_data and isinstance(fetched_news_data[0], dict):
            print("---Performing sentiment analysis on news data---")
            # Extract news summaries, filtering out empty ones, to feed to the LLM for sentiment analysis.
            news_summaries = [item.get('summary', '') for item in fetched_news_data if item.get('summary', '').strip()]
            ticker = tool_calls[0]['args'].get('ticker', 'the stock') # Get ticker from the initial tool call

            if news_summaries:
                # Construct a prompt for the LLM to perform sentiment analysis on the summaries.
                sentiment_prompt = f"""
                Analyze the following news summaries for {ticker.upper()} and provide a brief overall sentiment assessment
                (e.g., generally positive, negative, mixed, neutral).
                Also, summarize the key themes and important points from the news.

                News Summaries:
                {'- '.join(news_summaries)}

                Provide your response as a concise summary of the sentiment and key themes.
                """
                # Invoke the LLM with the sentiment analysis prompt.
                sentiment_response = llm_agent.invoke([HumanMessage(content=sentiment_prompt)])
                sentiment_summary = sentiment_response.content

                # Generate a final, user-facing response combining headlines and sentiment analysis.
                headlines_text = "\n".join([f"- {item.get('title', 'No Title Found')}" for item in fetched_news_data])
                final_response_content = f"Here is the latest news for {ticker.upper()}:\n{headlines_text}\n\nOverall Sentiment & Themes:\n{sentiment_summary}"
                messages.append(AIMessage(content=final_response_content))
                print("---Sentiment analysis performed and final response generated---")

            else:
                # If no summaries were found (e.g., only headlines available), still list headlines.
                headlines_text = "\n".join([f"- {item.get('title', 'No Title Found')}" for item in fetched_news_data])
                final_response_content = f"Here is the latest news for {ticker.upper()}:\n{headlines_text}\n\nCould not perform detailed sentiment analysis as summaries were not available."
                messages.append(AIMessage(content=final_response_content))
                print("---No summaries found, listed headlines---")

        elif fetched_news_data and isinstance(fetched_news_data, list) and fetched_news_data and isinstance(fetched_news_data[0], str):
            # Handle cases where the tool returned an error message string (e.g., "No news found...").
            error_message = fetched_news_data[0]
            final_response_content = f"Could not fetch news: {error_message}"
            messages.append(AIMessage(content=final_response_content))
            print("---News tool returned an error message---")

    return {"messages": messages}