# tools/tools.py

from langchain_core.tools import tool
import operator
import re
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage
from tavily import TavilyClient
import os
import yfinance as yf
# Ensure your .env file is loaded
from dotenv import load_dotenv
load_dotenv()

# Initialize Tavily client (ensure TAVILY_API_KEY is in your .env)
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Your existing tool definitions (e.g., perform_calculation, get_stock_news)

@tool
def perform_calculation(a: float, b: float, operation: str) -> float:
    """
    Performs a mathematical operation on two numbers.
    Args:
        a (float): The first number.
        b (float): The second number.
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
    Returns:
        float: The result of the operation.
    """
    print(f"---Executing perform_calculation tool with {a} {operation} {b}---")
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return a / b
    else:
        return "Error: Invalid operation"

@tool
def get_stock_news(ticker: str) -> List[dict]:
    """
    Fetches the latest news articles for a given stock ticker using yfinance.
    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains 'title' and 'summary' of a news article.
                     Returns an empty list if no news is found or an error occurs, or if valid titles/summaries aren't found after filtering.
    """
    print(f"---Executing get_stock_news tool for ticker: {ticker} using yfinance---")
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news # This is the list of raw news dictionaries
        
        if not news_items:
            print(f"---Yfinance returned an empty news list for {ticker}.---")
            return []
        
        extracted_news = []
        # Get top 5 news articles for processing, adjust as needed
        for item in news_items[:5]: 
            # CORRECTLY ACCESS NESTED 'content' DICTIONARY
            content = item.get('content', {}) # Use .get() with default {} to avoid KeyError if 'content' is missing
            
            title = content.get('title')
            summary = content.get('summary')
            
            # Only add news if both title and summary are present and not effectively empty
            # Also ensures 'N/A' strings are explicitly filtered
            if title and title.strip() and summary and summary.strip() and \
               title.lower() != 'n/a' and summary.lower() != 'n/a':
                extracted_news.append({
                    "title": title,
                    "summary": summary
                })
            else:
                # Debugging print for skipped items
                print(f"---Skipping news item due to missing/empty title or summary: ID={item.get('id', 'N/A')} Title='{title}' Summary='{summary}'---")
        
        # If after filtering, no valid news is found, return empty list
        if not extracted_news:
            print(f"---No valid news articles found for {ticker} after filtering.---")
            return []
        
        return extracted_news
    except Exception as e:
        print(f"---Error fetching stock news for {ticker}: {e}---")
        return [] # Return empty list on error

@tool
def tavily_search(query: str) -> str:
    """
    Performs a web search using the Tavily API and returns a concise summary of the results.
    Useful for answering questions that require up-to-date information, facts, or general knowledge.
    Args:
        query (str): The search query.
    Returns:
        str: A summarized string of the search results.
    """
    print(f"---Executing tavily_search tool with query: '{query}'---")
    try:
        # Perform a basic search and get a summary
        response = tavily_client.search(query=query, search_depth="basic") # "basic" or "advanced"
        
        # Extract relevant content from results
        results_summary = ""
        if response.get("results"):
            for result in response["results"]:
                # You might want to format this more nicely or summarize further
                results_summary += f"Title: {result.get('title')}\nURL: {result.get('url')}\nContent: {result.get('content')}\n\n"
        
        if not results_summary:
            return "No relevant search results found."
        
        # Optionally, use an LLM to summarize results if they are too long or complex
        # For simplicity, returning raw content for now.
        return results_summary[:1000] + "..." if len(results_summary) > 1000 else results_summary # Truncate if very long
    except Exception as e:
        return f"Error performing Tavily search: {e}"

# List all available tools here
tools = [perform_calculation, get_stock_news, tavily_search]