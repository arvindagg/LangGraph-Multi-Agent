from langchain_core.tools import tool
from typing import List, Dict, Any
import operator
import yfinance as yf # Import yfinance for fetching stock data

# --- Functions decorated with @tool ---
# These functions are exposed to LLMs that have been bound to 'tools'.
# They serve as specific capabilities that agents can invoke.

@tool
def perform_calculation(a: float, b: float, operation: str) -> float:
    """
    Performs a basic mathematical operation on two numbers.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (str): The mathematical operation to perform.
                         Supported operations are 'add', 'subtract', 'multiply', 'divide'.

    Returns:
        float: The result of the operation.

    Raises:
        ValueError: If an unsupported operation is provided or division by zero is attempted.
    """
    print(f"---Executing perform_calculation tool with {a} {operation} {b}---")
    if operation == 'add':
        return operator.add(a, b)
    elif operation == 'subtract':
        return operator.sub(a, b)
    elif operation == 'multiply':
        return operator.mul(a, b)
    elif operation == 'divide':
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return operator.truediv(a, b)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


@tool
def get_stock_news(ticker: str) -> List[Dict[str, str]]:
    """
    Fetches the latest news headlines and summaries for a given stock ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT", "GOOG").

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary contains
                               'title' and 'summary' of a news article.
                               Returns a list containing an error message string if fetching fails or no news is found.
    """
    print(f"---Executing get_stock_news tool for ticker: {ticker} using yfinance---")

    try:
        # Create a Ticker object for the given stock symbol
        stock = yf.Ticker(ticker)

        # Fetch news articles associated with the stock
        news_list = stock.news

        if not news_list:
            return [f"No recent news found for ticker {ticker.upper()}."]

        extracted_news = []
        for article in news_list:
            # yfinance news structure varies; safely extract title and summary.
            # Assuming 'title' and 'summary' are directly accessible or nested under 'content'
            title = article.get('title', 'No Title Found')
            summary = article.get('summary', 'No Summary Found')

            # Some yfinance versions might nest 'title' and 'summary' under a 'content' key,
            # so we'll add a fallback check if initial direct access fails.
            if title == 'No Title Found' and 'content' in article and isinstance(article['content'], dict):
                title = article['content'].get('title', 'No Title Found')
            if summary == 'No Summary Found' and 'content' in article and isinstance(article['content'], dict):
                summary = article['content'].get('summary', 'No Summary Found')

            extracted_news.append({"title": title, "summary": summary})

        return extracted_news

    except Exception as e:
        print(f"---Error fetching news using yfinance: {e}---")
        return [f"Error fetching news for {ticker.upper()}: {e}"]


# List of all tools available to the multi-agent system.
# Agents whose LLMs are bound to 'tools' can invoke any function in this list.
tools = [perform_calculation, get_stock_news]