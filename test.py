import yfinance as yf
try:
    stock_tsla = yf.Ticker("TSLA")
    news_tsla = stock_tsla.news
    print(f"---Raw Yfinance News for TSLA ({type(news_tsla)}):")
    print(news_tsla)
except Exception as e:
    print(f"Error directly fetching TSLA with yfinance: {e}")