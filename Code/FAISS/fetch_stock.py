# fetch_stock.py

import yfinance as yf

def fetch_stock_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info

    summary = f"""
    Company: {info.get('shortName', 'N/A')}
    Current Price: {info.get('currentPrice', 'N/A')}
    Market Cap: {info.get('marketCap', 'N/A')}
    PE Ratio: {info.get('trailingPE', 'N/A')}
    52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
    52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
    Sector: {info.get('sector', 'N/A')}
    Industry: {info.get('industry', 'N/A')}
    """
    return summary.strip()
