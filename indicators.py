import numpy as np
import pandas as pd

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = pd.Series(prices).diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence."""
    price_series = pd.Series(prices)
    fast_ema = price_series.ewm(span=fast, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1]
    }
