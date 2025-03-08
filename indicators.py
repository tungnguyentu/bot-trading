"""
Technical indicators for market analysis.
"""
import pandas as pd
import numpy as np
import ta
import logging
import config

logger = logging.getLogger("indicators")

def add_sma(df, short_period=None, long_period=None):
    """
    Add Simple Moving Average (SMA) indicators to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        short_period (int): Short period for SMA
        long_period (int): Long period for SMA
        
    Returns:
        pandas.DataFrame: Dataframe with SMA indicators
    """
    short_period = short_period or config.SMA_SHORT
    long_period = long_period or config.SMA_LONG
    
    df = df.copy()
    df[f'sma_{short_period}'] = ta.trend.sma_indicator(df['close'], window=short_period)
    df[f'sma_{long_period}'] = ta.trend.sma_indicator(df['close'], window=long_period)
    return df

def add_ema(df, short_period=None, long_period=None):
    """
    Add Exponential Moving Average (EMA) indicators to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        short_period (int): Short period for EMA
        long_period (int): Long period for EMA
        
    Returns:
        pandas.DataFrame: Dataframe with EMA indicators
    """
    short_period = short_period or config.EMA_SHORT
    long_period = long_period or config.EMA_LONG
    
    df = df.copy()
    df[f'ema_{short_period}'] = ta.trend.ema_indicator(df['close'], window=short_period)
    df[f'ema_{long_period}'] = ta.trend.ema_indicator(df['close'], window=long_period)
    return df

def add_macd(df, fast_period=None, slow_period=None, signal_period=None):
    """
    Add Moving Average Convergence Divergence (MACD) indicator to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        fast_period (int): Fast period for MACD
        slow_period (int): Slow period for MACD
        signal_period (int): Signal period for MACD
        
    Returns:
        pandas.DataFrame: Dataframe with MACD indicators
    """
    fast_period = fast_period or config.MACD_FAST
    slow_period = slow_period or config.MACD_SLOW
    signal_period = signal_period or config.MACD_SIGNAL
    
    df = df.copy()
    macd = ta.trend.MACD(
        df['close'], 
        window_fast=fast_period, 
        window_slow=slow_period, 
        window_sign=signal_period
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    return df

def add_rsi(df, period=None, overbought=None, oversold=None):
    """
    Add Relative Strength Index (RSI) indicator to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        period (int): Period for RSI calculation
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
    Returns:
        pandas.DataFrame: Dataframe with RSI indicator
    """
    period = period or config.RSI_PERIOD
    overbought = overbought or config.RSI_OVERBOUGHT
    oversold = oversold or config.RSI_OVERSOLD
    
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    df['rsi_overbought'] = overbought
    df['rsi_oversold'] = oversold
    return df

def add_bollinger_bands(df, window=20, window_dev=2):
    """
    Add Bollinger Bands indicator to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        window (int): Window for moving average
        window_dev (int): Standard deviation multiplier
        
    Returns:
        pandas.DataFrame: Dataframe with Bollinger Bands indicators
    """
    df = df.copy()
    bollinger = ta.volatility.BollingerBands(
        df['close'], window=window, window_dev=window_dev
    )
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['bollinger_width'] = bollinger.bollinger_wband()
    return df

def add_all_indicators(df):
    """
    Add all available indicators to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLCV dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with all indicators
    """
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    return df 