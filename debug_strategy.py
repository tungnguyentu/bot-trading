"""
Debug tool to analyze trading signals and indicators.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from exchange import Exchange
import config
import indicators
from strategy import get_strategy

def analyze_strategy(strategy_name, symbol=None, timeframe=None, limit=100, plot=True):
    """
    Analyze a trading strategy and show detailed output.
    
    Args:
        strategy_name (str): Name of the strategy to analyze
        symbol (str): Symbol to analyze
        timeframe (str): Timeframe to analyze
        limit (int): Limit of candles to fetch
        plot (bool): Whether to display a plot
    """
    # Override config settings if specified
    if symbol:
        config.SYMBOL = symbol
    if timeframe:
        config.TIMEFRAME = timeframe
    
    # Create exchange and strategy instances
    exchange = Exchange()
    strategy = get_strategy(strategy_name, exchange)
    
    if not strategy:
        print(f"❌ Unknown strategy: {strategy_name}")
        return
    
    print(f"🔍 Analyzing {strategy_name} on {config.SYMBOL} ({config.TIMEFRAME})")
    
    # Get market data
    df = exchange.get_ohlcv(config.SYMBOL, config.TIMEFRAME, limit)
    if df is None:
        print("❌ Failed to get market data")
        return
    
    print(f"📊 Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Generate signals
    df_with_signals = strategy.generate_signals(df)
    
    # Print signal statistics
    buy_signals = df_with_signals[df_with_signals['signal'] == 'buy']
    sell_signals = df_with_signals[df_with_signals['signal'] == 'sell']
    
    print(f"📈 Buy signals: {len(buy_signals)}")
    print(f"📉 Sell signals: {len(sell_signals)}")
    
    # Detailed analysis
    if len(buy_signals) > 0:
        print("\n🟢 BUY SIGNALS:")
        for idx, row in buy_signals.iterrows():
            print(f"   {idx}: Price {row['close']:.2f}")
    
    if len(sell_signals) > 0:
        print("\n🔴 SELL SIGNALS:")
        for idx, row in sell_signals.iterrows():
            print(f"   {idx}: Price {row['close']:.2f}")
    
    # Signal distribution over time
    if len(buy_signals) > 0 or len(sell_signals) > 0:
        print("\n⏱️ SIGNAL DISTRIBUTION BY MONTH:")
        df_with_signals['month'] = df_with_signals.index.to_period('M')
        monthly_signals = pd.crosstab(df_with_signals['month'], df_with_signals['signal'])
        print(monthly_signals)
    
    # Plot if requested
    if plot:
        plot_strategy_analysis(df_with_signals, strategy_name)

def plot_strategy_analysis(df, strategy_name):
    """
    Plot strategy analysis including price, indicators, and signals.
    
    Args:
        df (pandas.DataFrame): DataFrame with signals
        strategy_name (str): Name of the strategy
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    if 'rsi' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot price
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    
    # Plot indicators based on strategy
    if strategy_name.upper() == 'SMA':
        # Get SMA column names
        sma_cols = [col for col in df.columns if col.startswith('sma_')]
        for col in sma_cols:
            if not col.startswith('prev_'):
                ax1.plot(df.index, df[col], label=col, alpha=0.7)
    
    elif strategy_name.upper() == 'EMA':
        # Get EMA column names
        ema_cols = [col for col in df.columns if col.startswith('ema_')]
        for col in ema_cols:
            if not col.startswith('prev_'):
                ax1.plot(df.index, df[col], label=col, alpha=0.7)
    
    elif strategy_name.upper() == 'MACD':
        # Plot MACD
        ax1.plot(df.index, df['macd'], label='MACD', color='purple', alpha=0.7)
        ax1.plot(df.index, df['macd_signal'], label='Signal', color='orange', alpha=0.7)
    
    # Plot signals
    buy_signals = df[df['signal'] == 'buy']
    sell_signals = df[df['signal'] == 'sell']
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
    
    # Plot RSI in a separate subplot if available
    if 'rsi' in df.columns:
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax2.axhline(y=df['rsi_overbought'].iloc[0], color='r', linestyle='--', label='Overbought')
        ax2.axhline(y=df['rsi_oversold'].iloc[0], color='g', linestyle='--', label='Oversold')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 100)
    
    # Formatting
    ax1.set_title(f'{strategy_name} Strategy Analysis - {config.SYMBOL} ({config.TIMEFRAME})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'strategy_analysis_{strategy_name}_{config.SYMBOL.replace("/", "_")}.png')
    print(f"\n📊 Analysis plot saved as 'strategy_analysis_{strategy_name}_{config.SYMBOL.replace('/', '_')}.png'")
    plt.show()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Debug and analyze trading strategies')
    parser.add_argument('strategy', help='Strategy to analyze')
    parser.add_argument('--symbol', help='Symbol to analyze')
    parser.add_argument('--timeframe', help='Timeframe to analyze')
    parser.add_argument('--limit', type=int, default=100, help='Limit of candles to fetch')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting')
    parser.set_defaults(plot=True)
    
    args = parser.parse_args()
    analyze_strategy(args.strategy, args.symbol, args.timeframe, args.limit, args.plot)

if __name__ == '__main__':
    main()
