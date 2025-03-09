"""
Status display for the trading bot.
"""
import os
import pandas as pd
import time
from datetime import datetime
from exchange import Exchange
import config

def display_status():
    """Display current trading status."""
    os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
    
    print("=" * 50)
    print(f"📊 TRADING BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Exchange connection
    exchange = Exchange()
    
    # Get current price
    try:
        ticker = exchange.get_ticker(config.SYMBOL)
        if ticker:
            current_price = ticker.get('last', None)
            print(f"💹 Current {config.SYMBOL} price: {current_price}")
            
            # Get 24h change if available
            change_24h = ticker.get('percentage', None)
            if change_24h is not None:
                direction = "▲" if change_24h >= 0 else "▼"
                print(f"📈 24h change: {direction} {abs(change_24h):.2f}%")
    except Exception as e:
        print(f"❌ Failed to get current price: {e}")
    
    # Get balance
    try:
        balance = exchange.get_balance()
        if balance:
            # Extract balance for quote currency (e.g. USDT)
            quote_currency = config.SYMBOL.split('/')[1]
            quote_balance = balance.get('free', {}).get(quote_currency, 0)
            print(f"💰 Available {quote_currency}: {quote_balance}")
            
            # Extract balance for base currency (e.g. BTC)
            base_currency = config.SYMBOL.split('/')[0]
            base_balance = balance.get('free', {}).get(base_currency, 0)
            if base_balance > 0:
                print(f"🪙 Available {base_currency}: {base_balance}")
                if current_price:
                    print(f"   Value: {base_balance * current_price:.2f} {quote_currency}")
    except Exception as e:
        print(f"❌ Failed to get balance: {e}")
    
    # Show current config
    print("\n🔧 CONFIGURATION:")
    print(f"   Mode: {config.MODE.upper()}")
    print(f"   Strategy: {config.STRATEGY}")
    print(f"   Symbol: {config.SYMBOL}")
    print(f"   Timeframe: {config.TIMEFRAME}")
    if config.USE_FIXED_POSITION_SIZE:
        print(f"   Position size: {config.FIXED_POSITION_SIZE} {config.SYMBOL.split('/')[1]} (FIXED)")
    else:
        print(f"   Position size: {config.POSITION_SIZE * 100}% of balance")
    if config.LEVERAGE > 1:
        print(f"   Leverage: {config.LEVERAGE}x")
    
    # Recent candles
    try:
        print("\n📊 RECENT MARKET DATA:")
        df = exchange.get_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=5)
        if df is not None:
            # Format dataframe for display
            display_df = df.copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.round(decimals=2)
            print(display_df.tail())
    except Exception as e:
        print(f"❌ Failed to get market data: {e}")
    
    print("\n" + "=" * 50)
    print("Press Ctrl+C to exit")
    print("=" * 50)

if __name__ == "__main__":
    try:
        while True:
            display_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\nExiting status display...")
