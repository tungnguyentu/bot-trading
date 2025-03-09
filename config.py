import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading parameters
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
QUANTITY = float(os.getenv('QUANTITY', '0.001'))

# Strategy parameters
EMA_SHORT = int(os.getenv('EMA_SHORT', '9'))
EMA_LONG = int(os.getenv('EMA_LONG', '21'))
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))

# For backtesting, we can use slightly different thresholds to get more trades
BACKTEST_RSI_OVERBOUGHT = 65  # More sensitive than regular
BACKTEST_RSI_OVERSOLD = 35    # More sensitive than regular

# We can also use a more appropriate quantity for backtests based on the asset price
BACKTEST_QUANTITY = {
    'BTCUSDT': 0.001,
    'ETHUSDT': 0.01,
    'SOLUSDT': 1.0,
    'DOGEUSDT': 1000,
    'DEFAULT': 1.0  # Default quantity if symbol not found
}

# Take profit and stop loss settings
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '2.0'))  # 2% take profit
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '1.0'))  # 1% stop loss

# Time intervals
INTERVAL = os.getenv('INTERVAL', '15m')
SLEEP_TIME = int(os.getenv('SLEEP_TIME', '60'))  # seconds

# Test mode
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'

# Telegram notification settings
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Leverage settings
USE_LEVERAGE = os.getenv('USE_LEVERAGE', 'false').lower() == 'true'
LEVERAGE = int(os.getenv('LEVERAGE', '3'))
FUTURES_MODE = os.getenv('FUTURES_MODE', 'true').lower() == 'true'

# Dashboard settings
DASHBOARD_ENABLED = os.getenv('DASHBOARD_ENABLED', 'false').lower() == 'true'
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8050'))
