"""
Configuration settings for the trading bot.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE = os.getenv("EXCHANGE", "binance")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# Trading settings
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")  # 1m, 5m, 15m, 1h, 4h, 1d

# Strategy settings
STRATEGY = os.getenv("STRATEGY", "SMA")  # SMA, EMA, MACD, RSI, ML, ENSEMBLE
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Machine Learning settings
ML_MODEL = "ensemble"  # random_forest, xgboost, lstm, ensemble
ML_CONFIDENCE_THRESHOLD = 0.7
ML_PREDICTION_HORIZON = 3  # Number of periods ahead to predict
ML_TRAIN_TEST_SPLIT = 0.2  # Proportion of data to use for testing
ML_HYPERPARAMETER_TUNING = False  # Whether to tune hyperparameters
ML_HYPERPARAMETER_TRIALS = 50  # Number of trials for hyperparameter tuning

# Risk management
POSITION_SIZE = float(os.getenv("POSITION_SIZE", "0.1"))  # Percentage of available balance to use per trade
FIXED_POSITION_SIZE = float(os.getenv("FIXED_POSITION_SIZE", "0"))  # Fixed position size in quote currency (e.g., USDT)
USE_FIXED_POSITION_SIZE = os.getenv("USE_FIXED_POSITION_SIZE", "false").lower() == "true"  # Whether to use fixed position size
STOP_LOSS = 0.02  # 2% stop loss
TAKE_PROFIT = 0.04  # 4% take profit

# Leverage settings
LEVERAGE = float(os.getenv("LEVERAGE", "1"))  # Leverage multiplier (1 = no leverage)
USE_LEVERAGE = LEVERAGE > 1  # Whether to use leverage

# Backtesting
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2023-12-31"

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "trading_bot.log"

# Execution mode
MODE = os.getenv("MODE", "paper")  # paper, live

# Telegram notifications
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ON_SIGNALS = os.getenv("NOTIFY_ON_SIGNALS", "false").lower() == "true"  # Whether to notify on signals
NOTIFY_ON_TRADES = os.getenv("NOTIFY_ON_TRADES", "true").lower() == "true"  # Whether to notify on trades
NOTIFY_ON_ERRORS = os.getenv("NOTIFY_ON_ERRORS", "true").lower() == "true"  # Whether to notify on errors 