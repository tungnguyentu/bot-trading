# Trading Bot

A customizable cryptocurrency trading bot built with Python.

## Features

- Connect to multiple cryptocurrency exchanges using CCXT
- Implement various trading strategies
- Backtest strategies with historical data
- Real-time market data analysis
- Automated trade execution
- Configurable risk management
- **Machine Learning Models for Enhanced Signal Accuracy**
  - Random Forest
  - XGBoost
  - LSTM Neural Networks
  - Ensemble Methods
- **Hyperparameter Tuning** for optimal model performance
- **Ensemble Strategy** combining traditional indicators with ML predictions
- **Telegram Notifications** for real-time alerts
  - Open position notifications
  - Close position notifications
  - Take profit notifications
  - Stop loss notifications
  - Trading signal notifications
  - Error notifications
- **Leverage Trading** support for margin/futures trading

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API credentials:
   ```
   EXCHANGE=binance
   API_KEY=your_api_key
   API_SECRET=your_api_secret
   ```
4. Configure your trading strategy in `config.py`
5. Run the bot:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: Entry point for the trading bot
- `exchange.py`: Exchange connection and API interaction
- `strategy.py`: Trading strategy implementations
- `indicators.py`: Technical indicators for analysis
- `config.py`: Configuration settings
- `utils.py`: Utility functions
- `backtest.py`: Backtesting functionality
- `ml_models.py`: Machine learning models for signal prediction
- `notifications.py`: Telegram notification system

## Available Strategies

### Traditional Strategies
- **SMA**: Simple Moving Average crossover strategy
- **RSI**: Relative Strength Index strategy for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence strategy

### Machine Learning Strategies
- **ML**: Pure machine learning based strategy
- **ENSEMBLE**: Combination of traditional and ML strategies

## Command-Line Arguments

The bot supports various command-line arguments to customize its behavior:

### Basic Arguments
```
python main.py paper --strategy SMA --symbol BTC/USDT
```

### Position Sizing
You can specify a fixed position size directly from the command line using either `--fixed-size` or `--invest`:
```
python main.py paper --strategy SMA --symbol BTC/USDT --invest 50
```
or
```
python main.py paper --strategy SMA --symbol BTC/USDT --fixed-size 50
```
Both commands will use exactly 50 USDT for each trade.

### Leverage Trading
You can enable leverage trading from the command line:
```
python main.py paper --strategy SMA --symbol BTC/USDT --invest 50 --leverage 2
```
This will use 2x leverage, effectively trading with 100 USDT (50 USDT × 2).

### Full Example
```
python main.py paper --strategy ENSEMBLE --symbol ETH/USDT --fixed-size 50 --leverage 3 --schedule --interval 5
```
This will:
- Run in paper trading mode
- Use the ENSEMBLE strategy
- Trade ETH/USDT
- Use a fixed position size of 50 USDT
- Apply 3x leverage (effectively trading with 150 USDT)
- Run on a schedule every 5 minutes

## Machine Learning

The bot includes several machine learning models to enhance trading signal accuracy:

### Training a Model

```
python main.py train --model random_forest --symbol BTC/USDT --timeframe 1h --limit 1000
```

Options:
- `--model`: Model type (random_forest, xgboost, lstm, ensemble)
- `--symbol`: Trading pair symbol
- `--timeframe`: Timeframe (1m, 5m, 15m, 1h, etc.)
- `--limit`: Number of candles to fetch
- `--target`: Target variable (price_direction, signal)
- `--horizon`: Prediction horizon (number of periods ahead to predict)
- `--test-size`: Proportion of data to use for testing
- `--tune`: Enable hyperparameter tuning
- `--trials`: Number of trials for hyperparameter tuning
- `--cpu-only`: Use CPU-only mode with optimizations for systems without GPU

### CPU-Only Training

If you don't have a GPU or are experiencing GPU-related issues, you can use the `--cpu-only` flag to optimize training for CPU:

```
python main.py train --model lstm --symbol BTC/USDT --timeframe 1h --limit 1000 --cpu-only
```

This will:
- Disable GPU usage
- Use a smaller, more efficient model architecture
- Reduce the number of features used for training
- Optimize memory usage

For ensemble models, the CPU-only mode will skip the LSTM model and only use tree-based models (Random Forest and XGBoost) which are more efficient on CPU.

### Using ML for Trading

```
python main.py paper --strategy ML
```

Or use the ensemble strategy that combines traditional indicators with ML:

```
python main.py paper --strategy ENSEMBLE
```

## Telegram Notifications

The bot can send real-time notifications to Telegram when important events occur.

### Setting Up Telegram Bot

1. Create a new Telegram bot using BotFather:
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` command and follow the instructions
   - Copy the API token provided by BotFather

2. Get your chat ID:
   - Search for `@userinfobot` on Telegram
   - Start a conversation and it will reply with your chat ID

3. Update your `.env` file with Telegram settings:
   ```
   TELEGRAM_ENABLED=true
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   NOTIFY_ON_SIGNALS=false
   NOTIFY_ON_TRADES=true
   NOTIFY_ON_ERRORS=true
   ```

### Notification Types

- **Trade Notifications**: Sent when positions are opened or closed
- **Stop Loss Notifications**: Sent when stop loss is triggered
- **Take Profit Notifications**: Sent when take profit is triggered
- **Signal Notifications**: Sent when trading signals are generated (optional)
- **Error Notifications**: Sent when errors occur (optional)

## Backtesting

```
python main.py backtest --strategy ENSEMBLE --start-date 2023-01-01 --end-date 2023-12-31
```

## Paper Trading

```
python main.py paper --strategy ML --schedule --interval 5
```

## Examples

### Trading with a Fixed Investment Amount

To trade with a fixed investment of 50 USDT per trade:

```
python main.py paper --strategy SMA --symbol BTC/USDT --invest 50
```

### Trading with Leverage

To trade with 50 USDT and 3x leverage:

```
python main.py paper --strategy SMA --symbol BTC/USDT --invest 50 --leverage 3
```

This will effectively trade with 150 USDT (50 USDT × 3).

### High Leverage Trading Example

For high leverage trading (e.g., 20x leverage):

```
python main.py live --strategy SMA --symbol BTC/USDT --invest 50 --leverage 20
```

This will effectively trade with 1,000 USDT (50 USDT × 20). Use high leverage with extreme caution as it significantly increases risk.

## Live Trading

```
python main.py live --strategy ENSEMBLE
```

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and you can lose money. Using leverage increases risk substantially. 