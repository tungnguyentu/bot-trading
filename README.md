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

## Live Trading

```
python main.py live --strategy ENSEMBLE
```

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and you can lose money. 