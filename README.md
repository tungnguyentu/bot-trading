# Binance Trading Bot

A simple cryptocurrency trading bot for Binance exchange.

## Features

- Connects to Binance API (supports both Spot and Futures markets)
- Implements technical analysis with EMA, RSI, and MACD indicators
- Follows a simple trading strategy based on these indicators
- Executes buy/sell orders automatically
- Supports leveraged trading on Binance Futures
- Test mode for simulating trades without risking real funds
- Telegram notifications for important events
- Web dashboard for monitoring performance
- Backtesting utility to test strategies on historical data
- Configurable via environment variables

## Setup

1. Clone the repository
2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example` and add your Binance API keys:
   ```
   cp .env.example .env
   ```
5. Edit the `.env` file with your Binance API key, secret, and trading preferences

## Configuration

You can adjust the following parameters in the `.env` file:

- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret
- `TRADING_SYMBOL` - Symbol to trade (default: BTCUSDT)
- `QUANTITY` - Trading quantity (default: 0.001 BTC)
- `EMA_SHORT` - Short EMA period (default: 9)
- `EMA_LONG` - Long EMA period (default: 21)
- `RSI_PERIOD` - RSI period (default: 14)
- `RSI_OVERBOUGHT` - RSI overbought threshold (default: 70)
- `RSI_OVERSOLD` - RSI oversold threshold (default: 30)
- `TAKE_PROFIT_PCT` - Take profit percentage (default: 2.0%)
- `STOP_LOSS_PCT` - Stop loss percentage (default: 1.0%)
- `INTERVAL` - Candle interval (default: 15m)
- `SLEEP_TIME` - Time between checks in seconds (default: 60)
- `TEST_MODE` - Enable test mode to simulate trades (default: true)
- `TELEGRAM_ENABLED` - Enable Telegram notifications (default: false)
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- `USE_LEVERAGE` - Enable leveraged trading (default: false)
- `LEVERAGE` - Leverage multiplier (default: 3)
- `FUTURES_MODE` - Use Binance Futures instead of Spot (default: true)
- `DASHBOARD_ENABLED` - Enable web dashboard (default: true)
- `DASHBOARD_PORT` - Dashboard port (default: 8050)

## Usage

### Using the Command Line Interface

You can start the bot with various options:

```
python main.py [OPTIONS]
```

Available options:
- `--test`: Force test mode (simulated trading)
- `--live`: Force live trading mode (real trading)
- `--dashboard`: Start the web dashboard
- `--port PORT`: Specify dashboard port
- `--backtest`: Run backtesting on historical data
- `--days N`: Specify number of days for backtesting (default: 30)

### Using the Convenience Script

For easier usage, you can use the provided shell script:

```
./run_bot.sh [OPTIONS]
```

Make it executable first:
```
chmod +x run_bot.sh
```

Available options:
- `--test`: Run in test mode
- `--live`: Run in live mode
- `--dashboard`: Start the web dashboard
- `--backtest`: Run backtesting
- `--days N`: Days for backtesting
- `--help`: Show help

### Examples

1. Run in test mode with dashboard:
   ```
   ./run_bot.sh --test --dashboard
   ```

2. Run backtesting for the last 90 days:
   ```
   ./run_bot.sh --backtest --days 90
   ```

3. Run in live trading mode (be careful!):
   ```
   ./run_bot.sh --live
   ```

## Test Mode

The bot includes a test mode that simulates buy and sell operations without making actual trades. This is useful for:
- Testing your strategy without risking real funds
- Debugging the bot's behavior
- Learning how the bot works

When test mode is enabled:
- The bot starts with a simulated balance of 1000 USDT
- All trades are logged but not executed on the exchange
- You'll see the simulated balance change based on the trades

## Dashboard

The bot includes a web dashboard to monitor performance. When enabled:
- Access the dashboard at http://localhost:8050 (or your configured port)
- View current price and indicators
- See trade history and performance metrics
- Monitor your portfolio value over time

## Backtesting

The backtesting utility allows you to test your strategy on historical data:
```
python main.py --backtest --days 30
```

This will:
- Download historical data for the configured symbol
- Run your strategy against this data
- Calculate performance metrics (win rate, profit factor, etc.)
- Plot the results

## Telegram Notifications

The bot can send notifications to Telegram for important events:
- When a position is opened
- When a position is closed (with profit/loss information)
- When take profit or stop loss is triggered
- When trading signals are detected
- When errors occur

To set up Telegram notifications:

1. Create a new Telegram bot using BotFather
2. Get your chat ID by messaging @userinfobot
3. Add the following to your `.env` file:
```
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## Leverage Trading

The bot can trade with leverage on Binance Futures. When leveraged trading is enabled:
- Trades are executed on the Binance Futures platform
- Profits and losses are multiplied by the leverage factor
- The bot automatically sets the selected leverage on startup

To configure leverage trading:
```
# Enable or disable futures trading
FUTURES_MODE=true

# Enable or disable leverage
USE_LEVERAGE=true

# Set leverage multiplier (1-125 depending on the symbol)
LEVERAGE=3
```

**Warning:** Trading with leverage significantly increases risk. Use with caution.

## Warning

Trading cryptocurrencies involves risk, especially with leverage. This bot is provided for educational purposes only. Always test with small amounts and use at your own risk.

## License

MIT
