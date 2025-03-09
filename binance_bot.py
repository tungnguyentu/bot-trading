import time
import logging
import sys
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Binance API enums according to documentation
# https://developers.binance.com/docs/binance-spot-api-docs/enums
# Order Types
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_STOP_LOSS = "STOP_LOSS"
ORDER_TYPE_STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT"
ORDER_TYPE_TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
ORDER_TYPE_LIMIT_MAKER = "LIMIT_MAKER"

# Order Side
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# Time in Force
TIME_IN_FORCE_GTC = "GTC"  # Good Till Canceled
TIME_IN_FORCE_IOC = "IOC"  # Immediate or Cancel
TIME_IN_FORCE_FOK = "FOK"  # Fill or Kill

from config import (BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOL, QUANTITY, INTERVAL, 
                   SLEEP_TIME, TEST_MODE, USE_LEVERAGE, LEVERAGE, FUTURES_MODE, 
                   EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD)
from strategy import TradingStrategy
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger

class BinanceBot:
    def __init__(self):
        # Set up the client based on whether we're using futures or spot
        if FUTURES_MODE and not TEST_MODE:
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
            self.futures_client = self.client
        else:
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
            self.futures_client = None
            
        self.strategy = TradingStrategy()
        self.symbol = TRADING_SYMBOL
        self.quantity = QUANTITY
        self.interval = INTERVAL
        self.sleep_time = SLEEP_TIME
        self.test_mode = TEST_MODE
        self.use_leverage = USE_LEVERAGE and FUTURES_MODE
        self.leverage = LEVERAGE
        self.futures_mode = FUTURES_MODE
        self.test_balance = 1000.0  # Initial test balance in USDT
        self.test_crypto = 0.0      # Initial crypto balance
        
        # Initialize Telegram notifier
        self.notifier = TelegramNotifier()
        
        # Initialize trade logger
        self.trade_logger = TradeLogger()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("trading_bot.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BinanceBot')
        
        if self.test_mode:
            self.logger.info("Running in TEST MODE. No real trades will be executed.")
        else:
            if self.futures_mode:
                self._setup_futures()
    
    def _setup_futures(self):
        """Set up futures trading including leverage."""
        try:
            # Change margin type if needed (ISOLATED or CROSSED)
            try:
                self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
                self.logger.info(f"Changed margin type to ISOLATED for {self.symbol}")
            except BinanceAPIException as e:
                # Error 4046 means the margin type is already what we're setting it to
                if e.code != 4046:
                    self.logger.warning(f"Could not change margin type: {e}")
            
            # Set leverage
            if self.use_leverage:
                response = self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
                self.logger.info(f"Leverage set to {response['leverage']} for {self.symbol}")
                self.notifier.notify_info(f"Leverage set to {response['leverage']}x for {self.symbol}", self.test_mode)
                
        except BinanceAPIException as e:
            self.logger.error(f"Error setting up futures: {e}")
            self.notifier.notify_error(f"Error setting up futures: {e}", self.test_mode)

    def get_historical_klines(self):
        """Get historical candles data from Binance."""
        try:
            if self.futures_mode and not self.test_mode:
                klines = self.client.futures_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    limit=100  # Get last 100 candles
                )
            else:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    limit=100  # Get last 100 candles
                )
            
            candles = []
            for kline in klines:
                candle = {
                    'time': kline[0],
                    'open': kline[1],
                    'high': kline[2],
                    'low': kline[3],
                    'close': kline[4],
                    'volume': kline[5]
                }
                candles.append(candle)
            
            return candles
        except BinanceAPIException as e:
            self.logger.error(f"API Error: {e}")
            self.notifier.notify_error(f"API Error when fetching data: {e}", self.test_mode)
            return None
    
    def execute_buy(self):
        """Execute a buy order."""
        current_price = self._get_current_price()
        
        if self.test_mode:
            self._execute_test_buy(current_price)
            return {"test_order": True, "price": current_price, "quantity": self.quantity}
        else:
            try:
                self.logger.info(f"Executing buy order for {self.symbol}")
                
                if self.futures_mode:
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_MARKET,  # Using updated constant
                        quantity=self.quantity
                    )
                    self.logger.info(f"Futures order executed: {order}")
                    # Get fill price from futures API
                    trades = self.client.futures_account_trades(symbol=self.symbol, limit=1)
                    current_price = float(trades[0]['price']) if trades else current_price
                else:
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_BUY,  # Using updated constant
                        type=ORDER_TYPE_MARKET,  # Using updated constant
                        quantity=self.quantity
                    )
                    self.logger.info(f"Order executed: {order}")
                    current_price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else current_price
                
                # Calculate effective cost (different for leverage)
                cost = current_price * self.quantity
                if self.use_leverage:
                    effective_cost = cost / self.leverage
                else:
                    effective_cost = cost
                    
                # Log the trade
                self.trade_logger.log_buy(self.symbol, current_price, self.quantity, effective_cost)
                
                self.strategy.set_position(True, current_price)
                
                # Include leverage info in notification
                leverage_info = f" ({self.leverage}x leverage)" if self.use_leverage else ""
                self.notifier.notify_position_open(
                    self.symbol, current_price, self.quantity, self.test_mode, leverage_info
                )
                
                return order
            except BinanceAPIException as e:
                self.logger.error(f"Buy order failed: {e}")
                self.notifier.notify_error(f"Buy order failed: {e}", self.test_mode)
                return None
    
    def execute_sell(self):
        """Execute a sell order."""
        current_price = self._get_current_price()
        exit_reason = self.strategy.get_exit_reason()
        profit_loss = 0
        profit_percentage = 0
        
        if self.strategy.last_buy_price > 0:
            base_profit_loss = (current_price - self.strategy.last_buy_price) * self.quantity
            base_profit_percentage = ((current_price / self.strategy.last_buy_price) - 1) * 100
            
            # Apply leverage effect if using leverage
            if self.use_leverage:
                profit_loss = base_profit_loss * self.leverage
                profit_percentage = base_profit_percentage * self.leverage
            else:
                profit_loss = base_profit_loss
                profit_percentage = base_profit_percentage
        
        if self.test_mode:
            self._execute_test_sell(current_price)
            
            # Send notification
            leverage_info = f" ({self.leverage}x leverage)" if self.use_leverage else ""
            self.notifier.notify_position_close(
                self.symbol, current_price, self.quantity, 
                profit_loss, profit_percentage, exit_reason, 
                self.test_mode, leverage_info
            )
            return {"test_order": True, "price": current_price, "quantity": self.quantity}
        else:
            try:
                self.logger.info(f"Executing sell order for {self.symbol}")
                
                if self.futures_mode:
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,  # Using updated constant
                        quantity=self.quantity
                    )
                    self.logger.info(f"Futures order executed: {order}")
                else:
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_SELL,  # Using updated constant
                        type=ORDER_TYPE_MARKET,  # Using updated constant
                        quantity=self.quantity
                    )
                    self.logger.info(f"Order executed: {order}")
                
                # Calculate revenue
                revenue = current_price * self.quantity
                
                # Log the trade
                self.trade_logger.log_sell(
                    self.symbol, current_price, self.quantity, revenue,
                    profit_loss, profit_percentage, exit_reason
                )
                
                self.strategy.set_position(False)
                
                # Send notification with leverage info
                leverage_info = f" ({self.leverage}x leverage)" if self.use_leverage else ""
                self.notifier.notify_position_close(
                    self.symbol, current_price, self.quantity, 
                    profit_loss, profit_percentage, exit_reason, 
                    self.test_mode, leverage_info
                )
                
                return order
            except BinanceAPIException as e:
                self.logger.error(f"Sell order failed: {e}")
                self.notifier.notify_error(f"Sell order failed: {e}", self.test_mode)
                return None
    
    def _execute_test_buy(self, price):
        """Execute a simulated buy order in test mode."""
        # Calculate cost - apply leverage for test mode too
        base_cost = price * self.quantity
        effective_cost = base_cost
        if self.use_leverage and self.futures_mode:
            # In futures with leverage, we only use a fraction of the capital
            effective_cost = base_cost / self.leverage
        
        if effective_cost <= self.test_balance:
            self.test_balance -= effective_cost
            self.test_crypto += self.quantity
            
            leverage_text = f" with {self.leverage}x leverage" if self.use_leverage else ""
            self.logger.info(f"[TEST MODE] BUY: {self.quantity} {self.symbol} at ${price}{leverage_text}")
            self.logger.info(f"[TEST MODE] New balance: ${self.test_balance:.2f}, {self.test_crypto} {self.symbol.replace('USDT', '')}")
            self.strategy.set_position(True, price)
            
            # Log the trade
            self.trade_logger.log_buy(self.symbol, price, self.quantity, effective_cost)
            
            # Send notification with leverage info
            leverage_info = f" ({self.leverage}x leverage)" if self.use_leverage else ""
            self.notifier.notify_position_open(
                self.symbol, price, self.quantity, self.test_mode, leverage_info
            )
        else:
            self.logger.warning(f"[TEST MODE] Not enough balance for buy: ${self.test_balance:.2f} < ${effective_cost:.2f}")
            self.notifier.notify_error(f"Not enough balance for buy: ${self.test_balance:.2f} < ${effective_cost:.2f}", self.test_mode)

    def _execute_test_sell(self, price):
        """Execute a simulated sell order in test mode."""
        if self.test_crypto >= self.quantity:
            revenue = price * self.quantity
            base_profit_loss = 0
            base_profit_percentage = 0
            
            if self.strategy.last_buy_price > 0:
                base_profit_loss = (price - self.strategy.last_buy_price) * self.quantity
                base_profit_percentage = ((price / self.strategy.last_buy_price) - 1) * 100
            
            # Apply leverage effect if using leverage
            if self.use_leverage and self.futures_mode:
                profit_loss = base_profit_loss * self.leverage
                profit_percentage = base_profit_percentage * self.leverage
                # In futures with leverage, we get back our margin plus the leveraged profit
                effective_revenue = (revenue / self.leverage) + profit_loss
            else:
                profit_loss = base_profit_loss
                profit_percentage = base_profit_percentage
                effective_revenue = revenue
            
            self.test_balance += effective_revenue
            self.test_crypto -= self.quantity
            
            exit_reason = self.strategy.get_exit_reason()
            reason_text = ""
            if exit_reason == "take_profit":
                reason_text = "take profit"
            elif exit_reason == "stop_loss":
                reason_text = "stop loss"
            elif exit_reason == "signal":
                reason_text = "strategy signal"
            
            leverage_text = f" with {self.leverage}x leverage" if self.use_leverage else ""
            self.logger.info(f"[TEST MODE] SELL: {self.quantity} {self.symbol} at ${price} ({reason_text}){leverage_text}")
            self.logger.info(f"[TEST MODE] P/L: ${profit_loss:.2f} ({profit_percentage:.2f}%)")
            self.logger.info(f"[TEST MODE] New balance: ${self.test_balance:.2f}, {self.test_crypto} {self.symbol.replace('USDT', '')}")
            
            # Log the trade
            self.trade_logger.log_sell(
                self.symbol, price, self.quantity, effective_revenue, 
                profit_loss, profit_percentage, exit_reason
            )
            
            # Strategy position is updated in the execute_sell method
        else:
            self.logger.warning(f"[TEST MODE] Not enough crypto for sell: {self.test_crypto} < {self.quantity}")
            self.notifier.notify_error(f"Not enough crypto for sell: {self.test_crypto} < {self.quantity}", self.test_mode)
    
    def _get_current_price(self):
        """Get the current price of the trading symbol."""
        try:
            if self.futures_mode and not self.test_mode:
                ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            else:
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get current price: {e}")
            # In case of error, return the last price from candles or 0
            candles = self.get_historical_klines()
            return float(candles[-1]['close']) if candles else 0
    
    def _log_signal_info(self, signal_type):
        """Log detailed information about the current trading signal."""
        info = self.strategy.get_last_signal_info()
        
        if signal_type == "buy":
            stance_emoji = "🟢" if info['market_stance'] == "LONG" else "🔴" if info['market_stance'] == "SHORT" else "⚪"
            stance_indicator = f"{stance_emoji} MARKET STANCE: {info['market_stance']}"
            
            self.logger.info("===== BUY SIGNAL DETAILS =====")
            self.logger.info(f"Price: ${info['price']:.2f}")
            self.logger.info(stance_indicator)
            self.logger.info(f"Short EMA ({EMA_SHORT}): {info['short_ema']:.2f}")
            self.logger.info(f"Long EMA ({EMA_LONG}): {info['long_ema']:.2f}")
            self.logger.info(f"EMA Crossover: {info['ema_crossover']}")
            self.logger.info(f"RSI ({RSI_PERIOD}): {info['rsi']:.2f}")
            self.logger.info(f"RSI Oversold (<{RSI_OVERSOLD}): {info['oversold_condition']}")
            self.logger.info(f"MACD: {info['macd']:.6f}")
            self.logger.info(f"MACD Signal: {info['macd_signal']:.6f}")
            self.logger.info(f"MACD Histogram: {info['macd_histogram']:.6f}")
            self.logger.info(f"MACD Bullish: {info['macd_signal_condition']}")
            self.logger.info("=============================")
            
            # Also send telegram notification with more details
            if info['should_buy']:
                details = (f"{stance_emoji} <b>Buy Signal Details</b> - {info['market_stance']}\n\n"
                          f"Price: ${info['price']:.2f}\n"
                          f"Short EMA ({EMA_SHORT}): {info['short_ema']:.2f}\n"
                          f"Long EMA ({EMA_LONG}): {info['long_ema']:.2f}\n"
                          f"RSI ({RSI_PERIOD}): {info['rsi']:.2f}\n"
                          f"MACD: {info['macd']:.6f}\n"
                          f"MACD Signal: {info['macd_signal']:.6f}\n"
                          f"MACD Histogram: {info['macd_histogram']:.6f}")
                self.notifier.notify_signal(details, self.test_mode)
        
        elif signal_type == "sell":
            stance_emoji = "🟢" if info['market_stance'] == "LONG" else "🔴" if info['market_stance'] == "SHORT" else "⚪"
            stance_indicator = f"{stance_emoji} MARKET STANCE: {info['market_stance']}"
            
            exit_reason = info['exit_reason']
            reason_text = ""
            if exit_reason == "take_profit":
                reason_text = "Take Profit"
            elif exit_reason == "stop_loss":
                reason_text = "Stop Loss"
            elif exit_reason == "signal":
                reason_text = "Strategy Signal"
                
            self.logger.info("===== SELL SIGNAL DETAILS =====")
            self.logger.info(f"Exit Reason: {reason_text}")
            self.logger.info(stance_indicator)
            self.logger.info(f"Current Price: ${info['price']:.2f}")
            self.logger.info(f"Buy Price: ${info['last_buy_price']:.2f}")
            self.logger.info(f"P/L: ${info['profit_loss']:.2f} ({info['profit_percentage']:.2f}%)")
            
            if exit_reason == "signal":
                self.logger.info(f"Short EMA ({EMA_SHORT}): {info['short_ema']:.2f}")
                self.logger.info(f"Long EMA ({EMA_LONG}): {info['long_ema']:.2f}")
                self.logger.info(f"EMA Crossunder: {info['ema_crossunder']}")
                self.logger.info(f"RSI ({RSI_PERIOD}): {info['rsi']:.2f}")
                self.logger.info(f"RSI Overbought (>{RSI_OVERBOUGHT}): {info['overbought_condition']}")
                self.logger.info(f"MACD: {info['macd']:.6f}")
                self.logger.info(f"MACD Signal: {info['macd_signal']:.6f}")
                self.logger.info(f"MACD Histogram: {info['macd_histogram']:.6f}")
            elif exit_reason == "take_profit":
                self.logger.info("Take Profit Target: 2.00%")
                self.logger.info(f"Actual Profit: {info['profit_percentage']:.2f}%")
            elif exit_reason == "stop_loss":
                self.logger.info("Stop Loss Limit: -1.00%")
                self.logger.info(f"Actual Loss: {info['profit_percentage']:.2f}%")
                
            self.logger.info("=============================")
            
            # Also send telegram notification with more details
            if info['should_sell']:
                details = (f"{stance_emoji} <b>Sell Signal Details</b> - {reason_text} - {info['market_stance']}\n\n"
                          f"Current Price: ${info['price']:.2f}\n"
                          f"Buy Price: ${info['last_buy_price']:.2f}\n"
                          f"P/L: ${info['profit_loss']:.2f} ({info['profit_percentage']:.2f}%)")
                
                if exit_reason == "signal":
                    details += (f"\n\nShort EMA ({EMA_SHORT}): {info['short_ema']:.2f}\n"
                              f"Long EMA ({EMA_LONG}): {info['long_ema']:.2f}\n"
                              f"RSI ({RSI_PERIOD}): {info['rsi']:.2f}")
                              
                self.notifier.notify_signal(details, self.test_mode)
    
    # Add a function to log market stance changes even when no trade signals
    def check_and_log_market_stance(self, candles):
        """Check if market stance has changed and log it."""
        # First update the indicators
        self.strategy.should_buy(candles)  # This will update all indicators and stance
        
        # Get signal info
        info = self.strategy.get_last_signal_info()
        # Log if stance has changed
        if info.get('stance_changed', False):
            stance_emoji = "🟢" if info['market_stance'] == "LONG" else "🔴" if info['market_stance'] == "SHORT" else "⚪"
            
            self.logger.info(f"{stance_emoji} MARKET STANCE CHANGED: {info['market_stance']}")
            
            # Send notification
            details = (f"{stance_emoji} <b>Market Stance Changed</b>\n\n"
                      f"New Stance: {info['market_stance']}\n"
                      f"Price: ${info['price']:.2f}\n"
                      f"EMA Crossover: {info['ema_crossover']}\n"
                      f"MACD Histogram: {info['macd_histogram']:.6f}")
            self.notifier.notify_info(details, self.test_mode)

    def run(self):
        """Run the trading bot."""
        self.logger.info(f"Starting the trading bot for {self.symbol}")
        
        # Log leverage information
        if self.use_leverage and self.futures_mode:
            self.logger.info(f"Using {self.leverage}x leverage in Futures mode")
        elif self.futures_mode:
            self.logger.info(f"Running in Futures mode without leverage")
        
        if self.test_mode:
            self.logger.info(f"[TEST MODE] Initial balance: ${self.test_balance:.2f} USDT, {self.test_crypto} {self.symbol.replace('USDT', '')}")
        
        # Notify that bot has started
        leverage_info = f" with {self.leverage}x leverage" if self.use_leverage else ""
        mode_info = "Futures" if self.futures_mode else "Spot"
        self.notifier.notify_info(
            f"Trading bot started\nSymbol: {self.symbol}\nInterval: {self.interval}\nMode: {mode_info}{leverage_info}\nTest mode: {'Yes' if self.test_mode else 'No'}",
            self.test_mode
        )
        
        while True:
            try:
                # Get historical klines
                candles = self.get_historical_klines()
                
                if not candles:
                    self.logger.warning("Failed to retrieve candles data. Retrying...")
                    self._pretty_sleep(self.sleep_time)
                    continue
                
                current_price = float(candles[-1]['close'])
                self.logger.info(f"Current {self.symbol} price: {current_price}")
                
                # Always check for market stance changes
                self.check_and_log_market_stance(candles)
                
                # Check if we should buy
                if self.strategy.should_buy(candles):
                    # Log detailed signal information
                    self._log_signal_info("buy")
                    
                    self.logger.info(f"Buy signal detected at price {current_price}")
                    self.execute_buy()
                
                # Check if we should sell
                elif self.strategy.should_sell(candles):
                    # Log detailed signal information
                    self._log_signal_info("sell")
                    
                    self.logger.info(f"Sell signal detected at price {current_price}")
                    self.execute_sell()
                
                # Sleep before next iteration with a pretty countdown
                self._pretty_sleep(self.sleep_time)
            
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                self.notifier.notify_error(f"An error occurred: {e}", self.test_mode)
                self._pretty_sleep(self.sleep_time)
    
    def _pretty_sleep(self, seconds):
        """Sleep with a pretty countdown display.
        
        Args:
            seconds: The number of seconds to sleep
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=seconds)
        
        try:
            while datetime.now() < end_time:
                remaining = end_time - datetime.now()
                remaining_seconds = remaining.total_seconds()
                
                # Format countdown as mm:ss
                mins = int(remaining_seconds // 60)
                secs = int(remaining_seconds % 60)
                
                # Create a prettier countdown with market stance indicators
                stance_emoji = "🟢" if self.strategy.market_stance == "LONG" else "🔴" if self.strategy.market_stance == "SHORT" else "⚪"
                countdown = f"\r{stance_emoji} Next check in: {mins:02d}:{secs:02d} | Stance: {self.strategy.market_stance}"
                
                # Add current price if available
                try:
                    current_price = float(self._get_current_price())
                    countdown += f" | Price: ${current_price:.2f}"
                except:
                    pass
                
                # Print the countdown (carriage return for updating in place)
                sys.stdout.write(countdown)
                sys.stdout.flush()
                
                # Sleep for a short duration to not consume too much CPU
                time.sleep(0.5)
                
            # Clear the line after countdown is done
            sys.stdout.write("\r" + " " * len(countdown) + "\r")
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            # Clear line on keyboard interrupt
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
            raise
