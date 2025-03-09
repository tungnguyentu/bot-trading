import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta
import argparse
import logging

from config import (BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOL, QUANTITY,
                   EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
                   TAKE_PROFIT_PCT, STOP_LOSS_PCT, INTERVAL, USE_LEVERAGE, LEVERAGE)
from indicators import calculate_ema, calculate_rsi, calculate_macd
# Add import for TradingStrategy
from strategy import TradingStrategy

class Backtester:
    def __init__(self, symbol=TRADING_SYMBOL, interval=INTERVAL, start_date=None, end_date=None):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.strategy = TradingStrategy()
        self.logger = logging.getLogger('Backtester')
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("backtest.log"),
                logging.StreamHandler()
            ]
        )
    
    def get_historical_data(self):
        """Fetch historical data from Binance."""
        self.logger.info(f"Fetching historical data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Convert dates to millisecond timestamps
        start_ts = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch klines from Binance
        klines = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_str=start_ts,
            end_str=end_ts
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        self.logger.info(f"Fetched {len(df)} data points")
        return df
        
    def run_backtest(self, initial_balance=1000.0, use_leverage=USE_LEVERAGE, leverage=LEVERAGE, use_ml=False):
        """Run backtest on historical data."""
        df = self.get_historical_data()
        if df.empty:
            self.logger.error("No historical data available for backtesting")
            return None
        
        # Prepare results
        balance = initial_balance
        crypto_amount = 0.0
        trades = []
        equity_curve = [initial_balance]  # Start with initial balance
        in_position = False
        buy_price = 0
        
        # Add technical indicators to DataFrame
        self.logger.info("Calculating technical indicators...")
        
        # Calculate EMAs
        df[f'ema_{EMA_SHORT}'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df[f'ema_{EMA_LONG}'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate a simple momentum indicator
        df['price_change'] = df['close'].pct_change(5)  # 5-period price change
        
        # Drop rows with NaN values (due to indicators calculation)
        df = df.dropna()
        
        # Debug counters
        conditions_met = {
            'ema_crossover': 0,
            'oversold_condition': 0,
            'macd_signal': 0,
            'upward_momentum': 0,
            'all_conditions': 0,
            'insufficient_balance': 0
        }
        
        # Override RSI thresholds for backtesting to be more selective
        backtest_rsi_overbought = 70  # Increased to be more conservative on sell signals
        backtest_rsi_oversold = 35    # Adjusted to be more selective on buy signals
        
        # Minimum volume requirement to reduce noise trades
        min_volume_threshold = df['volume'].quantile(0.3)  # Only consider top 70% volume bars
        
        self.logger.info(f"Using optimized signal thresholds: RSI oversold={backtest_rsi_oversold}, overbought={backtest_rsi_overbought}")
        self.logger.info(f"Minimum volume threshold: {min_volume_threshold:.2f}")
        
        # Process each candle
        self.logger.info("Running backtest...")
        
        # Track consecutive losing trades to adjust risk
        consecutive_losses = 0
        max_consecutive_losses = 3  # After this many losses, be more conservative
        
        # Create a trend strength indicator based on longer time frame
        df['trend_strength'] = (df['close'].rolling(window=10).mean() - df['close'].rolling(window=30).mean()) / df['close'].rolling(window=30).mean() * 100
        
        # Initialize ML strategy if requested
        ml_strategy = None
        if use_ml:
            try:
                from ml_strategy import MLStrategy
                ml_strategy = MLStrategy(symbol=self.symbol)
                
                # If model doesn't exist yet, train it on historical data
                if not ml_strategy.load_model():
                    self.logger.info("Training ML model on historical data...")
                    # Use 80% of data for training
                    train_size = int(len(df) * 0.8)
                    train_df = df.iloc[:train_size]
                    metrics = ml_strategy.train_model(train_df)
                    self.logger.info(f"ML model trained. Metrics: {metrics}")
            except ImportError:
                self.logger.warning("Could not import ML strategy. Make sure required packages are installed.")
                use_ml = False
            except Exception as e:
                self.logger.error(f"Error initializing ML strategy: {e}")
                use_ml = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_price = row['close']
            
            # Simulate buy signal
            if not in_position:
                # More selective buy conditions
                ema_crossover = row[f'ema_{EMA_SHORT}'] > row[f'ema_{EMA_LONG}']
                ema_crossover_just_happened = (row[f'ema_{EMA_SHORT}'] > row[f'ema_{EMA_LONG}']) and (prev_row[f'ema_{EMA_SHORT}'] <= prev_row[f'ema_{EMA_LONG}'])
                oversold_condition = row['rsi'] < backtest_rsi_oversold
                macd_positive = row['macd_histogram'] > 0
                macd_turning_positive = row['macd_histogram'] > 0 and prev_row['macd_histogram'] <= 0
                upward_momentum = row['price_change'] > 0
                
                # Additional stronger filters
                strong_volume = row['volume'] > min_volume_threshold
                positive_trend = row.get('trend_strength', 0) > 0  # Positive trend direction
                
                # Get ML signal if available
                ml_signal = 0
                if use_ml and ml_strategy:
                    # Use data up to current point to avoid lookahead bias
                    current_data = df.iloc[:i+1].copy()
                    ml_signal = ml_strategy.get_trading_signal(current_data)
                
                # Update condition counters
                if ema_crossover:
                    conditions_met['ema_crossover'] += 1
                if oversold_condition:
                    conditions_met['oversold_condition'] += 1
                if macd_positive:
                    conditions_met['macd_signal'] += 1
                if upward_momentum:
                    conditions_met['upward_momentum'] += 1
                
                # Determine buy signal - incorporate ML if available
                buy_signal = False
                buy_reason = ""
                
                # More selective criteria after consecutive losses
                if consecutive_losses >= max_consecutive_losses:
                    # More strict requirements after losses
                    if ema_crossover_just_happened and oversold_condition and macd_positive and strong_volume and positive_trend:
                        if not use_ml or ml_signal > 0:  # ML confirms or not using ML
                            buy_signal = True
                            buy_reason = "Strong reversal signal after losses" + (" (ML confirmed)" if ml_signal > 0 and use_ml else "")
                else:
                    # Normal trading logic with better filters
                    if use_ml and ml_signal > 0:
                        # ML strongly suggests buying
                        if ema_crossover or macd_positive:  # Confirm with at least one technical indicator
                            buy_signal = True
                            buy_reason = "ML model buy signal with technical confirmation"
                    else:
                        # Traditional strategy
                        if ema_crossover_just_happened and macd_positive and strong_volume:
                            buy_signal = True
                            buy_reason = "EMA crossover with volume"
                        elif oversold_condition and macd_turning_positive and upward_momentum and positive_trend:
                            buy_signal = True
                            buy_reason = "Oversold reversal with momentum"
                
                if buy_signal:
                    conditions_met['all_conditions'] += 1
                    
                    # Calculate quantity based on current balance
                    quantity = QUANTITY
                    cost = current_price * quantity
                    
                    if use_leverage:
                        effective_cost = cost / leverage
                    else:
                        effective_cost = cost
                    
                    if balance >= effective_cost:
                        # Execute buy
                        balance -= effective_cost
                        crypto_amount += quantity
                        buy_price = current_price
                        in_position = True
                        
                        trade = {
                            'type': 'buy',
                            'price': current_price,
                            'quantity': quantity,
                            'cost': effective_cost,
                            'balance': balance,
                            'timestamp': row['timestamp'],
                            'reason': buy_reason
                        }
                        trades.append(trade)
                        self.logger.info(f"BUY at {current_price:.2f} - Reason: {buy_reason} - Balance: {balance:.2f}")
                    else:
                        conditions_met['insufficient_balance'] += 1
            
            # Simulate sell signal
            elif in_position:
                # Improved sell conditions
                current_profit_pct = ((current_price / buy_price) - 1) * 100
                
                ema_crossunder = row[f'ema_{EMA_SHORT}'] < row[f'ema_{EMA_LONG}']
                ema_crossunder_just_happened = (row[f'ema_{EMA_SHORT}'] < row[f'ema_{EMA_LONG}']) and (prev_row[f'ema_{EMA_SHORT}'] >= prev_row[f'ema_{EMA_LONG}'])
                overbought_condition = row['rsi'] > backtest_rsi_overbought
                macd_negative = row['macd_histogram'] < 0
                macd_turning_negative = row['macd_histogram'] < 0 and prev_row['macd_histogram'] >= 0
                downward_momentum = row['price_change'] < 0
                
                # Dynamic stop loss/take profit
                dynamic_stop_loss = STOP_LOSS_PCT
                dynamic_take_profit = TAKE_PROFIT_PCT
                
                # Adjust based on trend strength
                if row.get('trend_strength', 0) > 1.5:  # Strong uptrend
                    dynamic_take_profit *= 1.5  # Aim for larger profits in strong uptrends
                
                # If in profit but showing weakness, preserve profits
                if current_profit_pct > 0.5 and macd_turning_negative:
                    # Implement trailing stop when in profit
                    dynamic_stop_loss = max(0.3, current_profit_pct * 0.5)  # Don't let profits turn into losses
                
                take_profit = current_price >= buy_price * (1 + dynamic_take_profit / 100)
                stop_loss = current_price <= buy_price * (1 - dynamic_stop_loss / 100)
                
                # Determine exit reason
                exit_reason = ""
                if take_profit:
                    exit_reason = "take_profit"
                elif stop_loss:
                    exit_reason = "stop_loss"
                elif ema_crossunder_just_happened:
                    exit_reason = "ema_crossunder"
                elif overbought_condition and macd_negative:
                    exit_reason = "overbought_with_negative_macd"
                elif macd_turning_negative and current_profit_pct > 0.5:  # Only exit on MACD turn if in profit
                    exit_reason = "macd_turning_negative"
                
                # Incorporate ML for sell signals if available
                ml_signal = 0
                if use_ml and ml_strategy:
                    current_data = df.iloc[:i+1].copy()
                    ml_signal = ml_strategy.get_trading_signal(current_data)
                
                # Add ML sell condition to existing conditions
                ml_sell_signal = use_ml and ml_signal < 0 and current_profit_pct > 0.3
                
                # Sell when conditions met (including ML)
                if take_profit or stop_loss or ema_crossunder_just_happened or \
                   (overbought_condition and macd_negative) or \
                   (macd_turning_negative and current_profit_pct > 0.5) or \
                   ml_sell_signal:
                    
                    # Update exit reason if it's an ML-triggered exit
                    if ml_sell_signal:
                        exit_reason = "ml_model_signal"
                
                    # Calculate profit/loss
                    quantity = crypto_amount
                    base_profit_loss = (current_price - buy_price) * quantity
                    
                    if use_leverage:
                        profit_loss = base_profit_loss * leverage
                        revenue = (quantity * current_price / leverage) + profit_loss
                    else:
                        profit_loss = base_profit_loss
                        revenue = quantity * current_price
                    
                    # Execute sell
                    balance += revenue
                    crypto_amount = 0
                    in_position = False
                    
                    profit_pct = (current_price / buy_price - 1) * 100
                    if use_leverage:
                        profit_pct *= leverage
                    
                    trade = {
                        'type': 'sell',
                        'price': current_price,
                        'quantity': quantity,
                        'revenue': revenue,
                        'profit_loss': profit_loss,
                        'profit_percentage': profit_pct,
                        'balance': balance,
                        'exit_reason': exit_reason,
                        'timestamp': row['timestamp']
                    }
                    trades.append(trade)
                    self.logger.info(f"SELL at {current_price:.2f} - P/L: {profit_loss:.2f} ({profit_pct:.2f}%) - Reason: {exit_reason} - Balance: {balance:.2f}")
                    
                    # Track consecutive losses for risk management
                    if profit_pct <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0  # Reset on winning trade
            
            # Track equity curve (balance + current value of crypto)
            current_equity = balance + (crypto_amount * current_price)
            equity_curve.append(current_equity)
        
        # Close any open position at the end
        if in_position:
            final_price = df['close'].iloc[-1]
            
            # Calculate profit/loss
            quantity = crypto_amount
            base_profit_loss = (final_price - buy_price) * quantity
            
            if use_leverage:
                profit_loss = base_profit_loss * leverage
                revenue = (quantity * final_price / leverage) + profit_loss
            else:
                profit_loss = base_profit_loss
                revenue = quantity * final_price
            
            # Execute sell
            balance += revenue
            crypto_amount = 0
            
            profit_pct = (final_price / buy_price - 1) * 100
            if use_leverage:
                profit_pct *= leverage
            
            trade = {
                'type': 'sell',
                'price': final_price,
                'quantity': quantity,
                'revenue': revenue,
                'profit_loss': profit_loss,
                'profit_percentage': profit_pct,
                'balance': balance,
                'exit_reason': 'end_of_backtest',
                'timestamp': df['timestamp'].iloc[-1]
            }
            trades.append(trade)
            self.logger.info(f"Final position closed at {final_price:.2f} - P/L: {profit_loss:.2f} ({profit_pct:.2f}%) - Balance: {balance:.2f}")
        
        # Make sure equity_curve has the same length as dates
        dates = df['timestamp'].tolist()
        
        # If we have dates but no equity points beyond the initial value
        if len(equity_curve) <= 1 and len(dates) > 1:
            # Fill with initial balance for each date
            equity_curve = [initial_balance] * len(dates)
        # If equity curve is one element too long (common due to initial value)
        elif len(equity_curve) == len(dates) + 1:
            # Remove the initial element
            equity_curve = equity_curve[1:]
        # If equity curve is shorter than dates (should not happen usually)
        elif len(equity_curve) < len(dates):
            # Extend equity curve with last value
            last_value = equity_curve[-1] if equity_curve else initial_balance
            equity_curve.extend([last_value] * (len(dates) - len(equity_curve)))
        # If equity curve is longer than dates (could happen with many rapid trades)
        elif len(equity_curve) > len(dates):
            # Truncate equity curve to match dates
            equity_curve = equity_curve[:len(dates)]
        
        # Calculate performance metrics
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        total_trades = len(buy_trades)
        winning_trades = len([t for t in sell_trades if t.get('profit_loss', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('profit_loss', 0) <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) > 0])
        total_loss = abs(sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) <= 0]))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'return_pct': (balance / initial_balance - 1) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': trades,
            'equity_curve': equity_curve,
            'dates': dates
        }
        
        self.logger.info(f"Backtest completed - Return: {results['return_pct']:.2f}% - Win rate: {win_rate:.2f}%")
        
        # Print debug information about trade conditions
        self.logger.info("Buy condition statistics:")
        self.logger.info(f"EMA crossover conditions met: {conditions_met['ema_crossover']}")
        self.logger.info(f"Oversold RSI conditions met: {conditions_met['oversold_condition']}")
        self.logger.info(f"MACD signal conditions met: {conditions_met['macd_signal']}")
        self.logger.info(f"All combined conditions met: {conditions_met['all_conditions']}")
        self.logger.info(f"Trades not executed due to insufficient balance: {conditions_met['insufficient_balance']}")
        
        # Print signal frequency statistics
        self.logger.info("\nSignal frequency statistics:")
        self.logger.info(f"Total price bars analyzed: {len(df)}")
        self.logger.info(f"EMA crossover conditions: {conditions_met['ema_crossover']} ({conditions_met['ema_crossover']/len(df)*100:.1f}%)")
        self.logger.info(f"RSI oversold conditions: {conditions_met['oversold_condition']} ({conditions_met['oversold_condition']/len(df)*100:.1f}%)")
        self.logger.info(f"Positive MACD signals: {conditions_met['macd_signal']} ({conditions_met['macd_signal']/len(df)*100:.1f}%)")
        self.logger.info(f"Upward momentum signals: {conditions_met['upward_momentum']} ({conditions_met['upward_momentum']/len(df)*100:.1f}%)")
        self.logger.info(f"Combined buy signals: {conditions_met['all_conditions']} ({conditions_met['all_conditions']/len(df)*100:.1f}%)")

        return results
    
    def plot_results(self, results):
        """Plot backtest results."""
        if not results:
            self.logger.error("No backtest results to plot")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        
        # Fix for dimension mismatch between dates and equity_curve
        dates = results['dates']
        equity = results['equity_curve']
        
        # Handle case of no trades with flat equity curve
        if len(equity) <= 1:
            # If no trades were made, just create a flat line
            plt.plot(dates, [results['initial_balance']] * len(dates))
            plt.title('Backtest Results (No Trades)')
        else:
            # Handle case where there's one extra point in equity curve
            # (often happens because we start with initial balance)
            if len(equity) > len(dates):
                # Truncate equity to match dates length
                equity = equity[:len(dates)]
                self.logger.info(f"Adjusted equity curve length from {len(results['equity_curve'])} to {len(equity)} to match dates")
            elif len(dates) > len(equity):
                # Truncate dates to match equity length
                dates = dates[:len(equity)]
                self.logger.info(f"Adjusted dates length from {len(results['dates'])} to {len(dates)} to match equity curve")
        
            # Now plot with matching lengths
            plt.plot(dates, equity)
            plt.title('Backtest Results')
        
        plt.ylabel('Account Value')
        plt.grid(True)
        
        # Mark trades on the plot if there were any
        buy_trades = [t for t in results['trades'] if t['type'] == 'buy']
        sell_trades = [t for t in results['trades'] if t['type'] == 'sell']
        
        if buy_trades:
            buy_dates = [t['timestamp'] for t in buy_trades]
            buy_values = [t['price'] * t['quantity'] + t['balance'] for t in buy_trades]
            plt.scatter(buy_dates, buy_values, marker='^', color='g', label='Buy')
        
        if sell_trades:
            sell_dates = [t['timestamp'] for t in sell_trades]
            sell_values = [t['revenue'] + t['balance'] for t in sell_trades]
            plt.scatter(sell_dates, sell_values, marker='v', color='r', label='Sell')
        
        if buy_trades or sell_trades:
            plt.legend()
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        if len(equity) <= 1:
            # If no trades, no drawdown
            plt.plot(dates, [0] * len(dates))
            plt.title('Drawdown (%) - No Trades')
        else:
            equity_array = np.array(equity)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            plt.plot(dates, drawdown)
            plt.title('Drawdown (%)')
        
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
        
        self.logger.info("Results plot saved as backtest_results.png")
    
    def print_performance_summary(self, results):
        """Print a summary of backtest performance."""
        if not results:
            self.logger.error("No backtest results to summarize")
            return
        
        print("\n===== BACKTEST RESULTS =====")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Interval: {self.interval}")
        print(f"Strategy: EMA({EMA_SHORT}/{EMA_LONG}) + RSI({RSI_PERIOD})")
        print(f"Take Profit: {TAKE_PROFIT_PCT}%, Stop Loss: {STOP_LOSS_PCT}%")
        print("----------------------------")
        print(f"Initial Balance: ${results['initial_balance']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print("----------------------------")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.2f}%)")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print("=============================\n")

        
def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategy on historical data')
    parser.add_argument('--symbol', default=TRADING_SYMBOL, help='Trading symbol (default: from config)')
    parser.add_argument('--interval', default=INTERVAL, help='Candle interval (default: from config)')
    parser.add_argument('--start', default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                        help='Start date YYYY-MM-DD (default: 30 days ago)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), 
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--balance', type=float, default=1000.0, 
                        help='Initial balance for backtest (default: 1000)')
    parser.add_argument('--leverage', type=int, default=LEVERAGE, 
                        help='Leverage to use (default: from config)')
    parser.add_argument('--no-leverage', action='store_true', 
                        help='Disable leverage even if enabled in config')
    parser.add_argument('--plot', action='store_true', help='Plot results after backtest')
    parser.add_argument('--use-ml', action='store_true', help='Use machine learning model for trading signals')
    args = parser.parse_args()

    # Initialize backtester
    backtester = Backtester(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end
    )
    
    # Run backtest
    use_leverage = USE_LEVERAGE and not args.no_leverage
    results = backtester.run_backtest(
        initial_balance=args.balance,
        use_leverage=use_leverage,
        leverage=args.leverage,
        use_ml=args.use_ml
    )
    
    # Print and plot results
    if results:
        backtester.print_performance_summary(results)
        if args.plot:
            backtester.plot_results(results)

if __name__ == "__main__":
    main()
