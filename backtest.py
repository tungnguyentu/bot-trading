"""
Backtesting module for testing trading strategies with historical data.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import config
import indicators
from strategy import get_strategy

logger = logging.getLogger("backtest")

class Backtest:
    """
    Backtesting class for testing trading strategies with historical data.
    """
    
    def __init__(self, exchange, strategy_name, start_date=None, end_date=None):
        """
        Initialize the backtester.
        
        Args:
            exchange: Exchange instance for market data
            strategy_name (str): Name of the strategy to test
            start_date (str): Start date for backtesting (YYYY-MM-DD)
            end_date (str): End date for backtesting (YYYY-MM-DD)
        """
        self.exchange = exchange
        self.strategy_name = strategy_name
        self.start_date = start_date or config.BACKTEST_START
        self.end_date = end_date or config.BACKTEST_END
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.initial_balance = 1000.0  # Default initial balance in quote currency
        
        # Convert dates to timestamps
        self.start_timestamp = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        self.end_timestamp = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        
        logger.info(f"Initialized backtester for {strategy_name} from {start_date} to {end_date}")
    
    def get_historical_data(self):
        """
        Get historical data for backtesting.
        
        Returns:
            pandas.DataFrame: Historical OHLCV data
        """
        # This is a simplified version. In a real implementation, you would:
        # 1. Fetch data from the exchange for the specified date range
        # 2. Handle pagination if needed
        # 3. Handle rate limits
        
        try:
            # For simplicity, we'll just get the last 500 candles
            # In a real implementation, you would fetch data for the specific date range
            df = self.exchange.get_ohlcv(self.symbol, self.timeframe, limit=500)
            
            if df is None or len(df) == 0:
                logger.error("Failed to get historical data")
                return None
            
            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            logger.info(f"Got {len(df)} candles for backtesting")
            return df
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def run(self):
        """
        Run the backtest.
        
        Returns:
            dict: Backtest results
        """
        # Get historical data
        data = self.get_historical_data()
        if data is None or len(data) == 0:
            logger.error("No data for backtesting")
            return None
        
        # Initialize strategy
        strategy = get_strategy(self.strategy_name, self.exchange)
        if strategy is None:
            logger.error(f"Unknown strategy: {self.strategy_name}")
            return None
        
        # Generate signals
        df = strategy.generate_signals(data)
        
        # Initialize backtest variables
        balance = self.initial_balance
        position = None
        position_size = 0
        entry_price = 0
        trades = []
        
        # Run backtest
        for i, row in df.iterrows():
            price = row['close']
            signal = row.get('signal')
            
            # Handle buy signal
            if signal == 'buy' and position is None:
                # Calculate position size (in base currency)
                position_size = (balance * config.POSITION_SIZE) / price
                entry_price = price
                position = 'long'
                
                trades.append({
                    'timestamp': i,
                    'type': 'buy',
                    'price': price,
                    'amount': position_size,
                    'value': position_size * price
                })
                
                logger.debug(f"BUY: {position_size} at {price}")
            
            # Handle sell signal
            elif (signal == 'sell' or 
                  (position == 'long' and price <= entry_price * (1 - config.STOP_LOSS)) or
                  (position == 'long' and price >= entry_price * (1 + config.TAKE_PROFIT))
                 ) and position == 'long':
                
                # Calculate profit/loss
                pnl = position_size * (price - entry_price)
                balance += position_size * price
                
                trades.append({
                    'timestamp': i,
                    'type': 'sell',
                    'price': price,
                    'amount': position_size,
                    'value': position_size * price,
                    'pnl': pnl
                })
                
                logger.debug(f"SELL: {position_size} at {price}, PnL: {pnl}")
                
                position = None
                position_size = 0
                entry_price = 0
        
        # Close any open position at the end
        if position == 'long':
            price = df.iloc[-1]['close']
            pnl = position_size * (price - entry_price)
            balance += position_size * price
            
            trades.append({
                'timestamp': df.index[-1],
                'type': 'sell',
                'price': price,
                'amount': position_size,
                'value': position_size * price,
                'pnl': pnl
            })
            
            logger.debug(f"FINAL SELL: {position_size} at {price}, PnL: {pnl}")
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        if len(trades_df) == 0:
            logger.warning("No trades executed during backtest")
            return {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'return': 0,
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        # Calculate metrics
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        total_trades = len(sell_trades)
        winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
        losing_trades = len(sell_trades[sell_trades['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sell_trades[sell_trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(sell_trades[sell_trades['pnl'] <= 0]['pnl'].sum())
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_curve = [self.initial_balance]
        for trade in trades:
            if trade['type'] == 'sell':
                equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        equity_curve = np.array(equity_curve)
        max_equity = np.maximum.accumulate(equity_curve)
        drawdown = (max_equity - equity_curve) / max_equity
        max_drawdown = drawdown.max()
        
        # Prepare results
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'return': (balance - self.initial_balance) / self.initial_balance,
            'trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades_df': trades_df,
            'equity_curve': equity_curve
        }
        
        logger.info(f"Backtest results: {results}")
        return results
    
    def plot_results(self, results):
        """
        Plot backtest results.
        
        Args:
            results (dict): Backtest results
        """
        if results is None or 'trades_df' not in results:
            logger.error("No results to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Get data
        data = self.get_historical_data()
        trades_df = results['trades_df']
        
        # Plot price
        ax1.plot(data.index, data['close'], label='Price', color='blue')
        
        # Plot buy and sell signals
        for i, trade in trades_df.iterrows():
            if trade['type'] == 'buy':
                ax1.scatter(trade['timestamp'], trade['price'], marker='^', color='green', s=100)
            else:
                ax1.scatter(trade['timestamp'], trade['price'], marker='v', color='red', s=100)
        
        # Plot equity curve
        ax2.plot(results['equity_curve'], label='Equity', color='purple')
        
        # Set titles and labels
        ax1.set_title(f'Backtest Results: {self.strategy_name} on {self.symbol}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Trades')
        ax2.set_ylabel('Equity')
        ax2.grid(True)
        
        # Add text with performance metrics
        text = (
            f"Initial Balance: {results['initial_balance']:.2f}\n"
            f"Final Balance: {results['final_balance']:.2f}\n"
            f"Return: {results['return']*100:.2f}%\n"
            f"Trades: {results['trades']}\n"
            f"Win Rate: {results['win_rate']*100:.2f}%\n"
            f"Profit Factor: {results['profit_factor']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']*100:.2f}%"
        )
        
        plt.figtext(0.01, 0.01, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"backtest_{self.strategy_name}_{self.symbol.replace('/', '_')}.png")
        plt.close()
        
        logger.info(f"Saved backtest plot to backtest_{self.strategy_name}_{self.symbol.replace('/', '_')}.png")


def run_backtest(exchange, strategy_name, start_date=None, end_date=None):
    """
    Run a backtest for a strategy.
    
    Args:
        exchange: Exchange instance
        strategy_name (str): Name of the strategy to test
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        
    Returns:
        dict: Backtest results
    """
    backtest = Backtest(exchange, strategy_name, start_date, end_date)
    results = backtest.run()
    
    if results:
        backtest.plot_results(results)
    
    return results 