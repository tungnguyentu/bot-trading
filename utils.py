"""
Utility functions for the trading bot.
"""
import logging
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import config

logger = logging.getLogger("utils")

def setup_logging():
    """Set up logging configuration."""
    log_level = getattr(logging, config.LOG_LEVEL)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up file handler
    log_file = os.path.join('logs', config.LOG_FILE)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"Logging initialized at level {config.LOG_LEVEL}")

def save_to_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename (str): Filename to save to
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.debug(f"Saved data to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save data to {filename}: {e}")
        return False

def load_from_json(filename):
    """
    Load data from a JSON file.
    
    Args:
        filename (str): Filename to load from
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded data from {filename}")
        return data
    except Exception as e:
        print(f"Failed to load data from {filename}: {e}")
        return None

def plot_ohlcv(df, title=None, save_path=None):
    """
    Plot OHLCV data.
    
    Args:
        df (pandas.DataFrame): OHLCV data
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(df.index, df['close'], label='Close', color='blue')
        
        # Set titles and labels
        ax1.set_title(title or f'Price Chart: {config.SYMBOL}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot volume
        ax2.bar(df.index, df['volume'], label='Volume', color='purple', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Failed to plot OHLCV data: {e}")

def calculate_trade_stats(trades):
    """
    Calculate trade statistics.
    
    Args:
        trades (list): List of trade dictionaries
        
    Returns:
        dict: Trade statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_profit': 0
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Filter sell trades (with PnL)
    sell_trades = df[df['type'] == 'sell']
    
    if len(sell_trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_profit': 0
        }
    
    # Calculate statistics
    winning_trades = sell_trades[sell_trades['pnl'] > 0]
    losing_trades = sell_trades[sell_trades['pnl'] <= 0]
    
    total_trades = len(sell_trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    
    win_rate = num_winning / total_trades if total_trades > 0 else 0
    
    avg_profit = winning_trades['pnl'].mean() if num_winning > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if num_losing > 0 else 0
    
    total_profit = winning_trades['pnl'].sum() if num_winning > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if num_losing > 0 else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'winning_trades': num_winning,
        'losing_trades': num_losing,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_profit': total_profit + (losing_trades['pnl'].sum() if num_losing > 0 else 0)
    }

def format_number(number, decimals=8):
    """
    Format a number with a specific number of decimal places.
    
    Args:
        number (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number
    """
    return f"{number:.{decimals}f}"

def timestamp_to_datetime(timestamp):
    """
    Convert a timestamp to a datetime string.
    
    Args:
        timestamp (int): Timestamp in milliseconds
        
    Returns:
        str: Datetime string
    """
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

def create_data_directory():
    """Create a data directory if it doesn't exist."""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory")
    return os.path.abspath('data') 