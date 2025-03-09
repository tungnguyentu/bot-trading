import json
import os
import time
import logging
from datetime import datetime

class TradeLogger:
    """
    A class to log trading activities and maintain trade history.
    """
    def __init__(self, log_file='trade_history.json'):
        self.log_file = log_file
        self.logger = logging.getLogger('TradeLogger')
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)
        
        # Load existing trades
        try:
            with open(log_file, 'r') as f:
                self.trades = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.logger.warning(f"Could not load trade history from {log_file}, initializing empty history")
            self.trades = []
    
    def log_buy(self, symbol, price, quantity, cost, market_stance="NEUTRAL", timestamp=None):
        """Log a buy trade."""
        if timestamp is None:
            timestamp = time.time()
            
        trade = {
            'type': 'buy',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'cost': cost,
            'market_stance': market_stance,
            'timestamp': timestamp
        }
        
        self.trades.append(trade)
        self._save_trades()
        self.logger.info(f"Logged buy: {symbol} at ${price:.2f} - Stance: {market_stance}")
        return trade
    
    def log_sell(self, symbol, price, quantity, revenue, profit_loss, profit_percentage, exit_reason='', market_stance="NEUTRAL", timestamp=None):
        """Log a sell trade."""
        if timestamp is None:
            timestamp = time.time()
            
        trade = {
            'type': 'sell',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'revenue': revenue,
            'profit_loss': profit_loss,
            'profit_percentage': profit_percentage,
            'exit_reason': exit_reason,
            'market_stance': market_stance,
            'timestamp': timestamp
        }
        
        self.trades.append(trade)
        self._save_trades()
        self.logger.info(f"Logged sell: {symbol} at ${price:.2f}, P/L: ${profit_loss:.2f} ({profit_percentage:.2f}%) - Stance: {market_stance}")
        return trade

    def log_stance_change(self, symbol, price, old_stance, new_stance, timestamp=None):
        """Log a market stance change."""
        if timestamp is None:
            timestamp = time.time()
            
        event = {
            'type': 'stance_change',
            'symbol': symbol,
            'price': price,
            'old_stance': old_stance,
            'new_stance': new_stance,
            'timestamp': timestamp
        }
        
        self.trades.append(event)
        self._save_trades()
        self.logger.info(f"Logged stance change: {symbol} at ${price:.2f} from {old_stance} to {new_stance}")
        return event
    
    def get_trades(self, symbol=None, trade_type=None, start_time=None, end_time=None):
        """Get filtered trades."""
        filtered_trades = self.trades
        
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.get('symbol') == symbol]
        
        if trade_type:
            filtered_trades = [t for t in filtered_trades if t.get('type') == trade_type]
        
        if start_time:
            filtered_trades = [t for t in filtered_trades if t.get('timestamp', 0) >= start_time]
        
        if end_time:
            filtered_trades = [t for t in filtered_trades if t.get('timestamp', float('inf')) <= end_time]
        
        return filtered_trades
    
    def calculate_statistics(self, symbol=None):
        """Calculate trading statistics."""
        trades = self.get_trades(symbol=symbol)
        buy_trades = [t for t in trades if t.get('type') == 'buy']
        sell_trades = [t for t in trades if t.get('type') == 'sell']
        
        total_trades = len(buy_trades)
        winning_trades = len([t for t in sell_trades if t.get('profit_loss', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('profit_loss', 0) <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) > 0])
        total_loss = abs(sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) <= 0]))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'net_profit': total_profit - total_loss
        }
        
        return stats
    
    def _save_trades(self):
        """Save trades to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trades: {e}")
