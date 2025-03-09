from indicators import calculate_ema, calculate_rsi, calculate_macd
from config import (EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERBOUGHT, 
                   RSI_OVERSOLD, TAKE_PROFIT_PCT, STOP_LOSS_PCT)

class TradingStrategy:
    def __init__(self):
        self.in_position = False
        self.last_buy_price = 0
        self.exit_reason = ""
        self.last_signal_info = {}
        self.trade_history = []
        self.market_stance = "NEUTRAL"  # Can be "LONG", "SHORT", or "NEUTRAL"
    
    def should_buy(self, prices):
        """Determine if we should buy based on our strategy."""
        # Extract closing prices
        close_prices = [float(candle['close']) for candle in prices]
        current_price = close_prices[-1]
        
        # Calculate indicators
        short_ema = calculate_ema(close_prices, EMA_SHORT)
        long_ema = calculate_ema(close_prices, EMA_LONG)
        rsi = calculate_rsi(close_prices, RSI_PERIOD)
        macd_data = calculate_macd(close_prices)
        
        # Strategy logic: Buy when short EMA crosses above long EMA and RSI is oversold
        ema_crossover = short_ema > long_ema
        oversold_condition = rsi < RSI_OVERSOLD
        macd_signal = macd_data['histogram'] > 0 and macd_data['macd'] > macd_data['signal']
        
        # Determine market stance based on indicators
        if ema_crossover and macd_data['histogram'] > 0:
            new_stance = "LONG"
        elif not ema_crossover and macd_data['histogram'] < 0:
            new_stance = "SHORT"
        else:
            new_stance = "NEUTRAL"
            
        # Record if stance changed
        stance_changed = new_stance != self.market_stance
        self.market_stance = new_stance
        
        # Save signal information for logging
        self.last_signal_info = {
            'price': current_price,
            'short_ema': short_ema,
            'long_ema': long_ema,
            'rsi': rsi,
            'macd': macd_data['macd'],
            'macd_signal': macd_data['signal'],
            'macd_histogram': macd_data['histogram'],
            'ema_crossover': ema_crossover,
            'oversold_condition': oversold_condition,
            'macd_signal_condition': macd_signal,
            'market_stance': self.market_stance,
            'stance_changed': stance_changed,
            'should_buy': ema_crossover and oversold_condition and macd_signal and not self.in_position
        }
        return self.last_signal_info['should_buy']
    
    def should_sell(self, prices):
        """Determine if we should sell based on our strategy."""
        if not self.in_position:
            return False
            
        close_prices = [float(candle['close']) for candle in prices]
        current_price = close_prices[-1]
        
        # Calculate indicators
        short_ema = calculate_ema(close_prices, EMA_SHORT)
        long_ema = calculate_ema(close_prices, EMA_LONG)
        rsi = calculate_rsi(close_prices, RSI_PERIOD)
        macd_data = calculate_macd(close_prices)
        
        # Strategy logic: Sell when short EMA crosses below long EMA or RSI is overbought
        # Or take profit at configurable percentage or cut loss at configurable percentage
        ema_crossunder = short_ema < long_ema
        overbought_condition = rsi > RSI_OVERBOUGHT
        take_profit = current_price >= self.last_buy_price * (1 + TAKE_PROFIT_PCT / 100)
        stop_loss = current_price <= self.last_buy_price * (1 - STOP_LOSS_PCT / 100)
        
        # Determine market stance based on indicators
        if ema_crossunder and macd_data['histogram'] < 0:
            new_stance = "SHORT"
        elif not ema_crossunder and macd_data['histogram'] > 0:
            new_stance = "LONG"
        else:
            new_stance = "NEUTRAL"
            
        # Record if stance changed
        stance_changed = new_stance != self.market_stance
        self.market_stance = new_stance
        
        # Set exit reason for notification
        if take_profit:
            self.exit_reason = "take_profit"
        elif stop_loss:
            self.exit_reason = "stop_loss"
        elif ema_crossunder or overbought_condition:
            self.exit_reason = "signal"
        else:
            self.exit_reason = ""
        
        # Calculate profit/loss data
        profit_loss = 0
        profit_percentage = 0
        if self.last_buy_price > 0:
            profit_loss = current_price - self.last_buy_price
            profit_percentage = ((current_price / self.last_buy_price) - 1) * 100
        
        # Save signal information for logging
        self.last_signal_info = {
            'price': current_price,
            'last_buy_price': self.last_buy_price,
            'short_ema': short_ema,
            'long_ema': long_ema,
            'rsi': rsi,
            'macd': macd_data['macd'],
            'macd_signal': macd_data['signal'],
            'macd_histogram': macd_data['histogram'],
            'ema_crossunder': ema_crossunder,
            'overbought_condition': overbought_condition,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'profit_loss': profit_loss,
            'profit_percentage': profit_percentage,
            'take_profit_target': TAKE_PROFIT_PCT,
            'stop_loss_target': STOP_LOSS_PCT,
            'exit_reason': self.exit_reason,
            'market_stance': self.market_stance,
            'stance_changed': stance_changed,
            'should_sell': (ema_crossunder or overbought_condition or take_profit or stop_loss) and self.in_position
        }
            
        return self.last_signal_info['should_sell']
    
    def set_position(self, in_position, buy_price=None):
        """Update position status."""
        # If we're closing a position, record the trade in history
        if self.in_position and not in_position and self.last_buy_price > 0:
            trade_result = {
                'buy_price': self.last_buy_price,
                'sell_price': self.last_signal_info.get('price', 0),
                'profit_percentage': self.last_signal_info.get('profit_percentage', 0),
                'exit_reason': self.exit_reason,
                'timestamp': import_time().time()
            }
            self.trade_history.append(trade_result)
            
        self.in_position = in_position
        if buy_price:
            self.last_buy_price = buy_price
    
    def get_exit_reason(self):
        """Return the reason for exiting a position."""
        return self.exit_reason
    
    def get_last_signal_info(self):
        """Return detailed information about the last signal."""
        return self.last_signal_info
    
    def get_trade_history(self):
        """Return the history of trades."""
        return self.trade_history

    def get_market_stance(self):
        """Return the current market stance."""
        return self.market_stance

# Helper function to import time module only when needed
def import_time():
    import time
    return time
