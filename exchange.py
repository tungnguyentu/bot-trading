"""
Exchange connection and API interaction module.
"""
import ccxt
import logging
import pandas as pd
from datetime import datetime
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE if config.LOG_LEVEL != "DEBUG" else None
)
logger = logging.getLogger("exchange")

class Exchange:
    """
    Handles all exchange-related operations including:
    - Connection to exchange API
    - Fetching market data
    - Executing trades
    - Managing orders
    """
    
    def __init__(self):
        """Initialize the exchange connection."""
        self.exchange_id = config.EXCHANGE
        self.symbol = config.SYMBOL
        
        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': config.API_KEY,
                'secret': config.API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            print(f"Connected to {self.exchange_id}")
        except Exception as e:
            print(f"Failed to connect to {self.exchange_id}: {e}")
            raise
    
    def get_balance(self):
        """Get account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            print(f"Failed to fetch balance: {e}")
            return None
    
    def get_ticker(self, symbol=None):
        """Get current ticker data."""
        symbol = symbol or self.symbol
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol=None, timeframe=None, limit=100):
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (1m, 5m, 15m, 1h, etc.)
            limit (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        symbol = symbol or self.symbol
        timeframe = timeframe or config.TIMEFRAME
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Failed to fetch OHLCV data for {symbol}: {e}")
            return None
    
    def create_order(self, order_type, side, amount, price=None, params={}):
        """
        Create a new order.
        
        Args:
            order_type (str): 'limit' or 'market'
            side (str): 'buy' or 'sell'
            amount (float): Amount to buy/sell
            price (float): Price for limit orders
            params (dict): Additional parameters
            
        Returns:
            dict: Order information
        """
        if config.MODE == "paper":
            print(f"Paper trading: {side} {amount} {self.symbol} at {price}")
            return {
                "id": f"paper_{datetime.now().timestamp()}",
                "symbol": self.symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "closed" if order_type == "market" else "open"
            }
        
        try:
            order = self.exchange.create_order(
                self.symbol, order_type, side, amount, price, params
            )
            print(f"Created order: {order['id']}")
            return order
        except Exception as e:
            print(f"Failed to create order: {e}")
            return None
    
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        if config.MODE == "paper":
            print(f"Paper trading: Cancelled order {order_id}")
            return {"id": order_id, "status": "canceled"}
        
        try:
            result = self.exchange.cancel_order(order_id, self.symbol)
            print(f"Cancelled order: {order_id}")
            return result
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return None
    
    def get_order(self, order_id):
        """Get information about an order."""
        if config.MODE == "paper":
            print(f"Paper trading: Get order {order_id}")
            return {"id": order_id, "status": "closed"}
        
        try:
            order = self.exchange.fetch_order(order_id, self.symbol)
            return order
        except Exception as e:
            print(f"Failed to fetch order {order_id}: {e}")
            return None
    
    def get_open_orders(self):
        """Get all open orders."""
        if config.MODE == "paper":
            print("Paper trading: No open orders")
            return []
        
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            return orders
        except Exception as e:
            print(f"Failed to fetch open orders: {e}")
            return []
    
    def set_leverage(self, leverage, symbol=None):
        """
        Set leverage for a symbol.
        
        Args:
            leverage (float): Leverage multiplier
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Response from the exchange
        """
        symbol = symbol or self.symbol
        
        if config.MODE == "paper":
            print(f"Paper trading: Set leverage to {leverage}x for {symbol}")
            return {"leverage": leverage, "symbol": symbol}
        
        try:
            # Different exchanges have different methods to set leverage
            if hasattr(self.exchange, 'set_leverage'):
                result = self.exchange.set_leverage(leverage, symbol)
            elif hasattr(self.exchange, 'private_post_leverage'):
                # For some exchanges like BitMEX
                result = self.exchange.private_post_leverage({
                    'symbol': self.exchange.market_id(symbol),
                    'leverage': leverage
                })
            else:
                # Try to set leverage through options
                self.exchange.options['defaultLeverage'] = leverage
                result = {"leverage": leverage, "symbol": symbol}
            
            print(f"Set leverage to {leverage}x for {symbol}")
            return result
        except Exception as e:
            print(f"Failed to set leverage for {symbol}: {e}")
            return None 