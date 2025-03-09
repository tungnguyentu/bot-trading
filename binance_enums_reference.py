"""
Reference file for Binance API Enums
Based on documentation: https://developers.binance.com/docs/binance-spot-api-docs/enums
"""

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

# Order Status
ORDER_STATUS_NEW = "NEW"
ORDER_STATUS_PARTIALLY_FILLED = "PARTIALLY_FILLED"
ORDER_STATUS_FILLED = "FILLED"
ORDER_STATUS_CANCELED = "CANCELED"
ORDER_STATUS_PENDING_CANCEL = "PENDING_CANCEL"
ORDER_STATUS_REJECTED = "REJECTED"
ORDER_STATUS_EXPIRED = "EXPIRED"

# OCO Status
OCO_STATUS_RESPONSE = "RESPONSE"
OCO_STATUS_EXEC_STARTED = "EXEC_STARTED"
OCO_STATUS_ALL_DONE = "ALL_DONE"

# OCO Order Status
OCO_ORDER_STATUS_EXECUTING = "EXECUTING"
OCO_ORDER_STATUS_ALL_DONE = "ALL_DONE"
OCO_ORDER_STATUS_REJECT = "REJECT"

# Contingency Type
CONTINGENCY_TYPE_OCO = "OCO"

# Kline/Candlestick intervals
INTERVAL_1MINUTE = "1m"
INTERVAL_3MINUTE = "3m"
INTERVAL_5MINUTE = "5m"
INTERVAL_15MINUTE = "15m"
INTERVAL_30MINUTE = "30m"
INTERVAL_1HOUR = "1h"
INTERVAL_2HOUR = "2h"
INTERVAL_4HOUR = "4h"
INTERVAL_6HOUR = "6h"
INTERVAL_8HOUR = "8h"
INTERVAL_12HOUR = "12h"
INTERVAL_1DAY = "1d"
INTERVAL_3DAY = "3d"
INTERVAL_1WEEK = "1w"
INTERVAL_1MONTH = "1M"

# Rate limiters (rate limit types)
RATE_LIMIT_REQUEST_WEIGHT = "REQUEST_WEIGHT"
RATE_LIMIT_ORDERS = "ORDERS"
RATE_LIMIT_RAW_REQUESTS = "RAW_REQUESTS"

# Rate Limit Intervals
RATE_LIMIT_INTERVAL_SECOND = "SECOND"
RATE_LIMIT_INTERVAL_MINUTE = "MINUTE"
RATE_LIMIT_INTERVAL_DAY = "DAY"

# Permission types
PERMISSION_SPOT = "SPOT"
PERMISSION_MARGIN = "MARGIN"
PERMISSION_LEVERAGED = "LEVERAGED"
PERMISSION_TRD_GRP_002 = "TRD_GRP_002"
PERMISSION_TRD_GRP_003 = "TRD_GRP_003"
PERMISSION_TRD_GRP_004 = "TRD_GRP_004"

# Example usage in a function
def create_example_order(client, symbol, side, quantity):
    """
    Example of creating an order using the Binance API constants
    
    Args:
        client: Binance API client
        symbol: Trading pair symbol
        side: Order side (buy/sell)
        quantity: Amount to trade
        
    Returns:
        Order creation result
    """
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,  # Use SIDE_BUY or SIDE_SELL
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error creating order: {e}")
        return None
        
def create_example_limit_order(client, symbol, side, quantity, price):
    """Example of creating a limit order"""
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,  # Use SIDE_BUY or SIDE_SELL
            type=ORDER_TYPE_LIMIT,
            quantity=quantity,
            price=price,
            timeInForce=TIME_IN_FORCE_GTC
        )
        return order
    except Exception as e:
        print(f"Error creating limit order: {e}")
        return None

# Futures specific enums
# Position Side
POSITION_SIDE_BOTH = "BOTH"
POSITION_SIDE_LONG = "LONG"
POSITION_SIDE_SHORT = "SHORT"

# Contract Types
CONTRACT_TYPE_PERPETUAL = "PERPETUAL"
CONTRACT_TYPE_CURRENT_MONTH = "CURRENT_MONTH"
CONTRACT_TYPE_NEXT_MONTH = "NEXT_MONTH"
CONTRACT_TYPE_CURRENT_QUARTER = "CURRENT_QUARTER"
CONTRACT_TYPE_NEXT_QUARTER = "NEXT_QUARTER"

# Futures Order Types
FUTURES_ORDER_TYPE_LIMIT = "LIMIT"
FUTURES_ORDER_TYPE_MARKET = "MARKET"
FUTURES_ORDER_TYPE_STOP = "STOP"
FUTURES_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
FUTURES_ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT"
FUTURES_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
FUTURES_ORDER_TYPE_TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"

# Working Type
WORKING_TYPE_MARK_PRICE = "MARK_PRICE"
WORKING_TYPE_CONTRACT_PRICE = "CONTRACT_PRICE"

# Example of creating a futures market order
def create_futures_market_order(client, symbol, side, quantity, reduce_only=False):
    """Example of creating a futures market order"""
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,  # Use SIDE_BUY or SIDE_SELL
            type=FUTURES_ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=reduce_only
        )
        return order
    except Exception as e:
        print(f"Error creating futures market order: {e}")
        return None
