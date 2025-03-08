"""
Trading strategy implementations.
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import config
import indicators

logger = logging.getLogger("strategy")

# Import notifications if enabled
if config.TELEGRAM_ENABLED:
    try:
        from notifications import get_notifier
        notifier = get_notifier()
    except ImportError:
        logger.warning("Failed to import notifications module. Notifications will not be sent.")
        notifier = None
else:
    notifier = None

class Strategy(ABC):
    """Base strategy class that all strategies should inherit from."""
    
    def __init__(self, exchange):
        """
        Initialize the strategy.
        
        Args:
            exchange: Exchange instance for market data and order execution
        """
        self.exchange = exchange
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.position = None  # Current position: None, 'long', or 'short'
        self.position_size = config.POSITION_SIZE
        self.stop_loss = config.STOP_LOSS
        self.take_profit = config.TAKE_PROFIT
        self.entry_price = 0  # Price at which position was opened
        self.position_amount = 0  # Amount of position
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on market data.
        
        Args:
            data (pandas.DataFrame): Market data with indicators
            
        Returns:
            pandas.DataFrame: Data with signals column added
        """
        pass
    
    def get_data(self, limit=100):
        """Get market data with indicators."""
        df = self.exchange.get_ohlcv(self.symbol, self.timeframe, limit)
        if df is None:
            logger.error("Failed to get market data")
            return None
        return df
    
    def calculate_position_size(self, price):
        """
        Calculate the position size based on account balance and risk settings.
        
        Args:
            price (float): Current price
            
        Returns:
            float: Position size in base currency
        """
        balance = self.exchange.get_balance()
        if balance is None:
            logger.error("Failed to get balance")
            return 0
        
        # Get available balance in quote currency (e.g., USDT)
        quote_currency = self.symbol.split('/')[1]
        available = balance.get('free', {}).get(quote_currency, 0)
        
        # Calculate position size
        amount_in_quote = available * self.position_size
        amount_in_base = amount_in_quote / price
        
        return amount_in_base
    
    def execute_trade(self, signal, price):
        """
        Execute a trade based on the signal.
        
        Args:
            signal (str): 'buy', 'sell', or None
            price (float): Current price
            
        Returns:
            dict: Order information or None
        """
        if signal == 'buy' and self.position != 'long':
            # Calculate position size
            amount = self.calculate_position_size(price)
            if amount <= 0:
                logger.warning("Position size too small, skipping trade")
                return None
            
            # Create buy order
            order = self.exchange.create_order('market', 'buy', amount)
            if order:
                self.position = 'long'
                self.entry_price = price
                self.position_amount = amount
                logger.info(f"Opened long position: {amount} at {price}")
                
                # Send notification
                if notifier and config.NOTIFY_ON_TRADES:
                    notifier.send_trade_notification('buy', self.symbol, amount, price)
                
                return order
        
        elif signal == 'sell' and self.position == 'long':
            # Get balance in base currency
            base_currency = self.symbol.split('/')[0]
            balance = self.exchange.get_balance()
            if balance is None:
                logger.error("Failed to get balance")
                return None
            
            amount = balance.get('free', {}).get(base_currency, 0)
            if amount <= 0:
                logger.warning("No balance to sell")
                return None
            
            # Calculate profit/loss
            pnl = None
            if self.entry_price > 0 and self.position_amount > 0:
                pnl = (price - self.entry_price) * self.position_amount
            
            # Create sell order
            order = self.exchange.create_order('market', 'sell', amount)
            if order:
                self.position = None
                logger.info(f"Closed long position: {amount} at {price}")
                
                # Send notification
                if notifier and config.NOTIFY_ON_TRADES:
                    notifier.send_trade_notification('sell', self.symbol, amount, price, pnl)
                
                # Reset position tracking
                self.entry_price = 0
                self.position_amount = 0
                
                return order
        
        return None
    
    def check_stop_loss_take_profit(self, price):
        """
        Check if stop loss or take profit has been triggered.
        
        Args:
            price (float): Current price
            
        Returns:
            str: 'stop_loss', 'take_profit', or None
        """
        if self.position != 'long' or self.entry_price <= 0:
            return None
        
        # Check stop loss
        if price <= self.entry_price * (1 - self.stop_loss):
            # Calculate profit/loss
            pnl = (price - self.entry_price) * self.position_amount
            
            # Get balance in base currency
            base_currency = self.symbol.split('/')[0]
            balance = self.exchange.get_balance()
            if balance is None:
                logger.error("Failed to get balance")
                return None
            
            amount = balance.get('free', {}).get(base_currency, 0)
            if amount <= 0:
                logger.warning("No balance to sell")
                return None
            
            # Create sell order
            order = self.exchange.create_order('market', 'sell', amount)
            if order:
                self.position = None
                logger.info(f"Stop loss triggered: {amount} at {price}")
                
                # Send notification
                if notifier and config.NOTIFY_ON_TRADES:
                    notifier.send_trade_notification('stop_loss', self.symbol, amount, price, pnl)
                
                # Reset position tracking
                self.entry_price = 0
                self.position_amount = 0
                
                return 'stop_loss'
        
        # Check take profit
        elif price >= self.entry_price * (1 + self.take_profit):
            # Calculate profit/loss
            pnl = (price - self.entry_price) * self.position_amount
            
            # Get balance in base currency
            base_currency = self.symbol.split('/')[0]
            balance = self.exchange.get_balance()
            if balance is None:
                logger.error("Failed to get balance")
                return None
            
            amount = balance.get('free', {}).get(base_currency, 0)
            if amount <= 0:
                logger.warning("No balance to sell")
                return None
            
            # Create sell order
            order = self.exchange.create_order('market', 'sell', amount)
            if order:
                self.position = None
                logger.info(f"Take profit triggered: {amount} at {price}")
                
                # Send notification
                if notifier and config.NOTIFY_ON_TRADES:
                    notifier.send_trade_notification('take_profit', self.symbol, amount, price, pnl)
                
                # Reset position tracking
                self.entry_price = 0
                self.position_amount = 0
                
                return 'take_profit'
        
        return None
    
    def run(self, limit=100):
        """
        Run the strategy once.
        
        Args:
            limit (int): Number of candles to analyze
            
        Returns:
            dict: Order information or None
        """
        # Get market data
        df = self.get_data(limit)
        if df is None:
            return None
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Get the latest signal
        latest = df.iloc[-1]
        signal = latest.get('signal')
        price = latest['close']
        
        # Check stop loss / take profit
        sl_tp_result = self.check_stop_loss_take_profit(price)
        if sl_tp_result:
            return {'type': sl_tp_result, 'price': price}
        
        # Send signal notification if enabled
        if notifier and config.NOTIFY_ON_SIGNALS and signal:
            # Get confidence if available (for ML strategies)
            confidence = latest.get('confidence') if hasattr(latest, 'confidence') else None
            notifier.send_signal_notification(self.__class__.__name__, self.symbol, signal, confidence)
        
        # Execute trade based on signal
        return self.execute_trade(signal, price)


class SMAStrategy(Strategy):
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, exchange, short_period=None, long_period=None):
        """
        Initialize the SMA strategy.
        
        Args:
            exchange: Exchange instance
            short_period (int): Short period for SMA
            long_period (int): Long period for SMA
        """
        super().__init__(exchange)
        self.short_period = short_period or config.SMA_SHORT
        self.long_period = long_period or config.SMA_LONG
        logger.info(f"Initialized SMA strategy: {self.short_period}/{self.long_period}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on SMA crossover.
        
        Args:
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        # Add SMA indicators
        df = indicators.add_sma(data, self.short_period, self.long_period)
        
        # Calculate crossover
        df['signal'] = None
        
        # Previous values
        df[f'prev_sma_{self.short_period}'] = df[f'sma_{self.short_period}'].shift(1)
        df[f'prev_sma_{self.long_period}'] = df[f'sma_{self.long_period}'].shift(1)
        
        # Buy signal: short SMA crosses above long SMA
        buy_condition = (
            (df[f'prev_sma_{self.short_period}'] <= df[f'prev_sma_{self.long_period}']) & 
            (df[f'sma_{self.short_period}'] > df[f'sma_{self.long_period}'])
        )
        df.loc[buy_condition, 'signal'] = 'buy'
        
        # Sell signal: short SMA crosses below long SMA
        sell_condition = (
            (df[f'prev_sma_{self.short_period}'] >= df[f'prev_sma_{self.long_period}']) & 
            (df[f'sma_{self.short_period}'] < df[f'sma_{self.long_period}'])
        )
        df.loc[sell_condition, 'signal'] = 'sell'
        
        return df


class RSIStrategy(Strategy):
    """Relative Strength Index strategy."""
    
    def __init__(self, exchange, period=None, overbought=None, oversold=None):
        """
        Initialize the RSI strategy.
        
        Args:
            exchange: Exchange instance
            period (int): Period for RSI calculation
            overbought (int): Overbought threshold
            oversold (int): Oversold threshold
        """
        super().__init__(exchange)
        self.period = period or config.RSI_PERIOD
        self.overbought = overbought or config.RSI_OVERBOUGHT
        self.oversold = oversold or config.RSI_OVERSOLD
        logger.info(f"Initialized RSI strategy: period={self.period}, overbought={self.overbought}, oversold={self.oversold}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Args:
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        # Add RSI indicator
        df = indicators.add_rsi(data, self.period, self.overbought, self.oversold)
        
        # Calculate signals
        df['signal'] = None
        
        # Previous RSI value
        df['prev_rsi'] = df['rsi'].shift(1)
        
        # Buy signal: RSI crosses above oversold
        buy_condition = (df['prev_rsi'] <= self.oversold) & (df['rsi'] > self.oversold)
        df.loc[buy_condition, 'signal'] = 'buy'
        
        # Sell signal: RSI crosses below overbought
        sell_condition = (df['prev_rsi'] >= self.overbought) & (df['rsi'] < self.overbought)
        df.loc[sell_condition, 'signal'] = 'sell'
        
        return df


class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence strategy."""
    
    def __init__(self, exchange, fast_period=None, slow_period=None, signal_period=None):
        """
        Initialize the MACD strategy.
        
        Args:
            exchange: Exchange instance
            fast_period (int): Fast period for MACD
            slow_period (int): Slow period for MACD
            signal_period (int): Signal period for MACD
        """
        super().__init__(exchange)
        self.fast_period = fast_period or config.MACD_FAST
        self.slow_period = slow_period or config.MACD_SLOW
        self.signal_period = signal_period or config.MACD_SIGNAL
        logger.info(f"Initialized MACD strategy: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD.
        
        Args:
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        # Add MACD indicator
        df = indicators.add_macd(data, self.fast_period, self.slow_period, self.signal_period)
        
        # Calculate signals
        df['signal'] = None
        
        # Previous values
        df['prev_macd'] = df['macd'].shift(1)
        df['prev_macd_signal'] = df['macd_signal'].shift(1)
        
        # Buy signal: MACD crosses above signal line
        buy_condition = (df['prev_macd'] <= df['prev_macd_signal']) & (df['macd'] > df['macd_signal'])
        df.loc[buy_condition, 'signal'] = 'buy'
        
        # Sell signal: MACD crosses below signal line
        sell_condition = (df['prev_macd'] >= df['prev_macd_signal']) & (df['macd'] < df['macd_signal'])
        df.loc[sell_condition, 'signal'] = 'sell'
        
        return df


class MLStrategy(Strategy):
    """Machine Learning based strategy."""
    
    def __init__(self, exchange, ml_model, confidence_threshold=0.7):
        """
        Initialize the ML strategy.
        
        Args:
            exchange: Exchange instance
            ml_model: Machine learning model instance
            confidence_threshold (float): Threshold for signal confidence
        """
        super().__init__(exchange)
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initialized ML strategy with {ml_model.model_name} model")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on machine learning predictions.
        
        Args:
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        # Add all indicators
        df = indicators.add_all_indicators(data)
        
        # Make predictions
        if not self.ml_model.trained:
            logger.warning("ML model not trained yet, using random predictions for demonstration")
            # For demonstration, generate random predictions
            predictions = np.random.random(len(df))
            confidence = predictions
        else:
            # Get predictions from the model
            predictions = self.ml_model.predict(df)
            
            # For models that return probabilities, use them as confidence
            if hasattr(self.ml_model.model, 'predict_proba'):
                confidence = self.ml_model.model.predict_proba(df[self.ml_model.features].values)[:, 1]
            else:
                # For models that don't return probabilities, use a fixed confidence
                confidence = np.ones(len(predictions)) * 0.8
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['confidence'] = confidence
        
        # Generate signals based on predictions and confidence
        df['signal'] = None
        
        # Buy signal: prediction is 1 (price will go up) with high confidence
        buy_condition = (df['prediction'] == 1) & (df['confidence'] >= self.confidence_threshold)
        df.loc[buy_condition, 'signal'] = 'buy'
        
        # Sell signal: prediction is 0 (price will go down) with high confidence
        sell_condition = (df['prediction'] == 0) & (df['confidence'] >= self.confidence_threshold)
        df.loc[sell_condition, 'signal'] = 'sell'
        
        return df
    
    def run(self, limit=100):
        """
        Run the strategy once.
        
        Args:
            limit (int): Number of candles to analyze
            
        Returns:
            dict: Order information or None
        """
        # Get market data
        df = self.get_data(limit)
        if df is None:
            return None
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Get the latest signal and confidence
        latest = df.iloc[-1]
        signal = latest.get('signal')
        price = latest['close']
        confidence = latest.get('confidence', 0)
        
        # Check stop loss / take profit
        sl_tp_result = self.check_stop_loss_take_profit(price)
        if sl_tp_result:
            return {'type': sl_tp_result, 'price': price}
        
        # Send signal notification if enabled
        if notifier and config.NOTIFY_ON_SIGNALS and signal:
            notifier.send_signal_notification(self.__class__.__name__, self.symbol, signal, confidence)
        
        # Execute trade based on signal
        return self.execute_trade(signal, price)


class EnsembleStrategy(Strategy):
    """Ensemble strategy that combines multiple strategies."""
    
    def __init__(self, exchange, strategies=None, weights=None):
        """
        Initialize the ensemble strategy.
        
        Args:
            exchange: Exchange instance
            strategies (list): List of strategy instances
            weights (list): List of weights for each strategy
        """
        super().__init__(exchange)
        self.strategies = strategies or []
        self.weights = weights or [1] * len(self.strategies)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized ensemble strategy with {len(self.strategies)} strategies")
    
    def add_strategy(self, strategy, weight=1):
        """
        Add a strategy to the ensemble.
        
        Args:
            strategy: Strategy instance
            weight (float): Weight for the strategy
        """
        self.strategies.append(strategy)
        self.weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining signals from all strategies.
        
        Args:
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        if not self.strategies:
            logger.error("No strategies in ensemble")
            return data
        
        # Generate signals for each strategy
        signals = []
        for strategy in self.strategies:
            df = strategy.generate_signals(data.copy())
            signals.append(df['signal'])
        
        # Combine signals
        df = data.copy()
        df['signal'] = None
        
        # Convert signals to numeric: buy=1, sell=-1, None=0
        numeric_signals = []
        for signal in signals:
            numeric = pd.Series(0, index=signal.index)
            numeric[signal == 'buy'] = 1
            numeric[signal == 'sell'] = -1
            numeric_signals.append(numeric)
        
        # Calculate weighted average signal
        weighted_signal = sum(s * w for s, w in zip(numeric_signals, self.weights))
        
        # Convert back to categorical
        df.loc[weighted_signal > 0.5, 'signal'] = 'buy'
        df.loc[weighted_signal < -0.5, 'signal'] = 'sell'
        
        return df


def get_strategy(strategy_name, exchange):
    """
    Factory function to get a strategy instance.
    
    Args:
        strategy_name (str): Name of the strategy
        exchange: Exchange instance
        
    Returns:
        Strategy: Strategy instance
    """
    strategies = {
        'SMA': SMAStrategy,
        'RSI': RSIStrategy,
        'MACD': MACDStrategy,
        'ML': None,  # Will be handled separately
        'ENSEMBLE': None  # Will be handled separately
    }
    
    if strategy_name.upper() == 'ML':
        # For ML strategy, we need to import the ml_models module
        try:
            from ml_models import RandomForestModel
            model = RandomForestModel()
            return MLStrategy(exchange, model)
        except ImportError:
            logger.error("ml_models module not found")
            return None
    
    elif strategy_name.upper() == 'ENSEMBLE':
        # Create an ensemble of all available strategies
        ensemble = EnsembleStrategy(exchange)
        ensemble.add_strategy(SMAStrategy(exchange), 1)
        ensemble.add_strategy(RSIStrategy(exchange), 1)
        ensemble.add_strategy(MACDStrategy(exchange), 1)
        
        # Add ML strategy if available
        try:
            from ml_models import RandomForestModel
            model = RandomForestModel()
            ensemble.add_strategy(MLStrategy(exchange, model), 2)  # Higher weight for ML
        except ImportError:
            logger.warning("ml_models module not found, skipping ML strategy")
        
        return ensemble
    
    strategy_class = strategies.get(strategy_name.upper())
    if strategy_class is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None
    
    return strategy_class(exchange) 