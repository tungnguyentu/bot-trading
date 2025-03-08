"""
Notifications module for sending alerts to Telegram.
"""
import logging
import os
from datetime import datetime
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import config

logger = logging.getLogger("notifications")

class TelegramNotifier:
    """
    Class for sending notifications to Telegram.
    """
    
    def __init__(self, token=None, chat_id=None):
        """
        Initialize the Telegram notifier.
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
        """
        self.token = token or config.TELEGRAM_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.bot = None
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat ID not set. Notifications will not be sent.")
            return
        
        try:
            self.bot = telegram.Bot(token=self.token)
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None
    
    def send_message(self, message):
        """
        Send a message to Telegram.
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.bot:
            logger.warning("Telegram bot not initialized. Message not sent.")
            return False
        
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            logger.debug(f"Sent message to Telegram: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to Telegram: {e}")
            return False
    
    def send_trade_notification(self, trade_type, symbol, amount, price, pnl=None):
        """
        Send a trade notification to Telegram.
        
        Args:
            trade_type (str): Type of trade (buy, sell, stop_loss, take_profit)
            symbol (str): Trading pair symbol
            amount (float): Amount of the trade
            price (float): Price of the trade
            pnl (float): Profit/loss (for sell trades)
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if trade_type.lower() == 'buy':
            message = f"🟢 *OPENED POSITION*\n\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Amount:* {amount:.8f}\n" \
                      f"*Price:* {price:.8f}\n" \
                      f"*Total:* {amount * price:.2f}\n" \
                      f"*Time:* {timestamp}"
        
        elif trade_type.lower() == 'sell':
            message = f"🔴 *CLOSED POSITION*\n\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Amount:* {amount:.8f}\n" \
                      f"*Price:* {price:.8f}\n" \
                      f"*Total:* {amount * price:.2f}\n"
            
            if pnl is not None:
                pnl_emoji = "🟢" if pnl > 0 else "🔴"
                message += f"*PnL:* {pnl_emoji} {pnl:.2f}\n"
            
            message += f"*Time:* {timestamp}"
        
        elif trade_type.lower() == 'stop_loss':
            message = f"🛑 *STOP LOSS TRIGGERED*\n\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Amount:* {amount:.8f}\n" \
                      f"*Price:* {price:.8f}\n" \
                      f"*Total:* {amount * price:.2f}\n"
            
            if pnl is not None:
                message += f"*PnL:* 🔴 {pnl:.2f}\n"
            
            message += f"*Time:* {timestamp}"
        
        elif trade_type.lower() == 'take_profit':
            message = f"💰 *TAKE PROFIT TRIGGERED*\n\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Amount:* {amount:.8f}\n" \
                      f"*Price:* {price:.8f}\n" \
                      f"*Total:* {amount * price:.2f}\n"
            
            if pnl is not None:
                message += f"*PnL:* 🟢 {pnl:.2f}\n"
            
            message += f"*Time:* {timestamp}"
        
        else:
            message = f"*TRADE NOTIFICATION*\n\n" \
                      f"*Type:* {trade_type}\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Amount:* {amount:.8f}\n" \
                      f"*Price:* {price:.8f}\n" \
                      f"*Total:* {amount * price:.2f}\n" \
                      f"*Time:* {timestamp}"
        
        return self.send_message(message)
    
    def send_error_notification(self, error_message):
        """
        Send an error notification to Telegram.
        
        Args:
            error_message (str): Error message
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ *ERROR*\n\n" \
                  f"*Message:* {error_message}\n" \
                  f"*Time:* {timestamp}"
        
        return self.send_message(message)
    
    def send_signal_notification(self, strategy_name, symbol, signal, confidence=None):
        """
        Send a signal notification to Telegram.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading pair symbol
            signal (str): Trading signal (buy, sell)
            confidence (float): Signal confidence (for ML strategies)
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if signal.lower() == 'buy':
            message = f"📈 *BUY SIGNAL*\n\n" \
                      f"*Strategy:* {strategy_name}\n" \
                      f"*Symbol:* {symbol}\n"
            
            if confidence is not None:
                message += f"*Confidence:* {confidence:.2%}\n"
            
            message += f"*Time:* {timestamp}"
        
        elif signal.lower() == 'sell':
            message = f"📉 *SELL SIGNAL*\n\n" \
                      f"*Strategy:* {strategy_name}\n" \
                      f"*Symbol:* {symbol}\n"
            
            if confidence is not None:
                message += f"*Confidence:* {confidence:.2%}\n"
            
            message += f"*Time:* {timestamp}"
        
        else:
            message = f"🔔 *SIGNAL NOTIFICATION*\n\n" \
                      f"*Strategy:* {strategy_name}\n" \
                      f"*Symbol:* {symbol}\n" \
                      f"*Signal:* {signal}\n"
            
            if confidence is not None:
                message += f"*Confidence:* {confidence:.2%}\n"
            
            message += f"*Time:* {timestamp}"
        
        return self.send_message(message)


# Singleton instance
_notifier = None

def get_notifier():
    """
    Get the singleton notifier instance.
    
    Returns:
        TelegramNotifier: Notifier instance
    """
    global _notifier
    
    if _notifier is None:
        _notifier = TelegramNotifier()
    
    return _notifier 