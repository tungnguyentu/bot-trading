import logging
import telegram
from telegram.error import TelegramError
from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramNotifier:
    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.bot = None
        self.logger = logging.getLogger('TelegramNotifier')
        
        if self.enabled:
            if not self.token or not self.chat_id:
                self.logger.warning("Telegram notifications are enabled but token or chat_id is missing!")
                self.enabled = False
            else:
                try:
                    self.bot = telegram.Bot(token=self.token)
                    self.logger.info("Telegram bot initialized successfully")
                    # Send a test message
                    self.send_message("🤖 Trading Bot started and connected to Telegram!")
                except TelegramError as e:
                    self.logger.error(f"Failed to initialize Telegram bot: {e}")
                    self.enabled = False
        else:
            self.logger.info("Telegram notifications are disabled")
    
    def send_message(self, message):
        """Send a message to the specified Telegram chat."""
        if not self.enabled or not self.bot:
            return
        
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return True
        except TelegramError as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def notify_position_open(self, symbol, price, quantity, test_mode=False, leverage_info=""):
        """Send notification about opening a position."""
        mode_tag = "[TEST] " if test_mode else ""
        message = (f"🟢 {mode_tag}<b>Position Opened</b>{leverage_info}\n\n"
                  f"Symbol: {symbol}\n"
                  f"Price: ${price:.2f}\n"
                  f"Quantity: {quantity}\n"
                  f"Total: ${price * quantity:.2f}")
        return self.send_message(message)
    
    def notify_position_close(self, symbol, price, quantity, profit_loss, profit_percentage, reason="", test_mode=False, leverage_info=""):
        """Send notification about closing a position."""
        mode_tag = "[TEST] " if test_mode else ""
        emoji = "🔴" if reason == "stop_loss" else "🔵"
        reason_text = ""
        
        if reason == "take_profit":
            reason_text = "✅ Take Profit"
        elif reason == "stop_loss":
            reason_text = "❌ Stop Loss"
        elif reason == "signal":
            reason_text = "📊 Strategy Signal"
        
        profit_emoji = "💰" if profit_loss > 0 else "📉"
        
        message = (f"{emoji} {mode_tag}<b>Position Closed</b> {reason_text}{leverage_info}\n\n"
                  f"Symbol: {symbol}\n"
                  f"Price: ${price:.2f}\n"
                  f"Quantity: {quantity}\n"
                  f"Total: ${price * quantity:.2f}\n"
                  f"{profit_emoji} P/L: ${profit_loss:.2f} ({profit_percentage:.2f}%)")
        return self.send_message(message)
    
    def notify_signal(self, signal_details, test_mode=False):
        """Send notification about a trading signal with detailed information."""
        mode_tag = "[TEST] " if test_mode else ""
        message = f"{mode_tag}{signal_details}"
        return self.send_message(message)
    
    def notify_error(self, error_message, test_mode=False):
        """Send notification about an error."""
        mode_tag = "[TEST] " if test_mode else ""
        message = f"⚠️ {mode_tag}<b>Error</b>\n\n{error_message}"
        return self.send_message(message)
    
    def notify_info(self, info_message, test_mode=False):
        """Send an informational notification."""
        mode_tag = "[TEST] " if test_mode else ""
        message = f"ℹ️ {mode_tag}<b>Info</b>\n\n{info_message}"
        return self.send_message(message)
