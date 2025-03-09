#!/usr/bin/env python3

import time
import traceback
import logging
import argparse
import threading
import os
import sys
import signal
from binance_bot import BinanceBot
from config import (TEST_MODE, EMA_SHORT, EMA_LONG, RSI_PERIOD, 
                  RSI_OVERBOUGHT, RSI_OVERSOLD, USE_LEVERAGE, 
                  LEVERAGE, FUTURES_MODE, DASHBOARD_ENABLED,
                  DASHBOARD_PORT)

# Import dashboard conditionally to avoid errors if dependencies missing
try:
    from dashboard import run_dashboard
    dashboard_available = True
except ImportError:
    dashboard_available = False

# Global variable to hold the bot instance
bot = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global bot
    logger = logging.getLogger()
    logger.info("Shutdown signal received. Exiting gracefully...")
    sys.exit(0)

def start_dashboard(port=DASHBOARD_PORT):
    """Start the dashboard in a separate thread."""
    if dashboard_available:
        # Pass in_thread=True to avoid signal handling issues
        dashboard_thread = threading.Thread(
            target=run_dashboard,
            kwargs={'in_thread': True}
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        return dashboard_thread
    else:
        logger = logging.getLogger()
        logger.warning("Dashboard dependencies not available. Dashboard not started.")
        return None

def main():
    """Main entry point for the trading bot."""
    # Set up argparse for command line options
    parser = argparse.ArgumentParser(description='Binance Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in test mode (override config)')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode (override config)')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard (override config)')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help=f'Dashboard port (default: {DASHBOARD_PORT})')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--days', type=int, default=30, help='Number of days for backtesting (default: 30)')
    # Add symbol argument to the parser
    parser.add_argument('--symbol', default=None, help='Trading symbol to use (override config)')
    parser.add_argument('--ml', action='store_true', help='Use machine learning for signal enhancement')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.info("Starting Binance Trading Bot")
    
    # Handle test mode vs live mode (command line args override config)
    test_mode = TEST_MODE
    if args.test:
        test_mode = True
        logger.info("Command line flag: Running in TEST MODE")
    elif args.live:
        test_mode = False
        logger.info("Command line flag: Running in LIVE TRADING MODE")
    else:
        logger.info("Using configuration setting for test/live mode")
        
    # Log trading mode information
    if test_mode:
        logger.info("Running in TEST MODE - No real trades will be executed")
    else:
        logger.warning("Running in LIVE TRADING MODE - REAL trades will be executed!")
        # Prompt for confirmation when running live
        confirm = input("You are about to start LIVE trading. Type 'YES' to confirm: ")
        if confirm.upper() != 'YES':
            logger.info("Live trading canceled by user")
            return
    
    # Handle backtesting
    if args.backtest:
        from backtester import Backtester
        from datetime import datetime, timedelta
        import os
        import config
        
        # Override QUANTITY to use a larger value for backtesting to see the effects more clearly
        original_quantity = config.QUANTITY
        
        # Use symbol from args if provided, otherwise use config
        symbol = args.symbol if args.symbol else config.TRADING_SYMBOL
            
        # Set quantity based on symbol for backtesting
        backtest_quantities = {
            'BTCUSDT': 0.01,  # About $650
            'ETHUSDT': 0.1,   # About $300
            'SOLUSDT': 5.0,   # About $600
            'DOGEUSDT': 10000,# About $700
            'DEFAULT': 1.0
        }
        
        backtest_quantity = backtest_quantities.get(symbol, backtest_quantities.get('DEFAULT'))
        logger.info(f"Setting backtest quantity for {symbol} to {backtest_quantity}")
        config.QUANTITY = backtest_quantity
        
        # Use very relaxed RSI thresholds for backtesting
        original_rsi_overbought = config.RSI_OVERBOUGHT
        original_rsi_oversold = config.RSI_OVERSOLD
        config.RSI_OVERBOUGHT = 60  # Much more sensitive than 70
        config.RSI_OVERSOLD = 40    # Much more sensitive than 30
        
        logger.info(f"Using relaxed RSI thresholds for backtesting: Oversold={config.RSI_OVERSOLD}, Overbought={config.RSI_OVERBOUGHT}")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        
        logger.info(f"Running backtesting from {start_date} to {end_date}")
        backtester = Backtester(start_date=start_date, end_date=end_date, symbol=symbol)
        
        # Pass ML flag to backtester
        results = backtester.run_backtest(use_ml=args.ml)
        
        if results:
            backtester.print_performance_summary(results)
            backtester.plot_results(results)
        
            # Generate detailed trade analysis report
            logger.info("Generating trade analysis report...")
            trades = results.get('trades', [])
            if trades:
                buy_trades = [t for t in trades if t['type'] == 'buy']
                sell_trades = [t for t in trades if t['type'] == 'sell']
                
                print("\n===== DETAILED TRADE ANALYSIS =====")
                print(f"Total Buy Trades: {len(buy_trades)}")
                print(f"Total Sell Trades: {len(sell_trades)}")
                
                # Analyze sell reasons
                if sell_trades:
                    reasons = {}
                    for trade in sell_trades:
                        reason = trade.get('exit_reason', 'unknown')
                        reasons[reason] = reasons.get(reason, 0) + 1
                    
                    print("\nExit Reasons:")
                    for reason, count in reasons.items():
                        percentage = (count / len(sell_trades)) * 100
                        print(f"  - {reason}: {count} trades ({percentage:.1f}%)")
                    
                    # Average profit by exit reason
                    profit_by_reason = {}
                    for trade in sell_trades:
                        reason = trade.get('exit_reason', 'unknown')
                        if reason not in profit_by_reason:
                            profit_by_reason[reason] = []
                        profit_by_reason[reason].append(trade.get('profit_percentage', 0))
                    
                    print("\nAverage Profit by Exit Reason:")
                    for reason, profits in profit_by_reason.items():
                        avg_profit = sum(profits) / len(profits)
                        print(f"  - {reason}: {avg_profit:.2f}%")
                
                # Add additional statistics
                print("\nProfitability by Trade Number:")
                trade_counts = list(range(1, 11))  # Track first 10 trades
                profits_by_trade_count = {i: [] for i in trade_counts}
                
                # Group trades by sequences
                current_sequence = 1
                for i, trade in enumerate(sell_trades):
                    if current_sequence <= 10:  # Only track first 10
                        profits_by_trade_count[current_sequence].append(trade.get('profit_percentage', 0))
                    current_sequence += 1
                    if current_sequence > 10:
                        current_sequence = 1
                
                # Print average profit by trade number
                for count in trade_counts:
                    profits = profits_by_trade_count[count]
                    if profits:
                        avg_profit = sum(profits) / len(profits)
                        print(f"  Trade #{count}: {avg_profit:.2f}% ({len(profits)} trades)")
                
                # Print longest winning and losing streaks
                winning_streak = 0
                losing_streak = 0
                current_win_streak = 0
                current_lose_streak = 0
                
                for trade in sell_trades:
                    if trade.get('profit_percentage', 0) > 0:
                        current_win_streak += 1
                        current_lose_streak = 0
                        winning_streak = max(winning_streak, current_win_streak)
                    else:
                        current_lose_streak += 1
                        current_win_streak = 0
                        losing_streak = max(losing_streak, current_lose_streak)
                
                print(f"\nLongest Win Streak: {winning_streak}")
                print(f"Longest Losing Streak: {losing_streak}")
            else:
                print("\nNo trades were executed during the backtesting period.")
        
        # Restore original values
        config.QUANTITY = original_quantity
        config.RSI_OVERBOUGHT = original_rsi_overbought
        config.RSI_OVERSOLD = original_rsi_oversold
        
        return
    
    # Handle dashboard startup
    start_dashboard_flag = DASHBOARD_ENABLED
    if args.dashboard:
        start_dashboard_flag = True
        
    if start_dashboard_flag:
        logger.info(f"Starting dashboard on port {args.port}")
        dashboard_thread = start_dashboard(port=args.port)
        if dashboard_thread is None:
            logger.warning("Could not start dashboard. Install required packages with 'pip install dash plotly'")
    
    # Log strategy parameters
    if FUTURES_MODE:
        logger.info(f"Trading on Binance Futures")
        if USE_LEVERAGE:
            logger.info(f"Using {LEVERAGE}x leverage")
    else:
        logger.info("Trading on Binance Spot")
    
    logger.info(f"Strategy Parameters: EMA{EMA_SHORT}/{EMA_LONG}, RSI{RSI_PERIOD} (O/B:{RSI_OVERBOUGHT}, O/S:{RSI_OVERSOLD})")
    
    # Register signal handlers - only in the main thread
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the bot
    try:
        global bot
        # Subclass to override the test_mode property
        class ConfigOverrideBot(BinanceBot):
            def __init__(self, override_test_mode):
                self.override_test_mode = override_test_mode
                super().__init__()
                
            @property
            def test_mode(self):
                return self.override_test_mode
                
            @test_mode.setter
            def test_mode(self, value):
                self._test_mode = self.override_test_mode
        
        bot = ConfigOverrideBot(test_mode)
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
