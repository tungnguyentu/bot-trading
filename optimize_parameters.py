#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import itertools
from datetime import datetime, timedelta
import logging
import json
import sys
import os
from binance.client import Client

from config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOL
from backtester import Backtester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimizer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Optimizer')

def find_optimal_parameters(symbol=TRADING_SYMBOL, interval='15m', days=30):
    """Find optimal parameters for the trading strategy."""
    logger.info(f"Optimizing parameters for {symbol} over the past {days} days")
    
    # Parameter ranges to test - adjusted for better win rate
    ema_short_range = [5, 7, 9, 12]
    ema_long_range = [21, 25, 30, 35]
    rsi_period_range = [14, 21]
    rsi_oversold_range = [30, 35]
    rsi_overbought_range = [65, 70, 75]
    take_profit_range = [1.5, 2.0, 2.5]
    stop_loss_range = [0.75, 1.0, 1.25]
    
    # Create parameter combinations (be careful with the number of combinations!)
    param_combinations = list(itertools.product(
        ema_short_range,
        ema_long_range,
        rsi_period_range,
        rsi_oversold_range,
        rsi_overbought_range,
        take_profit_range,
        stop_loss_range
    ))
    
    # Filter out invalid combinations (ensure short EMA < long EMA)
    param_combinations = [p for p in param_combinations if p[0] < p[1]]
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Set time period for backtesting
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Test each parameter combination
    results = []
    for i, params in enumerate(param_combinations):
        ema_short, ema_long, rsi_period, rsi_oversold, rsi_overbought, take_profit, stop_loss = params
        
        # Skip if we've already run this combination (in case script was interrupted)
        result_file = f"results/optimize_{symbol}_{ema_short}_{ema_long}_{rsi_period}_{rsi_oversold}_{rsi_overbought}_{take_profit}_{stop_loss}.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
            logger.info(f"Loaded existing result for parameter set {i+1}/{len(param_combinations)}: {params}")
            continue
        
        logger.info(f"Testing parameter set {i+1}/{len(param_combinations)}: {params}")
        
        # Override configuration
        import config
        config.EMA_SHORT = ema_short
        config.EMA_LONG = ema_long
        config.RSI_PERIOD = rsi_period
        config.RSI_OVERSOLD = rsi_oversold
        config.RSI_OVERBOUGHT = rsi_overbought
        config.TAKE_PROFIT_PCT = take_profit
        config.STOP_LOSS_PCT = stop_loss
        
        # Run backtest
        backtester = Backtester(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            backtest_result = backtester.run_backtest()
            
            if not backtest_result:
                logger.warning(f"No results for parameter set: {params}")
                continue
                
            # Calculate a score that prioritizes win rate and profit factor
            win_rate = backtest_result['win_rate']
            profit_factor = backtest_result['profit_factor']
            return_pct = backtest_result['return_pct']
            total_trades = backtest_result['total_trades']
            
            # Penalize strategies with too few trades
            trade_score = min(1.0, total_trades / 50) if total_trades > 0 else 0
            
            # Score calculation focusing on win rate and consistency
            score = (win_rate * 0.5) + (return_pct * 0.2) + (min(profit_factor, 3) * 10) + (trade_score * 20)
                
            result = {
                'parameters': {
                    'ema_short': ema_short,
                    'ema_long': ema_long,
                    'rsi_period': rsi_period,
                    'rsi_oversold': rsi_oversold,
                    'rsi_overbought': rsi_overbought,
                    'take_profit_pct': take_profit,
                    'stop_loss_pct': stop_loss
                },
                'performance': {
                    'return_pct': backtest_result['return_pct'],
                    'total_trades': backtest_result['total_trades'],
                    'win_rate': backtest_result['win_rate'],
                    'profit_factor': backtest_result['profit_factor'],
                    'score': score
                }
            }
            
            # Save individual result to file
            os.makedirs("results", exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error testing parameters {params}: {e}")
    
    if not results:
        logger.error("No valid results found.")
        return None
    
    # Sort by score instead of return
    results.sort(key=lambda x: x['performance']['score'], reverse=True)
    
    # Save all results to a single file
    with open(f"results/all_results_{symbol}_{days}days.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Return top performing parameters
    top_result = results[0]
    logger.info(f"Optimal parameters found: {top_result['parameters']}")
    logger.info(f"Performance: Return: {top_result['performance']['return_pct']:.2f}%, Win Rate: {top_result['performance']['win_rate']:.2f}%, Profit Factor: {top_result['performance']['profit_factor']:.2f}")
    
    return top_result

def main():
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--symbol', default=TRADING_SYMBOL, help='Trading symbol to optimize for')
    parser.add_argument('--interval', default='15m', help='Candle interval')
    parser.add_argument('--days', type=int, default=30, help='Number of days for backtesting')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Find optimal parameters
    optimal_params = find_optimal_parameters(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days
    )
    
    if optimal_params:
        print("\n===== OPTIMAL PARAMETERS =====")
        print(f"Symbol: {args.symbol}")
        print(f"Interval: {args.interval}")
        print(f"Period: {args.days} days")
        print("----------------------------")
        print(f"EMA Short: {optimal_params['parameters']['ema_short']}")
        print(f"EMA Long: {optimal_params['parameters']['ema_long']}")
        print(f"RSI Period: {optimal_params['parameters']['rsi_period']}")
        print(f"RSI Oversold: {optimal_params['parameters']['rsi_oversold']}")
        print(f"RSI Overbought: {optimal_params['parameters']['rsi_overbought']}")
        print(f"Take Profit: {optimal_params['parameters']['take_profit_pct']}%")
        print(f"Stop Loss: {optimal_params['parameters']['stop_loss_pct']}%")
        print("----------------------------")
        print(f"Return: {optimal_params['performance']['return_pct']:.2f}%")
        print(f"Total Trades: {optimal_params['performance']['total_trades']}")
        print(f"Win Rate: {optimal_params['performance']['win_rate']:.2f}%")
        print(f"Profit Factor: {optimal_params['performance']['profit_factor']:.2f}")
        print("=============================")

if __name__ == "__main__":
    main()
