"""
Main entry point for the trading bot.
"""
import logging
import time
import argparse
import schedule
import os
from datetime import datetime

import config
import utils
from exchange import Exchange
from strategy import get_strategy
from backtest import run_backtest

# Set up logging
utils.setup_logging()
logger = logging.getLogger("main")

def run_bot(exchange, strategy_name=None):
    """
    Run the trading bot once.
    
    Args:
        exchange: Exchange instance
        strategy_name (str): Name of the strategy to use
        
    Returns:
        dict: Order information or None
    """
    strategy_name = strategy_name or config.STRATEGY
    
    # Get strategy
    strategy = get_strategy(strategy_name, exchange)
    if strategy is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None
    
    # Run strategy
    logger.info(f"Running strategy: {strategy_name}")
    result = strategy.run()
    
    return result

def schedule_bot(interval=1):
    """
    Schedule the trading bot to run at regular intervals.
    
    Args:
        interval (int): Interval in minutes
    """
    exchange = Exchange()
    
    def job():
        logger.info(f"Running scheduled job at {datetime.now()}")
        run_bot(exchange)
    
    # Schedule the job
    schedule.every(interval).minutes.do(job)
    
    # Run the job immediately
    job()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

def run_backtest_cmd(args):
    """
    Run a backtest.
    
    Args:
        args: Command line arguments
    """
    exchange = Exchange()
    
    # Run backtest
    results = run_backtest(
        exchange, 
        args.strategy, 
        args.start_date, 
        args.end_date
    )
    
    if results:
        # Save results to file
        data_dir = utils.create_data_directory()
        filename = os.path.join(
            data_dir, 
            f"backtest_{args.strategy}_{config.SYMBOL.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        utils.save_to_json(results, filename)
        logger.info(f"Saved backtest results to {filename}")

def run_live_cmd(args):
    """
    Run the bot in live mode.
    
    Args:
        args: Command line arguments
    """
    # Set fixed position size if specified (either via --fixed-size or --invest)
    if hasattr(args, 'invest') and args.invest:
        config.USE_FIXED_POSITION_SIZE = True
        config.FIXED_POSITION_SIZE = args.invest
        logger.info(f"Using investment amount: {args.invest} {args.symbol.split('/')[1]}")
    elif hasattr(args, 'fixed_size') and args.fixed_size:
        config.USE_FIXED_POSITION_SIZE = True
        config.FIXED_POSITION_SIZE = args.fixed_size
        logger.info(f"Using fixed position size: {args.fixed_size} {args.symbol.split('/')[1]}")
    
    # Set leverage if specified
    if args.leverage and args.leverage > 1:
        config.LEVERAGE = args.leverage
        logger.info(f"Using leverage: {args.leverage}x")
    
    if args.schedule:
        schedule_bot(args.interval)
    else:
        exchange = Exchange()
        
        # Set leverage on exchange if specified
        if args.leverage and args.leverage > 1:
            try:
                exchange.set_leverage(args.leverage, args.symbol)
            except Exception as e:
                logger.error(f"Failed to set leverage: {e}")
        
        run_bot(exchange, args.strategy)

def run_paper_cmd(args):
    """
    Run the bot in paper trading mode.
    
    Args:
        args: Command line arguments
    """
    # Set mode to paper
    config.MODE = "paper"
    
    # Set fixed position size if specified (either via --fixed-size or --invest)
    if hasattr(args, 'invest') and args.invest:
        config.USE_FIXED_POSITION_SIZE = True
        config.FIXED_POSITION_SIZE = args.invest
        logger.info(f"Using investment amount: {args.invest} {args.symbol.split('/')[1]}")
    elif hasattr(args, 'fixed_size') and args.fixed_size:
        config.USE_FIXED_POSITION_SIZE = True
        config.FIXED_POSITION_SIZE = args.fixed_size
        logger.info(f"Using fixed position size: {args.fixed_size} {args.symbol.split('/')[1]}")
    
    # Set leverage if specified
    if args.leverage and args.leverage > 1:
        config.LEVERAGE = args.leverage
        logger.info(f"Using leverage: {args.leverage}x")
    
    if args.schedule:
        schedule_bot(args.interval)
    else:
        exchange = Exchange()
        
        # Set leverage on exchange if specified
        if args.leverage and args.leverage > 1:
            try:
                exchange.set_leverage(args.leverage, args.symbol)
            except Exception as e:
                logger.error(f"Failed to set leverage: {e}")
        
        run_bot(exchange, args.strategy)

def train_ml_model_cmd(args):
    """
    Train a machine learning model.
    
    Args:
        args: Command line arguments
    """
    try:
        import ml_models
    except ImportError:
        logger.error("ml_models module not found")
        return
    
    exchange = Exchange()
    
    # Get historical data
    logger.info(f"Fetching historical data for {args.symbol}")
    df = exchange.get_ohlcv(args.symbol, args.timeframe, limit=args.limit)
    
    if df is None or len(df) == 0:
        logger.error("Failed to get historical data")
        return
    
    # Add indicators
    import indicators
    df = indicators.add_all_indicators(df)
    
    # Create model
    if args.model == 'random_forest':
        model = ml_models.RandomForestModel()
    elif args.model == 'xgboost':
        model = ml_models.XGBoostModel()
    elif args.model == 'lstm':
        model = ml_models.LSTMModel()
    elif args.model == 'ensemble':
        model = ml_models.EnsembleModel([
            ml_models.RandomForestModel(),
            ml_models.XGBoostModel(),
            ml_models.LSTMModel()
        ])
    else:
        logger.error(f"Unknown model type: {args.model}")
        return
    
    # Tune hyperparameters if requested
    if args.tune:
        logger.info(f"Tuning hyperparameters for {args.model}")
        tuner = ml_models.HyperparameterTuner(
            args.model, 
            df, 
            target_column=args.target, 
            n_trials=args.trials
        )
        best_params = tuner.tune()
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Create model with best parameters
        if args.model == 'random_forest':
            model = ml_models.RandomForestModel(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None)
            )
        elif args.model == 'xgboost':
            model = ml_models.XGBoostModel(
                n_estimators=best_params.get('n_estimators', 100),
                learning_rate=best_params.get('learning_rate', 0.1),
                max_depth=best_params.get('max_depth', 3)
            )
        elif args.model == 'lstm':
            model = ml_models.LSTMModel(
                units=best_params.get('units', 50),
                dropout=best_params.get('dropout', 0.2),
                batch_size=best_params.get('batch_size', 32)
            )
    
    # Train model
    logger.info(f"Training {args.model} model on {len(df)} samples")
    metrics = model.train(
        df, 
        target_column=args.target, 
        test_size=args.test_size, 
        prediction_horizon=args.horizon
    )
    
    logger.info(f"Training metrics: {metrics}")
    
    # Save model
    model.save_model()
    
    # Save metrics
    data_dir = utils.create_data_directory()
    metrics_file = os.path.join(
        data_dir, 
        f"{args.model}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    utils.save_to_json(metrics, metrics_file)
    logger.info(f"Saved metrics to {metrics_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('--strategy', type=str, default=config.STRATEGY, help='Strategy to use')
    backtest_parser.add_argument('--start-date', type=str, default=config.BACKTEST_START, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, default=config.BACKTEST_END, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--symbol', type=str, default=config.SYMBOL, help='Trading pair symbol')
    backtest_parser.add_argument('--fixed-size', type=float, help='Fixed position size in quote currency (e.g., 50 USDT)')
    backtest_parser.add_argument('--invest', type=float, help='Investment amount in quote currency (e.g., 50 USDT)')
    backtest_parser.add_argument('--leverage', type=float, default=1, help='Leverage to use (e.g., 2 for 2x leverage)')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--strategy', type=str, default=config.STRATEGY, help='Strategy to use')
    live_parser.add_argument('--schedule', action='store_true', help='Schedule the bot to run at intervals')
    live_parser.add_argument('--interval', type=int, default=1, help='Interval in minutes for scheduled runs')
    live_parser.add_argument('--symbol', type=str, default=config.SYMBOL, help='Trading pair symbol')
    live_parser.add_argument('--fixed-size', type=float, help='Fixed position size in quote currency (e.g., 50 USDT)')
    live_parser.add_argument('--invest', type=float, help='Investment amount in quote currency (e.g., 50 USDT)')
    live_parser.add_argument('--leverage', type=float, default=1, help='Leverage to use (e.g., 2 for 2x leverage)')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--strategy', type=str, default=config.STRATEGY, help='Strategy to use')
    paper_parser.add_argument('--schedule', action='store_true', help='Schedule the bot to run at intervals')
    paper_parser.add_argument('--interval', type=int, default=1, help='Interval in minutes for scheduled runs')
    paper_parser.add_argument('--symbol', type=str, default=config.SYMBOL, help='Trading pair symbol')
    paper_parser.add_argument('--fixed-size', type=float, help='Fixed position size in quote currency (e.g., 50 USDT)')
    paper_parser.add_argument('--invest', type=float, help='Investment amount in quote currency (e.g., 50 USDT)')
    paper_parser.add_argument('--leverage', type=float, default=1, help='Leverage to use (e.g., 2 for 2x leverage)')
    
    # Train ML model command
    train_parser = subparsers.add_parser('train', help='Train a machine learning model')
    train_parser.add_argument('--model', type=str, default=config.ML_MODEL, help='Model type (random_forest, xgboost, lstm, ensemble)')
    train_parser.add_argument('--symbol', type=str, default=config.SYMBOL, help='Trading pair symbol')
    train_parser.add_argument('--timeframe', type=str, default=config.TIMEFRAME, help='Timeframe (1m, 5m, 15m, 1h, etc.)')
    train_parser.add_argument('--limit', type=int, default=1000, help='Number of candles to fetch')
    train_parser.add_argument('--target', type=str, default='price_direction', help='Target variable (price_direction, signal)')
    train_parser.add_argument('--horizon', type=int, default=config.ML_PREDICTION_HORIZON, help='Prediction horizon')
    train_parser.add_argument('--test-size', type=float, default=config.ML_TRAIN_TEST_SPLIT, help='Test size')
    train_parser.add_argument('--tune', action='store_true', default=config.ML_HYPERPARAMETER_TUNING, help='Tune hyperparameters')
    train_parser.add_argument('--trials', type=int, default=config.ML_HYPERPARAMETER_TRIALS, help='Number of trials for hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Set symbol if specified
    if hasattr(args, 'symbol') and args.symbol:
        config.SYMBOL = args.symbol
    
    if args.command == 'backtest':
        run_backtest_cmd(args)
    elif args.command == 'live':
        run_live_cmd(args)
    elif args.command == 'paper':
        run_paper_cmd(args)
    elif args.command == 'train':
        train_ml_model_cmd(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 