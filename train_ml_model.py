#!/usr/bin/env python3

import argparse
import logging
import warnings  # Add missing import
from sklearn.exceptions import ConvergenceWarning  # Add additional import here
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_strategy import MLStrategy
from backtester import Backtester
from config import TRADING_SYMBOL
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ML-Trainer')

def train_model(symbol=TRADING_SYMBOL, days=180, test_size=0.2, target_horizon=5, param_search=False):
    """Train ML model on historical data"""
    logger.info(f"Training ML model for {symbol} using {days} days of historical data")
    logger.info(f"Target horizon: {target_horizon} periods, Parameter search: {param_search}")
    
    # Get historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    backtester = Backtester(symbol=symbol, start_date=start_date, end_date=end_date)
    df = backtester.get_historical_data()
    
    if df.empty:
        logger.error("No historical data available for training")
        return None
    
    # Add technical indicators - standard indicators will be enhanced in the ML strategy
    logger.info(f"Calculating indicators for {len(df)} data points...")
    
    df[f'ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df[f'ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Calculate price change
    df['price_change'] = df['close'].pct_change(5)
    
    # Create output directory for charts if needed
    os.makedirs("models", exist_ok=True)
    
    # Filter out extreme values and ensure data quality
    logger.info("Pre-processing data and removing extreme values...")
    df = df[(df['close'] > 0) & (df['volume'] > 0)]
    
    # Filter out extreme price movements
    returns = df['close'].pct_change()
    mean_returns = returns.mean()
    std_returns = returns.std()
    df = df[(returns <= mean_returns + 5*std_returns) & 
            (returns >= mean_returns - 5*std_returns)]
    
    logger.info(f"Data after cleaning: {len(df)} points")
    
    # Train the model
    try:
        ml_strategy = MLStrategy(symbol=symbol)
        
        # If in parameter search mode, print a warning about duration
        if param_search:
            logger.info("Parameter search can take 30+ minutes, please be patient...")
            
        logger.info("Training model, this may take a few minutes...")
        with warnings.catch_warnings():
            if not param_search:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            metrics = ml_strategy.train_model(
                df, 
                target_horizon=target_horizon, 
                test_size=test_size,
                param_search=param_search
            )
        
        logger.info(f"Model trained successfully with metrics: {metrics}")
        logger.info(f"Model saved to: models/{symbol}_model.pkl")
        
        # Generate validation chart
        generate_validation_chart(df, ml_strategy, symbol, target_horizon)
        
        return metrics
    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_validation_chart(df, ml_strategy, symbol, target_horizon):
    """Generate a validation chart showing model predictions vs actual price movements"""
    try:
        # Get the last 100 data points for visualization
        test_df = df.iloc[-100:].copy()
        
        # Get model predictions - need to ensure NaN values are handled
        prepared_data = ml_strategy.prepare_features(test_df)
        
        if prepared_data.empty:
            logger.warning("No data after feature preparation for validation chart")
            return
            
        # The feature preparation might have reduced the number of rows due to NaN values
        # We need to align our test data with the prepared data
        probabilities = ml_strategy.predict(test_df)
        
        if probabilities is not None and len(probabilities) > 0:
            # Handle mismatched lengths by aligning the data
            if len(probabilities) != len(test_df):
                logger.info(f"Aligning datasets - probabilities length: {len(probabilities)}, test_df length: {len(test_df)}")
                # Use only the last len(probabilities) rows of the test DataFrame
                test_df = test_df.iloc[-len(probabilities):].copy()
                logger.info(f"Adjusted test_df length to: {len(test_df)}")
            
            # Add predictions to DataFrame
            test_df['predicted_probability'] = probabilities
            
            # Calculate actual forward returns - handle missing data at the end
            test_df[f'actual_return'] = test_df['close'].shift(-target_horizon) / test_df['close'] - 1
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Price chart
            ax1.plot(test_df.index, test_df['close'], label='Price')
            
            # Color the background based on prediction
            for i in range(len(test_df)-1):
                if probabilities[i] > 0.6:  # Strong buy signal
                    ax1.axvspan(test_df.index[i], test_df.index[i+1], alpha=0.2, color='green')
                elif probabilities[i] < 0.4:  # Strong sell signal
                    ax1.axvspan(test_df.index[i], test_df.index[i+1], alpha=0.2, color='red')
            
            ax1.set_title(f'{symbol} Price with ML Predictions')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            
            # Prediction probability chart
            ax2.plot(test_df.index, probabilities, color='blue', label='Bullish Probability')
            ax2.axhline(y=0.5, color='gray', linestyle='--')
            ax2.axhline(y=0.6, color='green', linestyle='--')
            ax2.axhline(y=0.4, color='red', linestyle='--')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Probability')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"models/{symbol}_validation_chart.png")
            plt.close()
            
            logger.info(f"Validation chart saved to models/{symbol}_validation_chart.png")
    except Exception as e:
        logger.error(f"Error generating validation chart: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Train ML model for trading')
    parser.add_argument('--symbol', default=TRADING_SYMBOL, help=f'Symbol to train for (default: {TRADING_SYMBOL})')
    parser.add_argument('--days', type=int, default=180, help='Number of days of historical data to use (default: 180)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--target-horizon', type=int, default=5, help='Forward periods for prediction target (default: 5)')
    parser.add_argument('--param-search', action='store_true', help='Perform hyperparameter search')
    
    args = parser.parse_args()
    
    metrics = train_model(
        symbol=args.symbol, 
        days=args.days, 
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        param_search=args.param_search
    )
    
    if metrics:
        print("\n===== MODEL TRAINING RESULTS =====")
        print(f"Symbol: {args.symbol}")
        print(f"Training data period: {args.days} days")
        print(f"Target horizon: {args.target_horizon} periods")
        print(f"Test set size: {args.test_size * 100}%")
        print("--------------------------------")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics.get('roc_auc', 0):.4f}")
        print("===================================")
        print("\nNext steps:")
        print("1. Check the feature importance plot in the models directory")
        print("2. Review the validation chart to see model predictions")
        print("3. Run backtesting with ML: python main.py --backtest --ml")
        print("4. Try different target horizons for different timeframes")

if __name__ == "__main__":
    main()
