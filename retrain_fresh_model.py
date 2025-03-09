#!/usr/bin/env python3

import argparse
import logging
import warnings
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from backtester import Backtester
from config import TRADING_SYMBOL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelRetrainer')

def prepare_features(df):
    """Prepare features for ML model - simplified version to avoid dependency issues."""
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Basic features
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Technical indicators - using only the ones provided in the dataframe
    # RSI
    if 'rsi' in data.columns:
        data['rsi_change'] = data['rsi'].diff()
    else:
        # Calculate RSI if not available
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_change'] = data['rsi'].diff()
    
    # MACD
    if 'macd' in data.columns and 'macd_signal' in data.columns:
        data['macd_histogram'] = data['macd'] - data['macd_signal']
    else:
        # Calculate MACD if not available
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # EMA ratios
    for window in [9, 21]:
        col_name = f'ema_{window}'
        if col_name not in data.columns:
            data[col_name] = data['close'].ewm(span=window, adjust=False).mean()
    
    data['ema_ratio'] = data['ema_9'] / data['ema_21']
    
    # Forward returns for target
    data['forward_return_5'] = data['close'].shift(-5) / data['close'] - 1
    data['target_5'] = (data['forward_return_5'] > 0).astype(int)
    
    # Clean up NaN values
    data = data.dropna()
    
    return data

def train_simple_model(df, test_size=0.2, model_dir='models', symbol=TRADING_SYMBOL):
    """Train a simplified model without complex dependencies."""
    logger.info("Preparing data...")
    
    # Prepare data
    prepared_data = prepare_features(df)
    
    # Define features and target
    features = [
        'returns', 'volume_change', 'rsi', 'rsi_change',
        'macd', 'macd_signal', 'macd_histogram', 'ema_ratio'
    ]
    
    # Filter to only use available features
    features = [f for f in features if f in prepared_data.columns]
    
    X = prepared_data[features]
    y = prepared_data['target_5']
    
    # Split data
    logger.info(f"Splitting data with test size {test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for time series data
    )
    
    # Create a simpler pipeline with just one classifier to avoid compatibility issues
    logger.info("Building model pipeline...")
    model = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])
    
    # Train the model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Model performance: {metrics}")
    
    # Create model directory if needed
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_file = os.path.join(model_dir, f'{symbol}_model.pkl')
    logger.info(f"Saving model to {model_file}")
    joblib.dump(model, model_file)
    
    # Save sklearn version info
    import sklearn
    version_file = os.path.join(model_dir, f'{symbol}_model_version.txt')
    with open(version_file, 'w') as f:
        f.write(sklearn.__version__)
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Retrain a new ML model from scratch')
    parser.add_argument('--symbol', default=TRADING_SYMBOL, help=f'Symbol to train for (default: {TRADING_SYMBOL})')
    parser.add_argument('--days', type=int, default=180, help='Number of days of historical data (default: 180)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    # Check if model already exists
    model_file = os.path.join('models', f'{args.symbol}_model.pkl')
    if os.path.exists(model_file) and not args.force:
        logger.warning(f"Model already exists at {model_file}. Use --force to retrain.")
        return
    
    # Get historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching historical data for {args.symbol} from {start_date} to {end_date}")
    backtester = Backtester(symbol=args.symbol, start_date=start_date, end_date=end_date)
    df = backtester.get_historical_data()
    
    if df.empty:
        logger.error("No historical data available for training")
        return
    
    # Calculate technical indicators
    logger.info("Calculating indicators...")
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
    
    # Train model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, metrics = train_simple_model(
            df, 
            test_size=args.test_size,
            symbol=args.symbol
        )
    
    # Print results
    print("\n===== MODEL TRAINING RESULTS =====")
    print(f"Symbol: {args.symbol}")
    print(f"Training data period: {args.days} days")
    print(f"Test set size: {args.test_size * 100}%")
    print("--------------------------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("===================================")

if __name__ == "__main__":
    main()
