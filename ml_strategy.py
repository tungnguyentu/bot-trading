import pandas as pd
import numpy as np
import joblib
import os
import logging
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# Add these imports for version checking
from pkg_resources import parse_version
import sklearn

class MLStrategy:
    """Machine Learning based trading strategy"""
    
    def __init__(self, symbol, model_dir='models'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger('MLStrategy')
        self.model_file = os.path.join(model_dir, f'{symbol}_model.pkl')
        self.feature_importance_file = os.path.join(model_dir, f'{symbol}_feature_importance.png')
        self.model_version_file = os.path.join(model_dir, f'{symbol}_model_version.txt')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def prepare_features(self, df):
        """
        Prepare features for the machine learning model with advanced indicators.
        Returns a DataFrame with enhanced technical indicator features.
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Basic price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Multiple volatility features
        for window in [5, 10, 20]:
            data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
        data['normalized_range'] = (data['high'] - data['low']) / data['close']
        data['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Enhanced volume features
        data['volume_change'] = data['volume'].pct_change()
        for window in [5, 10, 20]:
            data[f'volume_ma_ratio_{window}'] = data['volume'] / data['volume'].rolling(window=window).mean()
        data['volume_price_trend'] = data['volume'] * data['returns']
        
        # Advanced trend features
        for window in [5, 10, 20, 50]:
            data[f'price_sma_ratio_{window}'] = data['close'] / data['close'].rolling(window=window).mean()
            data[f'price_ema_ratio_{window}'] = data['close'] / data['close'].ewm(span=window, adjust=False).mean()
        
        # RSI and derivatives at multiple timeframes
        for window in [7, 14, 21]:
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            # RSI derivatives
            data[f'rsi_{window}_change'] = data[f'rsi_{window}'].diff()
            data[f'rsi_{window}_ma_diff'] = data[f'rsi_{window}'] - data[f'rsi_{window}'].rolling(window=window).mean()
        
        # Cross-RSI features
        if 'rsi_7' in data.columns and 'rsi_14' in data.columns:
            data['rsi_cross'] = data['rsi_7'] - data['rsi_14']
            data['rsi_cross_change'] = data['rsi_cross'].diff()
        
        # MACD features with multiple settings
        for (fast, slow, signal) in [(12, 26, 9), (5, 35, 5)]:
            prefix = f'macd_{fast}_{slow}_{signal}'
            ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
            data[f'{prefix}_line'] = ema_fast - ema_slow
            data[f'{prefix}_signal'] = data[f'{prefix}_line'].ewm(span=signal, adjust=False).mean()
            data[f'{prefix}_histogram'] = data[f'{prefix}_line'] - data[f'{prefix}_signal']
            data[f'{prefix}_histogram_change'] = data[f'{prefix}_histogram'].diff()
        
        # Price pattern features
        for window in [3, 5, 7]:
            # Local highs and lows
            data[f'local_high_{window}'] = data['high'].rolling(window=window).max() == data['high']
            data[f'local_low_{window}'] = data['low'].rolling(window=window).min() == data['low']
            
            # Rolling Z-score (helps detect mean reversion opportunities)
            rolling_mean = data['close'].rolling(window=window).mean()
            rolling_std = data['close'].rolling(window=window).std()
            data[f'zscore_{window}'] = (data['close'] - rolling_mean) / rolling_std
        
        # Advanced momentum indicators
        # Rate of Change
        for window in [5, 10, 20]:
            data[f'roc_{window}'] = (data['close'] / data['close'].shift(window) - 1) * 100
        
        # Bollinger Bands
        for window in [20, 50]:
            rolling_mean = data['close'].rolling(window=window).mean()
            rolling_std = data['close'].rolling(window=window).std()
            data[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            data[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            data[f'bb_width_{window}'] = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']) / rolling_mean
            data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'])
        
        # Stochastic Oscillator
        for window in [14, 21]:
            low_min = data['low'].rolling(window=window).min()
            high_max = data['high'].rolling(window=window).max()
            data[f'k_{window}'] = ((data['close'] - low_min) / (high_max - low_min)) * 100
            data[f'd_{window}'] = data[f'k_{window}'].rolling(window=3).mean()
        
        # Advanced feature combinations
        if 'rsi' in data.columns and 'k_14' in data.columns:
            data['rsi_stoch_diff'] = data['rsi'] - data['k_14']
        
        # Create target for training (1 if price goes up in next periods, 0 otherwise)
        # Forward returns for different time horizons
        for period in [1, 3, 5, 8, 13, 21]:  # Fibonacci sequence for timeframes
            data[f'forward_return_{period}'] = data['close'].shift(-period) / data['close'] - 1
            # Binary targets based on different thresholds
            data[f'target_{period}'] = (data[f'forward_return_{period}'] > 0).astype(int)
            # Add stronger signals (significant moves) as separate targets
            data[f'target_strong_up_{period}'] = (data[f'forward_return_{period}'] > 0.01).astype(int)  # 1% threshold
            data[f'target_strong_down_{period}'] = (data[f'forward_return_{period}'] < -0.01).astype(int)  # -1% threshold
        
        # Clean up NaN values
        data = data.dropna()
        
        return data
    
    def create_training_data(self, data, target_horizon=5, include_target_features=False):
        """
        Create training features and target from the prepared data.
        
        Args:
            data: Prepared DataFrame with features
            target_horizon: Which forward period to use (default: 5)
            include_target_features: Whether to include target-derived features
        """
        # Define the features to use - focusing on the most predictive ones
        basic_features = [
            'returns', 'log_returns', 'volatility_10', 'volatility_20', 'normalized_range',
            'close_to_high', 'volume_change', 'volume_ma_ratio_10', 'volume_price_trend'
        ]
        
        trend_features = [
            'price_sma_ratio_10', 'price_sma_ratio_20', 'price_ema_ratio_10', 'price_ema_ratio_20'
        ]
        
        oscillator_features = [
            'rsi_7', 'rsi_14', 'rsi_7_change', 'rsi_14_change', 'rsi_cross',
            'k_14', 'd_14', 'rsi_stoch_diff'
        ]
        
        macd_features = [
            'macd_12_26_9_line', 'macd_12_26_9_signal', 'macd_12_26_9_histogram',
            'macd_5_35_5_histogram', 'macd_12_26_9_histogram_change'
        ]
        
        pattern_features = [
            'local_high_5', 'local_low_5', 'zscore_5', 'roc_10', 
            'bb_position_20', 'bb_width_20'
        ]
        
        # Combine all features
        feature_columns = (basic_features + trend_features + oscillator_features + 
                          macd_features + pattern_features)
        
        # Filter to only include columns that exist in the data
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        # Choose which target to predict
        target_column = f'target_{target_horizon}'
        
        # Select features and target
        X = data[feature_columns]
        y = data[target_column]
        
        # Return features, target, and the list of feature names
        return X, y, feature_columns
    
    def train_model(self, df, target_horizon=5, test_size=0.2, param_search=False):
        """
        Train a machine learning model with advanced options.
        
        Args:
            df: DataFrame with price data
            target_horizon: Period for forward returns (default: 5)
            test_size: Proportion of data for testing (default: 0.2)
            param_search: Whether to perform parameter search (default: False)
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare data
        prepared_data = self.prepare_features(df)
        X, y, feature_names = self.create_training_data(prepared_data, target_horizon)
        
        # Use time series split to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # No shuffle for time series data
        )
        
        # Create ensemble of models for better accuracy
        if not param_search:
            # Simple pipeline with default parameters
            self.model = Pipeline([
                ('scaler', RobustScaler()),  # RobustScaler is less sensitive to outliers
                ('classifier', VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(
                            n_estimators=100, 
                            max_depth=5,
                            random_state=42,
                            class_weight='balanced'
                        )),
                        ('gb', GradientBoostingClassifier(
                            n_estimators=100,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42
                        )),
                        ('mlp', MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation='relu',
                            alpha=0.001,  # Increased regularization
                            learning_rate_init=0.002,  # Adjusted learning rate
                            max_iter=500,  # Increased maximum iterations
                            early_stopping=True,  # Enable early stopping
                            validation_fraction=0.1,  # Use 10% of training data for validation
                            n_iter_no_change=20,  # Stop if no improvement after 20 iterations
                            solver='adam',  # Use adam optimizer
                            random_state=42
                        ))
                    ],
                    voting='soft'  # Use probability estimates for voting
                ))
            ])
        else:
            # Use parameter search to find optimal model configuration
            self.logger.info("Performing parameter search for optimal model...")
            
            # Define the parameter grid to search
            param_grid = {
                'classifier__rf__n_estimators': [50, 100],
                'classifier__rf__max_depth': [3, 5, 7],
                'classifier__gb__learning_rate': [0.01, 0.05, 0.1],
                'classifier__gb__max_depth': [2, 3, 4],
                'classifier__mlp__alpha': [0.0001, 0.001, 0.01],
                'classifier__mlp__learning_rate_init': [0.001, 0.01],
                'classifier__mlp__max_iter': [500, 1000]
            }
            
            # Create base pipeline with better MLP configuration
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
                        ('gb', GradientBoostingClassifier(random_state=42)),
                        ('mlp', MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation='relu',
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=20,
                            solver='adam',
                            random_state=42
                        ))
                    ],
                    voting='soft'
                ))
            ])
            
            # Set up grid search with time series cross-validation
            grid_search = GridSearchCV(
                pipeline, 
                param_grid=param_grid,
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1  # Add verbosity to see progress
            )
            
            # Perform the search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Train the model
        self.logger.info("Fitting final model...")
        self.model.fit(X_train, y_train)
        
        # Save sklearn version info with the model
        current_sklearn_version = sklearn.__version__
        try:
            with open(self.model_version_file, 'w') as f:
                f.write(current_sklearn_version)
            self.logger.info(f"Saved model with sklearn version: {current_sklearn_version}")
        except Exception as e:
            self.logger.warning(f"Could not save model version info: {e}")
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Feature importance analysis (if available)
        self._plot_feature_importance(X, feature_names)
        
        # Save the model
        self.save_model()
        
        return metrics
    
    def _plot_feature_importance(self, X, feature_names):
        """Plot feature importance if the model supports it."""
        try:
            # Get the classifier from the pipeline
            classifier = self.model.named_steps['classifier']
            
            # For voting classifier, use the first estimator (Random Forest)
            if isinstance(classifier, VotingClassifier):
                feature_importances = classifier.estimators_[0].feature_importances_
            elif hasattr(classifier, 'feature_importances_'):
                feature_importances = classifier.feature_importances_
            else:
                # Skip if model doesn't support feature_importances_
                return
                
            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            # Plot the top 15 most important features
            top_n = min(15, len(importance_df))
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'][:top_n][::-1], importance_df['Importance'][:top_n][::-1])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            plt.savefig(self.feature_importance_file)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {self.feature_importance_file}")
        except Exception as e:
            self.logger.warning(f"Could not generate feature importance plot: {e}")
    
    def predict(self, df):
        """
        Make predictions on new data.
        Returns probability of price going up.
        """
        if self.model is None:
            self.logger.warning("No model loaded. Please train or load a model first.")
            return None
        
        # Prepare features
        prepared_data = self.prepare_features(df)
        
        # If data is empty after preparation
        if prepared_data.empty:
            self.logger.warning("No data after feature preparation")
            return None
        
        # Get features
        X, _, _ = self.create_training_data(prepared_data)
        
        # Predict probability of price going up
        probabilities = self.model.predict_proba(X)
        
        # Return probability of class 1 (price going up)
        return probabilities[:, 1]
    
    def get_trading_signal(self, df, threshold=0.6):
        """
        Get trading signal based on model predictions.
        Returns 1 for buy, -1 for sell, 0 for hold.
        
        Args:
            df: DataFrame with price data
            threshold: Probability threshold for signals (default: 0.6)
        """
        if self.model is None:
            self.logger.warning("No model loaded. Please train or load a model first.")
            return 0
        
        probabilities = self.predict(df)
        
        if probabilities is None or len(probabilities) == 0:
            return 0
        
        # Get the latest prediction
        latest_prob = probabilities[-1]
        
        # Generate signals based on probability
        if latest_prob > threshold:  # Strong up prediction
            return 1  # Buy
        elif latest_prob < (1 - threshold):  # Strong down prediction
            return -1  # Sell
        else:
            return 0  # Hold
    
    def save_model(self):
        """Save model to disk"""
        if self.model is not None:
            try:
                joblib.dump(self.model, self.model_file)
                self.logger.info(f"Model saved to {self.model_file}")
                
                # Save current sklearn version
                current_sklearn_version = sklearn.__version__
                with open(self.model_version_file, 'w') as f:
                    f.write(current_sklearn_version)
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model from disk with version compatibility check"""
        try:
            if os.path.exists(self.model_file):
                # Check if version info exists and if it's compatible
                current_version = sklearn.__version__
                saved_version = None
                
                if os.path.exists(self.model_version_file):
                    with open(self.model_version_file, 'r') as f:
                        saved_version = f.read().strip()
                
                if saved_version and parse_version(saved_version) > parse_version(current_version):
                    self.logger.warning(
                        f"Model was trained with newer scikit-learn version ({saved_version}) "
                        f"than current version ({current_version}). "
                        f"This might cause compatibility issues. "
                        f"Consider retraining the model with the current scikit-learn version."
                    )
                    # Filter out specific scikit-learn InconsistentVersionWarning
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, 
                                               message=".*Trying to unpickle estimator.*")
                        self.model = joblib.load(self.model_file)
                else:
                    # Load normally
                    self.model = joblib.load(self.model_file)
                
                self.logger.info(f"Model loaded from {self.model_file}")
                if saved_version:
                    self.logger.info(f"Model was trained with scikit-learn version: {saved_version}")
                return True
            else:
                self.logger.info(f"No existing model found at {self.model_file}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
