"""
Machine learning models for enhancing trading signals.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
import joblib
import os

# Configure TensorFlow for CPU-only usage
import tensorflow as tf
# Disable GPU
try:
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    # Suppress TensorFlow logging except for errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        logger = logging.getLogger("ml_models")
        print("GPU is available but disabled by configuration")
except Exception as e:
    pass  # Ignore errors if GPU configuration fails

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna

import config
import indicators
import utils

logger = logging.getLogger("ml_models")

class MLModel(ABC):
    """Base class for machine learning models."""
    
    def __init__(self, model_name):
        """
        Initialize the ML model.
        
        Args:
            model_name (str): Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = None  # Not needed for classification
        self.features = []
        self.target = 'target'
        self.trained = False
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Created models directory")
    
    @abstractmethod
    def build_model(self):
        """Build the machine learning model."""
        pass
    
    def prepare_data(self, df, target_column='signal', prediction_horizon=1):
        """
        Prepare data for machine learning.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            prediction_horizon (int): Number of periods ahead to predict
            
        Returns:
            tuple: X (features), y (target)
        """
        # Create a copy of the dataframe
        data = df.copy()
        
        # Drop NaN values
        data = data.dropna()
        
        # Create target variable based on future price movement
        if target_column == 'signal':
            # Convert signal to numeric: buy=1, sell=-1, None=0
            data['target'] = 0
            data.loc[data['signal'] == 'buy', 'target'] = 1
            data.loc[data['signal'] == 'sell', 'target'] = -1
        elif target_column == 'price_direction':
            # Create target based on future price direction
            data['future_price'] = data['close'].shift(-prediction_horizon)
            data['target'] = (data['future_price'] > data['close']).astype(int)
            data = data.dropna()
        
        # Select features (all numeric columns except target, timestamp, and signal)
        self.features = [col for col in data.columns if col not in ['target', 'signal', 'timestamp', 'future_price'] 
                         and pd.api.types.is_numeric_dtype(data[col])]
        
        # Extract features and target
        X = data[self.features].values
        y = data['target'].values
        
        return X, y
    
    def train(self, df, target_column='signal', test_size=0.2, prediction_horizon=1):
        """
        Train the machine learning model.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            test_size (float): Proportion of data to use for testing
            prediction_horizon (int): Number of periods ahead to predict
            
        Returns:
            dict: Training metrics
        """
        # Prepare data
        X, y = self.prepare_data(df, target_column, prediction_horizon)
        
        # Split data into training and testing sets
        if target_column == 'price_direction':
            # Use time series split for price direction prediction
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
        else:
            # Use random split for signal prediction
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train = self.scaler_X.fit_transform(X_train)
        X_test = self.scaler_X.transform(X_test)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        if target_column == 'signal':
            # For multi-class classification (buy, sell, hold)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            # For binary classification (price up or down)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        
        # Add class distribution information
        metrics['y_test_distribution'] = {
            'positive': int(sum(y_test == 1)),
            'negative': int(sum(y_test == 0)) if 0 in y_test else int(sum(y_test == -1)),
            'total': len(y_test)
        }
        
        metrics['y_pred_distribution'] = {
            'positive': int(sum(y_pred == 1)),
            'negative': int(sum(y_pred == 0)) if 0 in y_pred else int(sum(y_pred == -1)),
            'total': len(y_pred)
        }
        
        print(f"Training metrics for {self.model_name}: {metrics}")
        print(f"Test data distribution: {metrics['y_test_distribution']}")
        print(f"Prediction distribution: {metrics['y_pred_distribution']}")
        
        self.trained = True
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions using the trained model.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            numpy.ndarray: Predictions
        """
        if not self.trained:
            print("Model not trained yet")
            return None
        
        # Prepare data
        data = df.copy()
        
        # Select features
        X = data[self.features].values
        
        # Scale features
        X = self.scaler_X.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def save_model(self):
        """Save the model to disk."""
        try:
            # Different saving methods depending on model type
            if hasattr(self.model, 'save'):
                # For Keras models
                model_path = os.path.join('models', f"{self.model_name}.keras")
                self.model.save(model_path, save_format='keras')
            else:
                # For scikit-learn models
                model_path = os.path.join('models', f"{self.model_name}.joblib")
                joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join('models', f"{self.model_name}_scaler.joblib")
            joblib.dump(self.scaler_X, scaler_path)
            
            # Save feature list
            features_path = os.path.join('models', f"{self.model_name}_features.joblib")
            joblib.dump(self.features, features_path)
            
            # Save sequence length if it exists
            if hasattr(self, 'sequence_length'):
                seq_length_path = os.path.join('models', f"{self.model_name}_seq_length.joblib")
                joblib.dump(self.sequence_length, seq_length_path)
            
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            # Try alternative saving method as fallback
            try:
                if hasattr(self.model, 'save'):
                    # Try h5 format for Keras models
                    h5_model_path = os.path.join('models', f"{self.model_name}.h5")
                    self.model.save(h5_model_path, save_format='h5')
                    logger.info(f"Saved model to {h5_model_path} using h5 format as fallback")
                else:
                    # Try pickle for scikit-learn models
                    pickle_path = os.path.join('models', f"{self.model_name}.pkl")
                    import pickle
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(self.model, f)
                    logger.info(f"Saved model to {pickle_path} using pickle as fallback")
                
                # Save other components
                scaler_path = os.path.join('models', f"{self.model_name}_scaler.joblib")
                joblib.dump(self.scaler_X, scaler_path)
                
                features_path = os.path.join('models', f"{self.model_name}_features.joblib")
                joblib.dump(self.features, features_path)
                
                if hasattr(self, 'sequence_length'):
                    seq_length_path = os.path.join('models', f"{self.model_name}_seq_length.joblib")
                    joblib.dump(self.sequence_length, seq_length_path)
                
                return True
            except Exception as e2:
                print(f"Failed to save model using fallback method: {e2}")
                return False
    
    def load_model(self):
        """Load the model from disk."""
        try:
            # Try different file formats
            keras_path = os.path.join('models', f"{self.model_name}.keras")
            h5_path = os.path.join('models', f"{self.model_name}.h5")
            joblib_path = os.path.join('models', f"{self.model_name}.joblib")
            pickle_path = os.path.join('models', f"{self.model_name}.pkl")
            
            # Check which format exists and load accordingly
            if os.path.exists(keras_path):
                # For Keras models
                self.model = load_model(keras_path)
                logger.info(f"Loaded Keras model from {keras_path}")
            elif os.path.exists(h5_path):
                # For older Keras models
                self.model = load_model(h5_path)
                logger.info(f"Loaded legacy Keras model from {h5_path}")
            elif os.path.exists(joblib_path):
                # For scikit-learn models
                self.model = joblib.load(joblib_path)
                logger.info(f"Loaded scikit-learn model from {joblib_path}")
            elif os.path.exists(pickle_path):
                # For pickle fallback
                import pickle
                with open(pickle_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded model from {pickle_path} using pickle")
            else:
                print(f"No model file found for {self.model_name}")
                return False
            
            # Load scaler
            scaler_path = os.path.join('models', f"{self.model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler_X = joblib.load(scaler_path)
            
            # Load feature list
            features_path = os.path.join('models', f"{self.model_name}_features.joblib")
            if os.path.exists(features_path):
                self.features = joblib.load(features_path)
            
            # Load sequence length if available
            seq_length_path = os.path.join('models', f"{self.model_name}_seq_length.joblib")
            if hasattr(self, 'sequence_length') and os.path.exists(seq_length_path):
                self.sequence_length = joblib.load(seq_length_path)
            
            self.trained = True
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


class RandomForestModel(MLModel):
    """Random Forest classifier for trading signals."""
    
    def __init__(self, n_estimators=100, max_depth=None):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
        """
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def build_model(self):
        """Build the Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        print(f"Built Random Forest model with {self.n_estimators} estimators")


class XGBoostModel(MLModel):
    """XGBoost classifier for trading signals."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate
            max_depth (int): Maximum depth of the trees
        """
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def build_model(self):
        """Build the XGBoost model."""
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        print(f"Built XGBoost model with {self.n_estimators} estimators")


class LSTMModel(MLModel):
    """LSTM neural network for trading signals."""
    
    def __init__(self, units=32, dropout=0.2, epochs=50, batch_size=32):
        """
        Initialize the LSTM model.
        
        Args:
            units (int): Number of LSTM units
            dropout (float): Dropout rate
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        super().__init__("lstm")
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = 10  # Number of time steps to look back
        self.scaler_X = MinMaxScaler()  # LSTM works better with normalized data
    
    def build_model(self):
        """Build the LSTM model."""
        # Define model architecture - simplified for CPU
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=False, input_shape=(self.sequence_length, len(self.features))),
            Dropout(self.dropout),
            BatchNormalization(),
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile model with reduced precision for CPU efficiency
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Built CPU-optimized LSTM model with {self.units} units")
    
    def prepare_data(self, df, target_column='price_direction', prediction_horizon=1):
        """
        Prepare data for LSTM model.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            prediction_horizon (int): Number of periods ahead to predict
            
        Returns:
            tuple: X (features), y (target)
        """
        # Create a copy of the dataframe
        data = df.copy()
        
        # Drop NaN values
        data = data.dropna()
        
        # Create target variable based on future price movement
        if target_column == 'signal':
            # Convert signal to numeric: buy=1, sell=-1, None=0
            data['target'] = 0
            data.loc[data['signal'] == 'buy', 'target'] = 1
            data.loc[data['signal'] == 'sell', 'target'] = -1
            # Convert to binary for LSTM
            data['target'] = (data['target'] > 0).astype(int)
        elif target_column == 'price_direction':
            # Create target based on future price direction
            data['future_price'] = data['close'].shift(-prediction_horizon)
            data['target'] = (data['future_price'] > data['close']).astype(int)
            data = data.dropna()
        
        # Select features (all numeric columns except target, timestamp, and signal)
        # Limit to most important features for CPU efficiency
        numeric_columns = [col for col in data.columns if col not in ['target', 'signal', 'timestamp', 'future_price'] 
                         and pd.api.types.is_numeric_dtype(data[col])]
        
        # Use only the most important features to reduce computation
        important_features = ['close', 'volume']
        
        # Add some technical indicators if available
        for col in numeric_columns:
            if any(indicator in col for indicator in ['sma', 'ema', 'rsi', 'macd']):
                important_features.append(col)
        
        # Limit to 10 features maximum for CPU efficiency
        self.features = important_features[:10]
        
        # Extract features and target
        feature_data = data[self.features].values
        target_data = data['target'].values
        
        # Scale features
        feature_data = self.scaler_X.fit_transform(feature_data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            X.append(feature_data[i:i + self.sequence_length])
            y.append(target_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, df, target_column='price_direction', test_size=0.2, prediction_horizon=1):
        """
        Train the LSTM model.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            test_size (float): Proportion of data to use for testing
            prediction_horizon (int): Number of periods ahead to predict
            
        Returns:
            dict: Training metrics
        """
        # Prepare data
        X, y = self.prepare_data(df, target_column, prediction_horizon)
        
        # Split data into training and testing sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join('models', f"{self.model_name}_best.keras"),
                monitor='val_loss',
                save_best_only=True,
                save_format='keras'  # Explicitly specify the format
            )
        ]
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"Error during model training: {e}")
            # Try with h5 format if keras format fails
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join('models', f"{self.model_name}_best.h5"),
                    monitor='val_loss',
                    save_best_only=True,
                    save_format='h5'  # Use h5 format as fallback
                )
            ]
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add class distribution information
        metrics['y_test_distribution'] = {
            'positive': int(sum(y_test == 1)),
            'negative': int(sum(y_test == 0)),
            'total': len(y_test)
        }
        
        metrics['y_pred_distribution'] = {
            'positive': int(sum(y_pred == 1)),
            'negative': int(sum(y_pred == 0)),
            'total': len(y_pred)
        }
        
        print(f"Training metrics for {self.model_name}: {metrics}")
        print(f"Test data distribution: {metrics['y_test_distribution']}")
        print(f"Prediction distribution: {metrics['y_pred_distribution']}")
        
        self.trained = True
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions using the trained model.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            numpy.ndarray: Predictions
        """
        if not self.trained:
            print("Model not trained yet")
            return None
        
        # Prepare data
        data = df.copy().tail(self.sequence_length)
        
        # Select features
        feature_data = data[self.features].values
        
        # Scale features
        feature_data = self.scaler_X.transform(feature_data)
        
        # Reshape for LSTM
        X = np.array([feature_data])
        
        # Make predictions
        predictions = (self.model.predict(X) > 0.5).astype(int)
        
        return predictions


class EnsembleModel(MLModel):
    """Ensemble model that combines predictions from multiple models."""
    
    def __init__(self, models=None):
        """
        Initialize the ensemble model.
        
        Args:
            models (list): List of model instances
        """
        super().__init__("ensemble")
        self.models = models or []
    
    def build_model(self):
        """Build the ensemble model."""
        # No need to build a model, as we'll use the individual models
        pass
    
    def add_model(self, model):
        """
        Add a model to the ensemble.
        
        Args:
            model: Model instance
        """
        self.models.append(model)
    
    def train(self, df, target_column='signal', test_size=0.2, prediction_horizon=1):
        """
        Train all models in the ensemble.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            test_size (float): Proportion of data to use for testing
            prediction_horizon (int): Number of periods ahead to predict
            
        Returns:
            dict: Training metrics for each model
        """
        metrics = {}
        
        for model in self.models:
            try:
                logger.info(f"Training model: {model.model_name}")
                model_metrics = model.train(df, target_column, test_size, prediction_horizon)
                metrics[model.model_name] = model_metrics
            except Exception as e:
                print(f"Error training model {model.model_name}: {e}")
                metrics[model.model_name] = {"error": str(e)}
        
        # Mark as trained if at least one model was trained successfully
        self.trained = any(not isinstance(m, dict) or "error" not in m for m in metrics.values())
        
        if not self.trained:
            logger.warning("No models in the ensemble were trained successfully")
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions using all models in the ensemble and combine them.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            numpy.ndarray: Combined predictions
        """
        if not self.trained:
            print("Models not trained yet")
            return None
        
        predictions = []
        
        for model in self.models:
            if model.trained:
                try:
                    model_preds = model.predict(df)
                    predictions.append(model_preds)
                except Exception as e:
                    print(f"Error getting predictions from {model.model_name}: {e}")
        
        if not predictions:
            print("No predictions available from any model")
            return None
        
        # Combine predictions (simple majority vote)
        combined_preds = np.mean(predictions, axis=0)
        combined_preds = (combined_preds > 0.5).astype(int)
        
        return combined_preds
    
    def save_model(self):
        """Save all models in the ensemble."""
        success = True
        
        # Create ensemble directory if it doesn't exist
        ensemble_dir = os.path.join('models', 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Save model list
        model_names = [model.model_name for model in self.models]
        model_list_path = os.path.join(ensemble_dir, 'model_list.joblib')
        try:
            joblib.dump(model_names, model_list_path)
            logger.info(f"Saved ensemble model list to {model_list_path}")
        except Exception as e:
            print(f"Failed to save ensemble model list: {e}")
            success = False
        
        # Save each individual model
        for model in self.models:
            try:
                if model.trained:
                    model_success = model.save_model()
                    if not model_success:
                        logger.warning(f"Failed to save model {model.model_name} in ensemble")
                        success = False
            except Exception as e:
                print(f"Error saving model {model.model_name} in ensemble: {e}")
                success = False
        
        return success
    
    def load_model(self):
        """Load all models in the ensemble."""
        success = True
        
        # Load model list
        ensemble_dir = os.path.join('models', 'ensemble')
        model_list_path = os.path.join(ensemble_dir, 'model_list.joblib')
        
        if not os.path.exists(model_list_path):
            print(f"Ensemble model list not found at {model_list_path}")
            return False
        
        try:
            model_names = joblib.load(model_list_path)
            logger.info(f"Loaded ensemble model list: {model_names}")
            
            # Clear current models
            self.models = []
            
            # Load each individual model
            for model_name in model_names:
                try:
                    if model_name == "random_forest":
                        model = RandomForestModel()
                    elif model_name == "xgboost":
                        model = XGBoostModel()
                    elif model_name == "lstm":
                        model = LSTMModel()
                    else:
                        logger.warning(f"Unknown model type: {model_name}")
                        continue
                    
                    model_success = model.load_model()
                    if model_success:
                        self.models.append(model)
                    else:
                        logger.warning(f"Failed to load model {model_name} in ensemble")
                        success = False
                except Exception as e:
                    print(f"Error loading model {model_name} in ensemble: {e}")
                    success = False
            
            self.trained = len(self.models) > 0
            
            if self.trained:
                logger.info(f"Successfully loaded ensemble with {len(self.models)} models")
            else:
                logger.warning("No models were loaded in the ensemble")
                success = False
            
            return success
        except Exception as e:
            print(f"Failed to load ensemble model: {e}")
            return False


class HyperparameterTuner:
    """Class for tuning hyperparameters of ML models using Optuna."""
    
    def __init__(self, model_type, df, target_column='price_direction', n_trials=100):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_type (str): Type of model to tune ('random_forest', 'xgboost', 'lstm')
            df (pandas.DataFrame): DataFrame with indicators
            target_column (str): Column name for the target variable
            n_trials (int): Number of trials for optimization
        """
        self.model_type = model_type
        self.df = df
        self.target_column = target_column
        self.n_trials = n_trials
    
    def objective(self, trial):
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial
            
        Returns:
            float: Validation score
        """
        if self.model_type == 'random_forest':
            # Define hyperparameters to tune
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            
            # Create and train model
            model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
            metrics = model.train(self.df, self.target_column)
            
            return metrics['f1']
        
        elif self.model_type == 'xgboost':
            # Define hyperparameters to tune
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            
            # Create and train model
            model = XGBoostModel(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth
            )
            metrics = model.train(self.df, self.target_column)
            
            return metrics['f1']
        
        elif self.model_type == 'lstm':
            # Define hyperparameters to tune
            units = trial.suggest_int('units', 32, 128)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create and train model
            model = LSTMModel(
                units=units,
                dropout=dropout,
                epochs=50,  # Reduced for faster tuning
                batch_size=batch_size
            )
            metrics = model.train(self.df, self.target_column)
            
            return metrics['f1']
    
    def tune(self):
        """
        Tune hyperparameters.
        
        Returns:
            dict: Best hyperparameters
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print(f"Best hyperparameters for {self.model_type}: {study.best_params}")
        print(f"Best score: {study.best_value}")
        
        return study.best_params


def create_ml_strategy(exchange, model_type='ensemble', train_data=None):
    """
    Create a machine learning strategy.
    
    Args:
        exchange: Exchange instance
        model_type (str): Type of model to use
        train_data (pandas.DataFrame): Data for training (if None, will fetch from exchange)
        
    Returns:
        tuple: Strategy instance, ML model
    """
    from strategy import MLStrategy
    
    # Create ML model based on type
    if model_type == 'random_forest':
        ml_model = RandomForestModel()
    elif model_type == 'xgboost':
        ml_model = XGBoostModel()
    elif model_type == 'lstm':
        ml_model = LSTMModel()
    elif model_type == 'ensemble':
        ml_model = EnsembleModel([
            RandomForestModel(),
            XGBoostModel(),
            LSTMModel()
        ])
    else:
        print(f"Unknown model type: {model_type}")
        return None, None
    
    # Create strategy with ML model
    strategy = MLStrategy(exchange, ml_model)
    
    # Train model if data is provided
    if train_data is not None:
        # Add indicators to data
        train_data = indicators.add_all_indicators(train_data)
        
        # Train model
        ml_model.train(train_data, target_column='price_direction')
    
    return strategy, ml_model 