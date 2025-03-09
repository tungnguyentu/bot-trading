#!/usr/bin/env python3

import os
import logging
import argparse
import joblib
import pickle
import numpy as np
import shutil
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelRepair')

def backup_model(model_file):
    """Create a backup of the model file before modifying it."""
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return False
    
    backup_dir = "model_backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.basename(model_file)
    backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}.bak")
    
    try:
        shutil.copy2(model_file, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def repair_model_pickle(model_file):
    """Attempt to repair a corrupted model file by re-serializing it."""
    if not backup_model(model_file):
        logger.warning("Proceeding without backup")
    
    try:
        # First, try to load the model with custom unpickling
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle problematic modules
                if module == 'sklearn._loss':
                    import types
                    dummy_module = types.ModuleType('_loss')
                    
                    # Create dummy functions
                    def log_loss(*args, **kwargs):
                        return 0
                    
                    def binary_log_loss(*args, **kwargs):
                        return 0
                    
                    dummy_module.log_loss = log_loss
                    dummy_module.binary_log_loss = binary_log_loss
                    
                    if name == 'log_loss':
                        return log_loss
                    elif name == 'binary_log_loss':
                        return binary_log_loss
                
                return super().find_class(module, name)
        
        # Try to load with custom unpickler
        with open(model_file, 'rb') as f:
            model = CustomUnpickler(f).load()
            
        logger.info(f"Successfully loaded model from {model_file}")
        
        # Re-save the model with joblib
        new_file = f"{model_file}.repaired"
        joblib.dump(model, new_file)
        logger.info(f"Re-saved model to {new_file}")
        
        # Replace the original file
        os.replace(new_file, model_file)
        logger.info(f"Replaced original model file with repaired version")
        
        return True
    except Exception as e:
        logger.error(f"Failed to repair model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def force_retrain_model(symbol):
    """Mark a model file for retraining by renaming it."""
    model_file = f"models/{symbol}_model.pkl"
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return False
    
    if not backup_model(model_file):
        logger.warning("Proceeding without backup")
    
    try:
        # Rename the model file to force retraining
        deprecated_file = f"{model_file}.deprecated"
        os.rename(model_file, deprecated_file)
        logger.info(f"Renamed {model_file} to {deprecated_file}")
        logger.info(f"The model will be retrained next time you run train_ml_model.py")
        return True
    except Exception as e:
        logger.error(f"Failed to mark model for retraining: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Repair or manage ML model files")
    parser.add_argument('--symbol', default='SOLUSDT', help='Symbol of the model to repair')
    parser.add_argument('--repair', action='store_true', help='Try to repair the model')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    parser.add_argument('--model-file', help='Path to specific model file')
    
    args = parser.parse_args()
    
    model_file = args.model_file if args.model_file else f"models/{args.symbol}_model.pkl"
    
    if args.repair:
        if repair_model_pickle(model_file):
            print(f"Successfully repaired model: {model_file}")
        else:
            print(f"Failed to repair model: {model_file}")
    elif args.retrain:
        if force_retrain_model(args.symbol):
            print(f"Model {args.symbol} marked for retraining")
        else:
            print(f"Failed to mark model for retraining")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
