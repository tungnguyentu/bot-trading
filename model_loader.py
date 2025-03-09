#!/usr/bin/env python3

import os
import logging
import warnings
import joblib
import importlib
from pkg_resources import parse_version

logger = logging.getLogger('ModelLoader')

def fix_sklearn_modules():
    """Apply fixes for common sklearn module import errors."""
    try:
        # Fix for '_loss' module missing
        import sklearn
        if not hasattr(sklearn, '_loss'):
            try:
                # Try to find _loss in neural_network (newer versions)
                from sklearn.neural_network import _loss
                sklearn._loss = _loss
                logger.info("Fixed '_loss' module reference")
            except ImportError:
                pass
        
        # Check if we need to fix other modules...
        return True
    except Exception as e:
        logger.error(f"Error applying sklearn fixes: {e}")
        return False

def monkey_patch_joblib():
    """Monkey patch joblib to handle missing modules."""
    try:
        original_load = joblib.load
        
        def patched_load(*args, **kwargs):
            try:
                return original_load(*args, **kwargs)
            except ModuleNotFoundError as e:
                if '_loss' in str(e):
                    logger.info("Fixing missing '_loss' module before loading model...")
                    fix_sklearn_modules()
                    return original_load(*args, **kwargs)
                else:
                    raise
        
        joblib.load = patched_load
        logger.info("Applied monkey patch to joblib.load")
        return True
    except Exception as e:
        logger.error(f"Failed to monkey patch joblib: {e}")
        return False

def safe_load_model(model_file):
    """Safely load a model handling common errors."""
    if not os.path.exists(model_file):
        logger.error(f"Model file does not exist: {model_file}")
        return None
    
    try:
        # Apply fixes before loading
        fix_sklearn_modules()
        monkey_patch_joblib()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = joblib.load(model_file)
            logger.info(f"Successfully loaded model from {model_file}")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def get_model_version(version_file):
    """Get the scikit-learn version used to train the model."""
    try:
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = f.read().strip()
            return version
        return None
    except Exception as e:
        logger.error(f"Error reading model version: {e}")
        return None

def check_scikit_learn_version(model_version):
    """Check if current scikit-learn version is compatible with model version."""
    try:
        import sklearn
        current_version = sklearn.__version__
        
        if model_version and parse_version(current_version) != parse_version(model_version):
            logger.warning(f"Current scikit-learn version ({current_version}) differs from model version ({model_version})")
            logger.warning("This might lead to compatibility issues.")
            
            # Provide recommendations
            if parse_version(current_version) < parse_version(model_version):
                logger.warning(f"Consider upgrading scikit-learn: pip install scikit-learn=={model_version}")
            else:
                logger.warning("Consider retraining the model with your current scikit-learn version")
        
        return current_version, model_version
    except ImportError:
        logger.error("scikit-learn not installed")
        return None, model_version

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Load and check ML models')
    parser.add_argument('model_file', help='Path to the model file')
    parser.add_argument('--fix', action='store_true', help='Apply fixes before loading')
    args = parser.parse_args()
    
    if args.fix:
        fix_sklearn_modules()
    
    version_file = args.model_file.replace('_model.pkl', '_model_version.txt')
    model_version = get_model_version(version_file)
    current_version, _ = check_scikit_learn_version(model_version)
    
    logger.info(f"Attempting to load model: {args.model_file}")
    model = safe_load_model(args.model_file)
    
    if model:
        logger.info("Model loaded successfully")
    else:
        logger.error("Failed to load model")
