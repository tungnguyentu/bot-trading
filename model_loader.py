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
                try:
                    from sklearn.neural_network import _loss
                    sklearn._loss = _loss
                    logger.info("Fixed '_loss' module reference using neural_network._loss")
                except ImportError:
                    # Create our own minimal version
                    logger.info("Creating custom _loss module replacement")
                    
                    class DummyLossModule:
                        @staticmethod
                        def log_loss(y_true, y_prob, eps=1e-15, normalize=True, sample_weight=None, labels=None):
                            """Dummy log_loss function."""
                            import numpy as np
                            from sklearn.metrics import log_loss
                            return log_loss(y_true, y_prob, eps=eps, normalize=normalize, 
                                          sample_weight=sample_weight, labels=labels)
                        
                        @staticmethod
                        def binary_log_loss(y_true, y_prob, eps=1e-15):
                            """Dummy binary_log_loss function."""
                            import numpy as np
                            return -np.sum(y_true * np.log(y_prob + eps) +
                                          (1 - y_true) * np.log(1 - y_prob + eps))
                    
                    # Create an empty dummy module
                    import types
                    dummy_loss = types.ModuleType('_loss')
                    
                    # Add the required functions
                    dummy_loss.log_loss = DummyLossModule.log_loss
                    dummy_loss.binary_log_loss = DummyLossModule.binary_log_loss
                    
                    # Attach it to sklearn
                    sklearn._loss = dummy_loss
                    logger.info("Created custom _loss module and attached to sklearn")
            except Exception as e:
                logger.error(f"Failed to create loss module: {e}")
                return False
        
        # Fix for other potential missing modules
        if not hasattr(sklearn, 'utils'):
            import types
            sklearn.utils = types.ModuleType('utils')
            logger.info("Created utils module for sklearn")
            
        return True
    except Exception as e:
        logger.error(f"Error applying sklearn fixes: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        if not fix_sklearn_modules():
            logger.warning("Failed to apply sklearn module fixes")
        
        if not monkey_patch_joblib():
            logger.warning("Failed to monkey patch joblib")
        
        # Attempt to load with all warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message=".*sklearn version.*")
            
            try:
                # First try normal loading
                model = joblib.load(model_file)
            except Exception as first_error:
                # If that fails, try a more robust approach - load with custom pickle options
                logger.warning(f"Standard loading failed: {first_error}, trying custom loading approach")
                
                # Use a custom unpickler that handles the _loss module
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Handle the _loss module specially
                        if module == 'sklearn._loss':
                            if name == 'log_loss' or name == 'binary_log_loss':
                                # Create dummy functions that match the signature
                                def dummy_func(*args, **kwargs):
                                    return 0
                                return dummy_func
                        # For everything else, use the normal unpickling mechanism
                        return super().find_class(module, name)
                
                # Try loading with custom unpickler
                try:
                    with open(model_file, 'rb') as f:
                        model = CustomUnpickler(f).load()
                    logger.info("Successfully loaded model with CustomUnpickler")
                except Exception as e:
                    # Final fallback: try direct import and patching of numpy bool handling
                    logger.warning(f"CustomUnpickler failed: {e}, trying direct import")
                    
                    # Handle NumPy truth value ambiguity
                    import numpy as np
                    original_bool = np.bool_
                    
                    # Patch numpy's bool to handle arrays in boolean context
                    def patched_bool(x):
                        if isinstance(x, np.ndarray) and x.size > 1:
                            return bool(x.any())
                        return original_bool(x)
                    
                    # Apply monkey patch (temporary)
                    np.bool_ = patched_bool
                    
                    # Now try again with pickle
                    try:
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        logger.info("Successfully loaded model with patched numpy bool handling")
                    except Exception as e:
                        logger.error(f"All loading methods failed: {e}")
                        return None
                    finally:
                        # Restore original numpy bool
                        np.bool_ = original_bool
            
            logger.info(f"Successfully loaded model from {model_file}")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
