#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from glob import glob
from model_loader import safe_load_model, get_model_version, check_scikit_learn_version

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CompatibilityCheck')

def check_all_models(models_dir='models'):
    """Check compatibility of all models in the directory."""
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return
    
    # Find all model files
    model_files = glob(os.path.join(models_dir, '*_model.pkl'))
    
    if not model_files:
        logger.info(f"No model files found in {models_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model files")
    
    # Check each model
    compatibility_issues = 0
    for model_file in model_files:
        version_file = model_file.replace('_model.pkl', '_model_version.txt')
        model_version = get_model_version(version_file)
        
        logger.info(f"Checking model: {os.path.basename(model_file)}")
        current_version, model_version = check_scikit_learn_version(model_version)
        
        # Try loading the model
        try:
            model = safe_load_model(model_file)
            if model:
                logger.info(f"✓ Model {os.path.basename(model_file)} loaded successfully")
            else:
                logger.error(f"✗ Model {os.path.basename(model_file)} failed to load")
                compatibility_issues += 1
        except Exception as e:
            logger.error(f"✗ Error loading model {os.path.basename(model_file)}: {e}")
            compatibility_issues += 1
    
    if compatibility_issues > 0:
        logger.warning(f"Found {compatibility_issues} compatibility issues")
        logger.info("You can fix these issues by:")
        logger.info("1. Running: python fix_sklearn_version.py --fix-loss-module")
        logger.info("2. Or reinstalling the correct scikit-learn version: python fix_sklearn_version.py")
        logger.info("3. Or retraining the models: python fix_sklearn_version.py --force-retrain")
    else:
        logger.info("All models are compatible with the current environment")

def main():
    parser = argparse.ArgumentParser(description='Check ML model compatibility')
    parser.add_argument('--models-dir', default='models', help='Directory containing model files')
    args = parser.parse_args()
    
    check_all_models(args.models_dir)

if __name__ == "__main__":
    main()
