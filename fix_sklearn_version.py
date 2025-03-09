#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VersionFixer')

def check_sklearn_version():
    """Check current scikit-learn version."""
    try:
        import sklearn
        current_version = sklearn.__version__
        logger.info(f"Current scikit-learn version: {current_version}")
        return current_version
    except ImportError:
        logger.warning("scikit-learn is not installed")
        return None

def find_model_versions(models_dir='models'):
    """Find scikit-learn versions used for existing models."""
    versions = set()
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory '{models_dir}' not found")
        return versions
    
    for file in os.listdir(models_dir):
        if file.endswith('_model_version.txt'):
            try:
                with open(os.path.join(models_dir, file), 'r') as f:
                    version = f.read().strip()
                    versions.add(version)
                    logger.info(f"Found model with scikit-learn version: {version} ({file})")
            except Exception as e:
                logger.error(f"Error reading version file '{file}': {e}")
    
    return versions

def install_sklearn_version(version):
    """Install specified scikit-learn version."""
    logger.info(f"Installing scikit-learn version {version}")
    try:
        # First uninstall current version to avoid conflicts
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "scikit-learn"])
        
        # Then install the specified version
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"scikit-learn=={version}"])
        
        logger.info(f"Successfully installed scikit-learn {version}")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install scikit-learn {version}: {e}")
        return False

def backup_models(models_dir='models', backup_dir='models_backup'):
    """Backup model files before making changes."""
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory '{models_dir}' not found, nothing to backup")
        return False
    
    try:
        # Create backup directory if it doesn't exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        # Count backed up files
        count = 0
        for file in os.listdir(models_dir):
            if file.endswith('_model.pkl'):
                source = os.path.join(models_dir, file)
                dest = os.path.join(backup_dir, file)
                shutil.copy2(source, dest)
                count += 1
                
        if count > 0:
            logger.info(f"Successfully backed up {count} model files to {backup_dir}")
        else:
            logger.info("No model files found to backup")
        
        return True
    except Exception as e:
        logger.error(f"Error backing up model files: {e}")
        return False

def fix_missing_loss_module():
    """Fix specific '_loss' module missing error."""
    try:
        import sklearn
        # The _loss module is in sklearn.neural_network in newer versions
        # but older code might be looking for it elsewhere
        if not hasattr(sklearn, '_loss'):
            # Try checking if the neural_network module has it
            from sklearn.neural_network import _loss
            # Make it accessible from the sklearn namespace
            sklearn._loss = _loss
            logger.info("Fixed missing '_loss' module reference")
        return True
    except ImportError:
        logger.error("Could not fix '_loss' module - neural_network module might be missing")
        return False
    except Exception as e:
        logger.error(f"Error fixing '_loss' module: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix scikit-learn version compatibility issues')
    parser.add_argument('--version', help='Specific scikit-learn version to install')
    parser.add_argument('--check-only', action='store_true', help='Only check versions without installing')
    parser.add_argument('--models-dir', default='models', help='Directory containing model files')
    parser.add_argument('--backup', action='store_true', help='Backup model files before making changes')
    parser.add_argument('--fix-loss-module', action='store_true', help="Fix missing '_loss' module error")
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining with current sklearn version')
    args = parser.parse_args()
    
    if args.backup:
        backup_models(args.models_dir)
    
    if args.fix_loss_module:
        if fix_missing_loss_module():
            logger.info("Applied fix for missing '_loss' module")
        return
    
    if args.force_retrain:
        logger.info("Forcing retraining of models with current scikit-learn version")
        # Remove or rename old model files to force retraining
        if os.path.exists(args.models_dir):
            for file in os.listdir(args.models_dir):
                if file.endswith('_model.pkl'):
                    try:
                        deprecated_file = os.path.join(args.models_dir, f"{file}.deprecated")
                        os.rename(os.path.join(args.models_dir, file), deprecated_file)
                        logger.info(f"Renamed {file} to {file}.deprecated to force retraining")
                    except Exception as e:
                        logger.error(f"Error renaming model file {file}: {e}")
        return
    
    current_version = check_sklearn_version()
    
    # Find versions used in existing models
    model_versions = find_model_versions(args.models_dir)
    
    if args.version:
        # Use specified version
        target_version = args.version
    elif model_versions:
        # Use the newest version found in models
        target_version = sorted(model_versions, reverse=True)[0]
        logger.info(f"Found {len(model_versions)} different versions, using newest: {target_version}")
    else:
        logger.info("No model version files found. Using current scikit-learn version.")
        if current_version:
            print(f"Current scikit-learn version: {current_version}")
        return
    
    if args.check_only:
        if current_version:
            if current_version == target_version:
                logger.info(f"Current version ({current_version}) matches target version ({target_version})")
            else:
                logger.info(f"Current version ({current_version}) differs from target version ({target_version})")
    else:
        if current_version != target_version:
            if install_sklearn_version(target_version):
                logger.info(f"Installed scikit-learn {target_version}")
                logger.info("Please restart your application to use the new version")
            else:
                logger.error("Failed to update scikit-learn version")
        else:
            logger.info(f"Already using correct scikit-learn version ({current_version})")

if __name__ == "__main__":
    main()
