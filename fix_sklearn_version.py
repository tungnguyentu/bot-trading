#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import logging

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
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"scikit-learn=={version}"])
        logger.info(f"Successfully installed scikit-learn {version}")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install scikit-learn {version}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix scikit-learn version compatibility issues')
    parser.add_argument('--version', help='Specific scikit-learn version to install')
    parser.add_argument('--check-only', action='store_true', help='Only check versions without installing')
    parser.add_argument('--models-dir', default='models', help='Directory containing model files')
    args = parser.parse_args()
    
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
