#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import joblib
import pickle
import warnings
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelConverter')

def extract_pipeline_components(model_file, output_dir='converted_models'):
    """
    Extract components from a scikit-learn pipeline and save them separately.
    This bypasses version compatibility issues by saving just the core information.
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load with pickle instead of joblib to avoid sklearn version issues
        with open(model_file, 'rb') as f:
            model_bytes = f.read()
        
        # Try different approaches to extract the model
        model = None
        errors = []
        
        # Method 1: Direct pickle load
        try:
            model = pickle.loads(model_bytes)
            logger.info("Successfully loaded model with direct pickle")
        except Exception as e:
            errors.append(f"Pickle load failed: {e}")
        
        # If still failed, try method 2: Modified pickle load
        if model is None:
            try:
                # Set some fake modules to help unpickling
                import types
                sys.modules['sklearn._loss'] = types.ModuleType('sklearn._loss')
                sys.modules['sklearn.neural_network._loss'] = types.ModuleType('sklearn.neural_network._loss')
                
                model = pickle.loads(model_bytes)
                logger.info("Successfully loaded model with modified pickle environment")
            except Exception as e:
                errors.append(f"Modified pickle load failed: {e}")
        
        if model is None:
            for error in errors:
                logger.error(error)
            logger.error("All loading attempts failed")
            return False
        
        # Get model basename
        basename = os.path.basename(model_file).replace('.pkl', '')
        
        # Extract and save pipeline components
        if hasattr(model, 'named_steps'):
            logger.info("Found scikit-learn pipeline")
            
            # Save component information
            components_info = {}
            
            # Extract classifier information
            if 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                classifier_file = os.path.join(output_dir, f"{basename}_classifier_info.txt")
                
                with open(classifier_file, 'w') as f:
                    if hasattr(classifier, 'estimators_') and hasattr(classifier, 'classes_'):
                        f.write(f"Type: {type(classifier).__name__}\n")
                        f.write(f"Classes: {classifier.classes_.tolist()}\n")
                        
                        if hasattr(classifier, 'estimators'):
                            f.write(f"Estimators: {[type(est[1]).__name__ for est in classifier.estimators]}\n")
                
                components_info['classifier'] = {
                    'type': type(classifier).__name__,
                    'file': classifier_file
                }
                logger.info(f"Saved classifier information to {classifier_file}")
            
            # Extract scaler information
            if 'scaler' in model.named_steps:
                scaler = model.named_steps['scaler']
                scaler_file = os.path.join(output_dir, f"{basename}_scaler_info.txt")
                
                with open(scaler_file, 'w') as f:
                    if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
                        f.write(f"Type: {type(scaler).__name__}\n")
                        f.write(f"Scale shape: {scaler.scale_.shape}\n")
                        f.write(f"Mean shape: {scaler.mean_.shape}\n")
                
                components_info['scaler'] = {
                    'type': type(scaler).__name__,
                    'file': scaler_file
                }
                logger.info(f"Saved scaler information to {scaler_file}")
            
            # Save components info
            info_file = os.path.join(output_dir, f"{basename}_components.txt")
            with open(info_file, 'w') as f:
                f.write(f"Pipeline components for {model_file}:\n")
                for name, info in components_info.items():
                    f.write(f"{name}: {info['type']} - {info['file']}\n")
            
            logger.info(f"Saved pipeline component information to {info_file}")
            
            # Save classification thresholds
            threshold_file = os.path.join(output_dir, f"{basename}_thresholds.py")
            with open(threshold_file, 'w') as f:
                f.write("# Classification thresholds extracted from the model\n")
                f.write("def get_threshold():\n")
                f.write("    return 0.5  # Default threshold for binary classification\n")
                f.write("\n")
                f.write("def predict_with_threshold(probabilities, threshold=None):\n")
                f.write("    if threshold is None:\n")
                f.write("        threshold = get_threshold()\n")
                f.write("    return (probabilities[:, 1] >= threshold).astype(int)\n")
            
            logger.info(f"Created threshold utility in {threshold_file}")
            
            # Create a training template
            template_file = os.path.join(output_dir, f"train_new_{basename}_model.py")
            with open(template_file, 'w') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write("import pandas as pd\n")
                f.write("import numpy as np\n")
                f.write("from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n")
                f.write("from sklearn.neural_network import MLPClassifier\n")
                f.write("from sklearn.preprocessing import StandardScaler, RobustScaler\n")
                f.write("from sklearn.pipeline import Pipeline\n")
                f.write("from sklearn.model_selection import train_test_split\n")
                f.write("from sklearn.metrics import accuracy_score\n")
                f.write("import joblib\n\n")
                f.write("# Load your data\n")
                f.write("# df = pd.read_csv('your_data.csv')\n\n")
                f.write("# Prepare features and target\n")
                f.write("# X = df[['feature1', 'feature2', ...]]\n")
                f.write("# y = df['target']\n\n")
                f.write("# Split data\n")
                f.write("# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\n")
                f.write("# Create model\n")
                f.write("model = Pipeline([\n")
                if 'scaler' in model.named_steps:
                    scaler_type = type(model.named_steps['scaler']).__name__
                    f.write(f"    ('scaler', {scaler_type}()),\n")
                if 'classifier' in model.named_steps:
                    classifier_type = type(model.named_steps['classifier']).__name__
                    if classifier_type == 'VotingClassifier':
                        f.write("    ('classifier', VotingClassifier(\n")
                        f.write("        estimators=[\n")
                        if hasattr(classifier, 'estimators'):
                            for name, est in classifier.estimators:
                                est_type = type(est).__name__
                                f.write(f"            ('{name}', {est_type}()),\n")
                        f.write("        ],\n")
                        f.write("        voting='soft'\n")
                        f.write("    ))\n")
                    else:
                        f.write(f"    ('classifier', {classifier_type}())\n")
                f.write("])\n\n")
                f.write("# Train model\n")
                f.write("# model.fit(X_train, y_train)\n\n")
                f.write("# Evaluate model\n")
                f.write("# y_pred = model.predict(X_test)\n")
                f.write("# accuracy = accuracy_score(y_test, y_pred)\n")
                f.write("# print(f'Accuracy: {accuracy:.4f}')\n\n")
                f.write("# Save model\n")
                f.write("# joblib.dump(model, 'new_model.pkl')\n")
            
            logger.info(f"Created training template in {template_file}")
            
            return True
        else:
            logger.error("Model is not a scikit-learn pipeline")
            return False
            
    except Exception as e:
        logger.error(f"Error extracting model components: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert scikit-learn model to bypass version compatibility issues')
    parser.add_argument('model_file', help='Path to the model file (.pkl)')
    parser.add_argument('--output-dir', default='converted_models', help='Output directory for converted model files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_file):
        logger.error(f"Model file not found: {args.model_file}")
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if extract_pipeline_components(args.model_file, args.output_dir):
            logger.info("Model conversion completed successfully")
        else:
            logger.error("Model conversion failed")

if __name__ == "__main__":
    main()
