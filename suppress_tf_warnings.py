#!/usr/bin/env python
"""
Script to run the trading bot with TensorFlow warnings suppressed.
"""
import os
import sys
import subprocess

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Disable TensorFlow debugging
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    
    # Disable eager execution for better performance on CPU
    tf.compat.v1.disable_eager_execution()
except ImportError:
    pass  # TensorFlow not installed, which is fine

# Run the actual command
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get the command to run (everything after this script name)
        command = sys.argv[1:]
        
        # Print the command being run
        print(f"Running command with TensorFlow warnings suppressed: python {' '.join(command)}")
        
        # Run the command
        result = subprocess.run(["python"] + command)
        
        # Exit with the same exit code
        sys.exit(result.returncode)
    else:
        print("Usage: python suppress_tf_warnings.py [command]")
        print("Example: python suppress_tf_warnings.py main.py train --model ensemble --symbol SOL/USDT --timeframe 1h --limit 1000 --cpu-only")
        sys.exit(1) 