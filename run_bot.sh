#!/bin/bash
# Script to run the trading bot with TensorFlow warnings suppressed

# Set environment variables to suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=-1

# Run the command
python "$@" 