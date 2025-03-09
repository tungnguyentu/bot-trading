#!/bin/bash

echo "==== Binance Trading Bot Setup ===="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit the .env file with your API keys."
fi

# Make run script executable
chmod +x run_bot.sh

echo ""
echo "Installation completed!"
echo ""
echo "To start the bot:"
echo "1. Edit .env file with your API keys"
echo "2. Run the bot with: ./run_bot.sh --test --dashboard"
echo ""
echo "For more options, try: ./run_bot.sh --help"
echo ""
echo "Happy trading!"
