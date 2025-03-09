#!/bin/bash

# Simple script to run the trading bot with different configurations

function usage {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --test      Run in test mode (simulate trades)"
    echo "  --live      Run in live mode (real trading)"
    echo "  --dashboard Start the web dashboard"
    echo "  --backtest  Run backtesting on historical data"
    echo "  --days N    Number of days for backtesting (default: 30)"
    echo "  --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --test --dashboard"
}

# Process command line arguments
TEST_MODE=""
DASHBOARD=""
BACKTEST=""
DAYS="30"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE="--test"
            shift
            ;;
        --live)
            TEST_MODE="--live"
            shift
            ;;
        --dashboard)
            DASHBOARD="--dashboard"
            shift
            ;;
        --backtest)
            BACKTEST="--backtest"
            shift
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check for virtual environment
if [ -d "venv" ]; then
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the trading bot
echo "Starting Trading Bot..."

if [[ -n "$BACKTEST" ]]; then
    python main.py $BACKTEST --days $DAYS
else
    python main.py $TEST_MODE $DASHBOARD
fi
