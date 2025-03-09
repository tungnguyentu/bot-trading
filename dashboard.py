import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import threading
import logging
import json
import os
from binance.client import Client

from config import (BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOL, DASHBOARD_PORT, 
                   EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, 
                   TAKE_PROFIT_PCT, STOP_LOSS_PCT, INTERVAL, DASHBOARD_ENABLED)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Dashboard')

# Initialize data storage
trade_history_file = 'trade_history.json'
if not os.path.exists(trade_history_file):
    with open(trade_history_file, 'w') as f:
        json.dump([], f)

# Initialize Dash app
app = dash.Dash(__name__, title=f"Trading Bot Dashboard - {TRADING_SYMBOL}")

# Get Binance client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Create layout
app.layout = html.Div([
    html.H1(f"Trading Bot Dashboard - {TRADING_SYMBOL}", style={'textAlign': 'center'}),
    
    # Strategy Parameters
    html.Div([
        html.Div([
            html.H3("Strategy Parameters", style={'textAlign': 'center'}),
            html.Table([
                html.Tr([html.Td("Symbol:"), html.Td(TRADING_SYMBOL)]),
                html.Tr([html.Td("Interval:"), html.Td(INTERVAL)]),
                html.Tr([html.Td("EMA Short/Long:"), html.Td(f"{EMA_SHORT}/{EMA_LONG}")]),
                html.Tr([html.Td("RSI Period:"), html.Td(f"{RSI_PERIOD}")]),
                html.Tr([html.Td("RSI Thresholds:"), html.Td(f"Overbought: {RSI_OVERBOUGHT}, Oversold: {RSI_OVERSOLD}")]),
                html.Tr([html.Td("Take Profit:"), html.Td(f"{TAKE_PROFIT_PCT}%")]),
                html.Tr([html.Td("Stop Loss:"), html.Td(f"{STOP_LOSS_PCT}%")]),
            ], style={'margin': 'auto'})
        ], className='six columns', style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '5px'}),
        
        # Performance Metrics
        html.Div([
            html.H3("Performance Metrics", style={'textAlign': 'center'}),
            html.Div(id='metrics-container', style={'padding': '10px'})
        ], className='six columns', style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '5px'})
    ], className='row', style={'margin': '10px', 'display': 'flex'}),
    
    # Price Chart
    html.Div([
        html.H3("Price Chart with Indicators", style={'textAlign': 'center'}),
        dcc.Graph(id='price-chart'),
    ], style={'margin': '10px'}),
    
    # Trade History
    html.Div([
        html.H3("Trade History", style={'textAlign': 'center'}),
        html.Div(id='trade-table')
    ], style={'margin': '10px'}),
    
    # Portfolio Performance
    html.Div([
        html.H3("Portfolio Performance", style={'textAlign': 'center'}),
        dcc.Graph(id='portfolio-chart'),
    ], style={'margin': '10px'}),
    
    # Refresh Interval
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # in milliseconds (1 min)
        n_intervals=0
    )
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('metrics-container', 'children'),
     Output('trade-table', 'children'),
     Output('portfolio-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components."""
    # Fetch price data
    df = get_price_data()
    
    # Fetch trade history
    trades = get_trade_history()
    
    # Update metrics
    metrics = calculate_metrics(df, trades)
    
    # Generate chart
    price_figure = create_price_chart(df, trades)
    
    # Generate trade table
    trade_table = create_trade_table(trades)
    
    # Create portfolio chart
    portfolio_figure = create_portfolio_chart(trades, df)
    
    return price_figure, metrics, trade_table, portfolio_figure

def get_price_data():
    """Fetch price data from Binance and calculate indicators."""
    try:
        # Fetch klines from Binance
        klines = client.get_klines(
            symbol=TRADING_SYMBOL,
            interval=INTERVAL,
            limit=100  # Last 100 candles
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate indicators
        # EMA
        df[f'ema_{EMA_SHORT}'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df[f'ema_{EMA_LONG}'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

def get_trade_history():
    """Load trade history from file."""
    try:
        with open(trade_history_file, 'r') as f:
            trades = json.load(f)
        
        # Convert timestamp strings to datetime objects
        for trade in trades:
            if 'timestamp' in trade:
                trade['timestamp'] = datetime.fromtimestamp(trade['timestamp'])
        
        return trades
    
    except Exception as e:
        logger.error(f"Error loading trade history: {e}")
        return []

def calculate_metrics(df, trades):
    """Calculate performance metrics."""
    if df.empty:
        return html.Div("No data available")
    
    # Current price
    current_price = df['close'].iloc[-1]
    
    # Trading metrics
    buy_trades = [t for t in trades if t.get('type') == 'buy']
    sell_trades = [t for t in trades if t.get('type') == 'sell']
    
    total_trades = len(buy_trades)
    winning_trades = len([t for t in sell_trades if t.get('profit_loss', 0) > 0])
    losing_trades = len([t for t in sell_trades if t.get('profit_loss', 0) <= 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) > 0])
    total_loss = abs(sum([t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) <= 0]))
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Create metrics display
    return html.Table([
        html.Tr([html.Td("Current Price:"), html.Td(f"${current_price:.2f}")]),
        html.Tr([html.Td("Total Trades:"), html.Td(f"{total_trades}")]),
        html.Tr([html.Td("Winning Trades:"), html.Td(f"{winning_trades} ({win_rate:.1f}%)")]),
        html.Tr([html.Td("Losing Trades:"), html.Td(f"{losing_trades}")]),
        html.Tr([html.Td("Total Profit:"), html.Td(f"${total_profit:.2f}")]),
        html.Tr([html.Td("Total Loss:"), html.Td(f"${total_loss:.2f}")]),
        html.Tr([html.Td("Profit Factor:"), html.Td(f"{profit_factor:.2f}")]),
    ], style={'margin': 'auto'})

def create_price_chart(df, trades):
    """Create an interactive price chart with indicators."""
    if df.empty:
        return {}
    
    # Create candlestick chart
    candlestick = go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )
    
    # Add EMA indicators
    ema_short = go.Scatter(
        x=df['timestamp'],
        y=df[f'ema_{EMA_SHORT}'],
        name=f'EMA {EMA_SHORT}',
        line=dict(color='rgba(255, 165, 0, 0.7)')
    )
    
    ema_long = go.Scatter(
        x=df['timestamp'],
        y=df[f'ema_{EMA_LONG}'],
        name=f'EMA {EMA_LONG}',
        line=dict(color='rgba(75, 0, 130, 0.7)')
    )
    
    # Calculate market stance for visualization
    # This is a simplified version for display purposes only
    df['ema_diff'] = df[f'ema_{EMA_SHORT}'] - df[f'ema_{EMA_LONG}']
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Create a stance indicator (green for long, red for short, gray for neutral)
    df['market_stance'] = 'NEUTRAL'
    df.loc[(df['ema_diff'] > 0) & (df['macd_diff'] > 0), 'market_stance'] = 'LONG'
    df.loc[(df['ema_diff'] < 0) & (df['macd_diff'] < 0), 'market_stance'] = 'SHORT'
    
    # Create a color array for stance visualization
    stance_colors = []
    for stance in df['market_stance']:
        if stance == 'LONG':
            stance_colors.append('green')
        elif stance == 'SHORT':
            stance_colors.append('red')
        else:
            stance_colors.append('gray')
    
    # Add market stance indicator to the chart
    stance_indicator = go.Scatter(
        x=df['timestamp'],
        y=[df['close'].min() * 0.99] * len(df),  # Place it slightly below min price
        mode='markers',
        name='Market Stance',
        marker=dict(
            color=stance_colors,
            size=10,
            symbol='circle'
        )
    )
    
    # Add buy/sell markers
    buy_times = [t['timestamp'] for t in trades if t.get('type') == 'buy']
    buy_prices = [t['price'] for t in trades if t.get('type') == 'buy']
    
    sell_times = [t['timestamp'] for t in trades if t.get('type') == 'sell']
    sell_prices = [t['price'] for t in trades if t.get('type') == 'sell']
    
    buy_markers = go.Scatter(
        x=buy_times,
        y=buy_prices,
        mode='markers',
        name='Buy',
        marker=dict(
            color='green',
            size=10,
            symbol='triangle-up'
        )
    )
    
    sell_markers = go.Scatter(
        x=sell_times,
        y=sell_prices,
        mode='markers',
        name='Sell',
        marker=dict(
            color='red',
            size=10,
            symbol='triangle-down'
        )
    )
    
    # Create the figure
    price_chart = {
        'data': [candlestick, ema_short, ema_long, buy_markers, sell_markers, stance_indicator],
        'layout': go.Layout(
            title='Price with EMA Indicators and Market Stance',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Price'),
            legend=dict(orientation="h"),
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    }
    
    # Create RSI subplot
    rsi_chart = {
        'data': [
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI'
            ),
            go.Scatter(
                x=df['timestamp'],
                y=[RSI_OVERBOUGHT] * len(df),
                name='Overbought',
                line=dict(color='red', dash='dash')
            ),
            go.Scatter(
                x=df['timestamp'],
                y=[RSI_OVERSOLD] * len(df),
                name='Oversold',
                line=dict(color='green', dash='dash')
            )
        ],
        'layout': go.Layout(
            title='RSI',
            xaxis=dict(title='Time'),
            yaxis=dict(title='RSI', range=[0, 100]),
            height=300,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    }
    
    # Combine charts into a single figure
    figure = {
        'data': price_chart['data'] + rsi_chart['data'],
        'layout': go.Layout(
            title='Price with Indicators',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Price'),
            yaxis2=dict(
                title='RSI',
                range=[0, 100],
                overlaying='y',
                side='right'
            ),
            legend=dict(orientation="h"),
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    }
    
    return figure

def create_trade_table(trades):
    """Create a table displaying trade history."""
    if not trades:
        return html.Div("No trades yet")
    
    # Prepare trade data for display
    trade_data = []
    for trade in trades:
        trade_type = trade.get('type', '')
        
        if trade_type == 'buy':
            trade_data.append({
                'Type': 'Buy',
                'Time': trade.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if isinstance(trade.get('timestamp'), datetime) else 'N/A',
                'Price': f"${trade.get('price', 0):.2f}",
                'Quantity': f"{trade.get('quantity', 0)}",
                'Cost': f"${trade.get('cost', 0):.2f}",
                'Profit/Loss': 'N/A',
                'Exit Reason': 'N/A'
            })
        elif trade_type == 'sell':
            trade_data.append({
                'Type': 'Sell',
                'Time': trade.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if isinstance(trade.get('timestamp'), datetime) else 'N/A',
                'Price': f"${trade.get('price', 0):.2f}",
                'Quantity': f"{trade.get('quantity', 0)}",
                'Cost': 'N/A',
                'Profit/Loss': f"${trade.get('profit_loss', 0):.2f} ({trade.get('profit_percentage', 0):.2f}%)",
                'Exit Reason': trade.get('exit_reason', 'N/A').capitalize()
            })
    
    # Create table
    return dash_table.DataTable(
        id='trade-history-table',
        columns=[
            {'name': 'Type', 'id': 'Type'},
            {'name': 'Time', 'id': 'Time'},
            {'name': 'Price', 'id': 'Price'},
            {'name': 'Quantity', 'id': 'Quantity'},
            {'name': 'Cost/Revenue', 'id': 'Cost'},
            {'name': 'Profit/Loss', 'id': 'Profit/Loss'},
            {'name': 'Exit Reason', 'id': 'Exit Reason'}
        ],
        data=trade_data,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {
                    'filter_query': '{Type} = "Buy"',
                },
                'backgroundColor': 'rgba(0, 255, 0, 0.1)'
            },
            {
                'if': {
                    'filter_query': '{Type} = "Sell"',
                },
                'backgroundColor': 'rgba(255, 0, 0, 0.1)'
            }
        ]
    )

def create_portfolio_chart(trades, df):
    """Create a chart showing portfolio performance over time."""
    if not trades or df.empty:
        return {}
    
    # Create a DataFrame with all dates in the range
    if trades:
        first_trade_date = min([t['timestamp'] for t in trades if isinstance(t.get('timestamp'), datetime)])
        date_range = pd.date_range(start=first_trade_date, end=datetime.now(), freq='D')
        portfolio_df = pd.DataFrame(index=date_range)
        portfolio_df['balance'] = None
        
        # Initialize with starting balance
        initial_balance = 1000  # Default starting balance
        
        # Reconstruct portfolio value over time
        current_balance = initial_balance
        current_crypto = 0
        
        for trade in sorted(trades, key=lambda x: x['timestamp'] if isinstance(x.get('timestamp'), datetime) else datetime(1970, 1, 1)):
            trade_date = trade['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
            
            if trade['type'] == 'buy':
                current_balance -= trade.get('cost', 0)
                current_crypto += trade.get('quantity', 0)
            elif trade['type'] == 'sell':
                current_balance += trade.get('revenue', 0)
                current_crypto -= trade.get('quantity', 0)
            
            # Update portfolio value at this date
            portfolio_df.loc[trade_date, 'balance'] = current_balance
        
        # Forward fill to get continuous balance
        portfolio_df['balance'] = portfolio_df['balance'].fillna(method='ffill')
        portfolio_df['balance'] = portfolio_df['balance'].fillna(initial_balance)  # Fill initial value
        
        # Create the figure
        figure = {
            'data': [
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['balance'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                )
            ],
            'layout': go.Layout(
                title='Portfolio Performance Over Time',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Portfolio Value (USD)'),
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
        }
        
        return figure
    else:
        return {}

def run_dashboard(in_thread=False):
    """Run the dashboard server.
    
    Args:
        in_thread: Whether the dashboard is running in a thread
    """
    logger.info(f"Starting dashboard on port {DASHBOARD_PORT}")
    # When running in a thread, disable debug mode to avoid signal handling issues
    debug_mode = not in_thread
    app.run_server(debug=debug_mode, port=DASHBOARD_PORT, use_reloader=False)

if __name__ == '__main__':
    if DASHBOARD_ENABLED:
        # When run directly, not in a thread
        run_dashboard(in_thread=False)
    else:
        logger.info("Dashboard is disabled. Set DASHBOARD_ENABLED=true to enable it.")
