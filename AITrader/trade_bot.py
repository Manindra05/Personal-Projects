import joblib
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY"       # <--- PASTE KEYS AGAIN
SECRET_KEY = "YOUR_SECRET_KEY" # <--- PASTE KEYS AGAIN
SYMBOL = "QQQ"
QUANTITY = 1                   # How many shares to trade

# 1. SETUP CLIENTS
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True) # paper=True is CRITICAL

def get_market_data():
    """Fetches the last 300 days of data to calculate indicators."""
    print(f"Fetching data for {SYMBOL}...")
    start_date = datetime.now() - timedelta(days=300)
    params = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Day,
        start=start_date
    )
    bars = data_client.get_stock_bars(params)
    df = bars.df.reset_index()
    
    # Calculate Indicators (Must match training EXACTLY)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df.iloc[-1:].copy() # Return only today's row

def make_prediction(latest_data):
    """Loads the brain and guesses the future."""
    model = joblib.load("trading_model.pkl")
    features = ['SMA_50', 'SMA_200', 'RSI', 'ATR']
    prediction = model.predict(latest_data[features])[0]
    probability = model.predict_proba(latest_data[features])[0]
    return prediction, probability

def execute_trade(signal, probability):
    """The 'Hand' that places the order."""
    # Check if we already own the stock
    positions = trading_client.get_all_positions()
    current_qty = 0
    for p in positions:
        if p.symbol == SYMBOL:
            current_qty = float(p.qty)

    print(f"Current Position: {current_qty} shares")

    # --- LOGIC: BUY SIGNAL ---
    if signal == 1 and current_qty == 0:
        print(f"🟢 AI says BUY (Conf: {probability[2]*100:.1f}%) -> Buying {QUANTITY} share.")
        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=QUANTITY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=order)

    # --- LOGIC: SELL SIGNAL ---
    elif signal == -1 and current_qty > 0:
        print(f"🔴 AI says SELL (Conf: {probability[0]*100:.1f}%) -> Selling all shares.")
        trading_client.close_position(SYMBOL)
    
    # --- LOGIC: SHORT SIGNAL (Advanced) ---
    elif signal == -1 and current_qty == 0:
        print(f"🔴 AI says SHORT (Conf: {probability[0]*100:.1f}%) -> Betting against market.")
        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=QUANTITY,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=order)
        
    else:
        print("⚪ No action required (Hold or Low Confidence).")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Check if market is open
    clock = trading_client.get_clock()
    if clock.is_open:
        row = get_market_data()
        pred, prob = make_prediction(row)
        execute_trade(pred, prob)
    else:
        print("💤 Market is closed. Bot is sleeping.")