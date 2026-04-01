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
import time

# --- CONFIGURATION ---
API_KEY = "API_KEY" 
SECRET_KEY = "SECRET_KEY" 

# LIST OF STOCKS TO TRADE
SYMBOLS = ["QQQ", "ASTS", "AMZN", "NVDA", "TSLA", "META"] 
QUANTITY = 5                  

# 1. SETUP CLIENTS
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True) #paper=true for paper trading

def get_market_data(symbol):
    """Fetches data for a SPECIFIC symbol."""
    print(f"\nFetching data for {symbol}...")
    start_date = datetime.now() - timedelta(days=300)
    params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date
    )
    bars = data_client.get_stock_bars(params)
    df = bars.df.reset_index()
    
    
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df.iloc[-1:].copy()

def make_prediction(latest_data):
    """Loads the brain and predicts."""
    model = joblib.load("trading_model.pkl")
    features = ['EMA_20', 'SMA_200', 'RSI', 'ATR']
    
    prediction = model.predict(latest_data[features])[0]
    probability = model.predict_proba(latest_data[features])[0]
    return prediction, probability

def execute_trade(signal, probability, symbol):
    """Executes trade for the SPECIFIC symbol passed in."""
    
    # 1. CHECK CURRENT POSITION FOR THIS STOCK
    positions = trading_client.get_all_positions()
    current_qty = 0
    for p in positions:
        if p.symbol == symbol:
            current_qty = float(p.qty)

    print(f"   Current {symbol} Position: {current_qty} shares")

    # Helper to place orders easily
    def submit_order(side):
        trading_client.submit_order(order_data=MarketOrderRequest(
            symbol=symbol,
            qty=QUANTITY,
            side=side,
            time_in_force=TimeInForce.DAY
        ))
        print(f"   -> Order Submitted: {side} {symbol}")

    # 2. TRADE LOGIC
    if signal == 1: # BUY SIGNAL 🟢
        if current_qty == 0:
            print(f"   🟢 BUY {symbol} (Conf: {probability[2]*100:.1f}%) -> Entry.")
            submit_order(OrderSide.BUY)
        elif current_qty < 0:
            print(f"   🟢 BUY {symbol} (Conf: {probability[2]*100:.1f}%) -> Closing Short.")
            trading_client.close_position(symbol)

    elif signal == -1: # SELL SIGNAL 🔴
        if current_qty > 0:
            print(f"   🔴 SELL {symbol} (Conf: {probability[0]*100:.1f}%) -> Liquidating.")
            trading_client.close_position(symbol)
        elif current_qty == 0:
            print(f"   🔴 SHORT {symbol} (Conf: {probability[0]*100:.1f}%) -> Entry.")
            submit_order(OrderSide.SELL)
            
    else:
        print(f"   ⚪ Neutral on {symbol}. No action.")

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    # FORCE TEST MODE

    
    clock = trading_client.get_clock()
    if clock.is_open or 'is_test' in locals():
        print(f"--- STARTING BATCH SCAN: {len(SYMBOLS)} STOCKS ---")
        
        for stock in SYMBOLS:
            try:
                
                row = get_market_data(stock)
                
            
                pred, prob = make_prediction(row)
                
                
                execute_trade(pred, prob, stock)
                
                
                time.sleep(1) 
                
            except Exception as e:
                print(f"⚠️ Error with {stock}: {e}")
                
        print("\n--- SCAN COMPLETE ---")
    else:
        print("💤 Market is closed. Bot is sleeping.")
