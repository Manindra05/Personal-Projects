import pandas as pd
import pandas_ta as ta
import joblib
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# --- CONFIG ---
API_KEY = "PKZLAB27TCM4V4GIMIQPI5CQ5C"       # <--- PASTE YOUR KEY HERE
SECRET_KEY = "28FZZ2Xrr7QYnoR5xYkoLbqFiaJX5HbAtr6vHo3WGXwH" # <--- PASTE YOUR SECRET HERE
SYMBOL = "GLD"                 # We will predict the S&P 500

# 1. GET THE LATEST DATA
# We need enough history to calculate the 200 SMA (at least 200 days)
print(f"Fetching data for {SYMBOL}...")
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Look back 300 days to be safe
start_date = datetime.now() - timedelta(days=400) 
request_params = StockBarsRequest(
    symbol_or_symbols=[SYMBOL],
    timeframe=TimeFrame.Day,
    start=start_date,
    adjustment='split'
)

bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()

# 2. CALCULATE INDICATORS (Must match training exactly)
# The AI only knows SMA_50, SMA_200, RSI, ATR. We must provide exactly that.
df['SMA_50'] = ta.sma(df['close'], length=50)
df['SMA_200'] = ta.sma(df['close'], length=200)
df['RSI'] = ta.rsi(df['close'], length=14)
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# Get the very last row (Today's market close)
latest_data = df.iloc[-1:].copy()
print("\n--- TODAY'S MARKET VITALS ---")
print(latest_data[['timestamp', 'close', 'RSI', 'SMA_50']].to_string(index=False))

# 3. LOAD THE BRAIN
print("\nLoading AI Brain...")
model = joblib.load("trading_model.pkl")

# 4. PREDICT
# We only pass the columns the AI was trained on
features = ['SMA_50', 'SMA_200', 'RSI', 'ATR']
prediction = model.predict(latest_data[features])
probability = model.predict_proba(latest_data[features])

# 5. THE VERDICT
print("\n--------------------------------")
print(f"AI PREDICTION FOR {SYMBOL}:")
if prediction[0] == 1:
    print(f"🟢 BUY (Confidence: {probability[0][2]*100:.1f}%)")
    print("The AI thinks price will rise > 2% soon.")
elif prediction[0] == -1:
    print(f"🔴 SELL / AVOID (Confidence: {probability[0][0]*100:.1f}%)")
    print("The AI warns of a drop.")
else:
    print(f"⚪ HOLD / NEUTRAL")
    print("No strong signal detected.")
print("--------------------------------")