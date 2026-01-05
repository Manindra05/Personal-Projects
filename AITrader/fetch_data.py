import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# 1. SETUP
API_KEY = "PKZLAB27TCM4V4GIMIQPI5CQ5C"
SECRET_KEY = "28FZZ2Xrr7QYnoR5xYkoLbqFiaJX5HbAtr6vHo3WGXwH"
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 2. CONFIGURE THE REQUEST
# We pull 2 years of daily data for a mix of assets
request_params = StockBarsRequest(
    symbol_or_symbols=["SPY", "QQQ", "AAPL", "TSLA"],
    timeframe=TimeFrame.Day,
    start=datetime(2023, 1, 1),
    adjustment='split' # Important: adjusts for stock splits so the AI isn't confused
)

print("Fetching data from Alpaca...")
bars = client.get_stock_bars(request_params)

# 3. CONVERT TO DATAFRAME
df = bars.df.reset_index()

# 4. FEATURE ENGINEERING (Teaching the model to "see")
# We calculate features per stock symbol
def add_features(group):
    group = group.copy()
    # Trend Indicator: SMA 50 vs 200
    group['SMA_50'] = ta.sma(group['close'], length=50)
    group['SMA_200'] = ta.sma(group['close'], length=200)
    # Momentum Indicator: RSI
    group['RSI'] = ta.rsi(group['close'], length=14)
    # Volatility Indicator: ATR (Average True Range)
    group['ATR'] = ta.atr(group['high'], group['low'], group['close'], length=14)
    return group

df = df.groupby('symbol', group_keys=False).apply(add_features)

# 5. SAVE FOR PHASE 2
df.to_csv("data/trading_data_v1.csv", index=False)
print("SUCCESS: Phase 1 complete. 'trading_data_v1.csv' created.")
print(df.tail(10))