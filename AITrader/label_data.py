import pandas as pd
import numpy as np

# 1. LOAD PHASE 1 DATA
df = pd.read_csv("data/trading_data_v1.csv")

# 2. THE LABELING ENGINE
def label_row(df, index, horizon=5, target_pct=0.02):
    """
    Looks forward 'horizon' days to see if we hit a 2% profit target.
    """
    if index + horizon >= len(df):
        return 0 # Not enough data to look forward
    
    current_price = df.iloc[index]['close']
    future_prices = df.iloc[index + 1 : index + horizon + 1]['close']
    
    max_future_return = (future_prices.max() - current_price) / current_price
    min_future_return = (future_prices.min() - current_price) / current_price
    
    # Simple Labeling: 1 if it goes up 2%, -1 if it drops 1% first
    if max_future_return >= target_pct:
        return 1  # Success!
    elif min_future_return <= -0.01:
        return -1 # Stop Loss Hit
    else:
        return 0  # Nothing happened (Neutral)

print("Labeling data... this might take a minute.")
df['target'] = [label_row(df, i) for i in range(len(df))]

# 3. SAVE TO NEW FILE
df.to_csv("data/labeled_trading_data.csv", index=False)
print("Phase 2 Complete! Your AI now has 'answers' to learn from.")