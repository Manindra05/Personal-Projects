import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pandas_ta as ta 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  
import pandas_ta as ta 



# 2. LOAD DATA
df = pd.read_csv("data/labeled_trading_data.csv")

# 1. CALCULATE BOTH INDICATORS
df['EMA_20'] = ta.ema(df['close'], length=20)
df['SMA_200'] = ta.sma(df['close'], length=200) 

df['RSI'] = ta.rsi(df['close'], length=14)
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

df = df.dropna()

# 2. GIVE THE AI BOTH TOOLS
features = ['EMA_20', 'SMA_200', 'RSI', 'ATR']

X = df[features]
y = df['target']


# 3. SPLIT INTO TRAINING (80%) AND TESTING (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL
print("Training the AI...")
model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
model.fit(X_train, y_train)

# 5. TEST THE MODEL
print("Evaluating performance...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"--- RESULTS ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# 6. SAVE THE BRAIN
joblib.dump(model, "trading_model.pkl")
print("Saved model to 'trading_model.pkl'")
