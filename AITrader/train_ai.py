import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # This is what saves the "Brain" as a file

# 1. LOAD THE DATA
print("Loading labeled data...")
df = pd.read_csv("data/labeled_trading_data.csv")

# Clean up: Drop rows with NaN values (common in the first 200 rows due to SMA calculations)
df = df.dropna()

# 2. SEPARATE "QUESTIONS" (Features) FROM "ANSWERS" (Target)
# We only want the AI to see the technical indicators
feature_cols = ['SMA_50', 'SMA_200', 'RSI', 'ATR']
X = df[feature_cols]
y = df['target']

# 3. SPLIT INTO TRAINING (80%) AND TESTING (20%)
# random_state=42 ensures we get the same split every time (good for science!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL (The "Study" Phase)
# n_estimators=100 means "Create 100 mini-decision trees and vote on the answer"
print("Training the AI... (This plays the Rocky montage music)")
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
# This creates the physical file you can use later
joblib.dump(model, "trading_model.pkl")
print("Saved model to 'trading_model.pkl'")