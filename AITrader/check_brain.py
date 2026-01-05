import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. LOAD THE BRAIN
model = joblib.load("trading_model.pkl")
features = ['SMA_50', 'SMA_200', 'RSI', 'ATR']

# 2. EXTRACT IMPORTANCE
# The Random Forest can tell us which column helped it decide the most
importances = model.feature_importances_
indices = np.argsort(importances)

# 3. VISUALIZE
plt.figure(figsize=(10, 6))
plt.title('What Does Your AI Care About?')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()