import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv("../data/sample/snp500_ohlc_2000_2019_top10.csv", parse_dates=["Date"])
# Select AAPL for the example
df = df[df['symbol'] == 'AAPL'].copy()
df.sort_values("Date", inplace=True)

# Take a small fraction for testing (first 1000 rows for more data)
df = df.head(1000)

print("Data preview:")
print(df.head())

# --------------------------------------
# Feature Engineering
# --------------------------------------
# Compute daily returns
df['return'] = df['Close'].pct_change()

# Create lag features
df['lag_1'] = df['return'].shift(1)
df['lag_2'] = df['return'].shift(2)
df['lag_5'] = df['return'].shift(5)
df['lag_10'] = df['return'].shift(10)

# Add some technical indicators
df['ma_5'] = df['Close'].rolling(window=5).mean()
df['ma_10'] = df['Close'].rolling(window=10).mean()
df['volatility'] = df['return'].rolling(window=5).std()

# Drop rows with NaN (from lagging and rolling)
df.dropna(inplace=True)

# Define features and target
features = ['lag_1', 'lag_2', 'lag_5', 'lag_10', 'ma_5', 'ma_10', 'volatility']
X = df[features]
y = df['return']

print("\nFeature preview:")
print(X.head())

# --------------------------------------
# Train/Test Split (time-aware)
# --------------------------------------
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --------------------------------------
# Random Forest Regression
# --------------------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

# --------------------------------------
# Evaluation
# --------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, preds))
directional_acc = np.mean(np.sign(preds) == np.sign(y_test))

print("\nRandom Forest Regression Results")
print("RMSE:", rmse)
print("Directional Accuracy:", directional_acc)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# --------------------------------------
# Convert Returns to Price Predictions
# --------------------------------------
# Get the last training price to start predictions
last_train_price = df['Close'].iloc[split-1]

# Convert predicted returns to cumulative returns, then to prices
pred_returns_cumsum = np.cumsum(preds)
pred_prices = last_train_price * (1 + pred_returns_cumsum)

# Actual prices for test period
actual_prices = df['Close'].iloc[split:].values

print(f"Test period price range: ${actual_prices[0]:.2f} to ${actual_prices[-1]:.2f}")

# --------------------------------------
# Plot Predictions vs Actual (Prices)
# --------------------------------------
plt.figure(figsize=(14, 6))

# Plot 1: Price Predictions vs Actual
plt.subplot(1, 2, 1)
plt.plot(df['Date'].iloc[split:], actual_prices, label='Actual Price', linewidth=2, color='blue')
plt.plot(df['Date'].iloc[split:], pred_prices, label='Predicted Price', linewidth=2, color='red', linestyle='--')
plt.legend()
plt.title("Random Forest: Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_forest_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as: random_forest_analysis.png")
plt.show()

# --------------------------------------
# Optional: Hyperparameter Tuning Comparison
# --------------------------------------
print("\n" + "="*50)
print("Hyperparameter Comparison:")

configs = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 15},
]

for config in configs:
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds_config = model.predict(X_test)
    rmse_config = np.sqrt(mean_squared_error(y_test, preds_config))
    dir_acc_config = np.mean(np.sign(preds_config) == np.sign(y_test))

    print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']} | RMSE={rmse_config:.6f} | Dir_Acc={dir_acc_config:.3f}")

    print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']} | RMSE={rmse_config:.6f} | Dir_Acc={dir_acc_config:.3f}")