import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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
# Ridge Regression
# --------------------------------------
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)

# --------------------------------------
# Evaluation
# --------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, preds))
directional_acc = np.mean(np.sign(preds) == np.sign(y_test))

print("\nRidge Regression Results")
print("RMSE:", rmse)
print("Directional Accuracy:", directional_acc)

# Feature coefficients
feature_coefficients = pd.DataFrame({
    'feature': features,
    'coefficient': ridge.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(feature_coefficients)
print("Intercept:", ridge.intercept_)

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
plt.title("Ridge Regression: Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 2: Feature Coefficients
plt.subplot(1, 2, 2)
plt.barh(feature_coefficients['feature'], feature_coefficients['coefficient'])
plt.title("Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_regression_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as: ridge_regression_analysis.png")
plt.show()

# --------------------------------------
# Optional: Regularization Sweep
# --------------------------------------
print("\n" + "="*50)
print("Alpha Comparison:")

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    preds_alpha = model.predict(X_test)
    rmse_alpha = np.sqrt(mean_squared_error(y_test, preds_alpha))
    dir_acc_alpha = np.mean(np.sign(preds_alpha) == np.sign(y_test))

    print(f"alpha={alpha:.2f} | RMSE={rmse_alpha:.6f} | Dir_Acc={dir_acc_alpha:.3f}")