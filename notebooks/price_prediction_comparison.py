import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def prepare_data():
    """Prepare AAPL data with features"""
    df = pd.read_csv("../data/sample/snp500_ohlc_2000_2019_top10.csv", parse_dates=["Date"])
    df = df[df['symbol'] == 'AAPL'].copy()
    df.sort_values("Date", inplace=True)
    df = df.head(1000)

    # Feature Engineering
    df['return'] = df['Close'].pct_change()
    df['lag_1'] = df['return'].shift(1)
    df['lag_2'] = df['return'].shift(2)
    df['lag_5'] = df['return'].shift(5)
    df['lag_10'] = df['return'].shift(10)
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['volatility'] = df['return'].rolling(window=5).std()
    df.dropna(inplace=True)

    return df


def train_and_predict(df, model_type='ridge'):
    """Train model and return predictions"""
    features = ['lag_1', 'lag_2', 'lag_5', 'lag_10', 'ma_5', 'ma_10', 'volatility']
    X = df[features]
    y = df['return']

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    else:  # random forest
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return df, split, y_test, preds


def convert_to_prices(df, split, preds):
    """Convert return predictions to price predictions"""
    last_train_price = df['Close'].iloc[split-1]
    pred_returns_cumsum = np.cumsum(preds)
    pred_prices = last_train_price * (1 + pred_returns_cumsum)
    actual_prices = df['Close'].iloc[split:].values

    return actual_prices, pred_prices


# Prepare data
df = prepare_data()

# Train both models
df_ridge, split_ridge, y_test_ridge, preds_ridge = train_and_predict(df, 'ridge')
df_rf, split_rf, y_test_rf, preds_rf = train_and_predict(df, 'rf')

# Convert to price predictions
actual_prices_ridge, pred_prices_ridge = convert_to_prices(df_ridge, split_ridge, preds_ridge)
actual_prices_rf, pred_prices_rf = convert_to_prices(df_rf, split_rf, preds_rf)

# Create comparison plot
plt.figure(figsize=(16, 8))

# Ridge Regression Price Chart
plt.subplot(2, 2, 1)
plt.plot(df_ridge['Date'].iloc[split_ridge:], actual_prices_ridge,
         label='Actual Price', linewidth=2, color='blue')
plt.plot(df_ridge['Date'].iloc[split_ridge:], pred_prices_ridge,
         label='Ridge Predicted', linewidth=2, color='red', linestyle='--')
plt.legend()
plt.title("Ridge Regression: Actual vs Predicted AAPL Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Random Forest Price Chart
plt.subplot(2, 2, 2)
plt.plot(df_rf['Date'].iloc[split_rf:], actual_prices_rf,
         label='Actual Price', linewidth=2, color='blue')
plt.plot(df_rf['Date'].iloc[split_rf:], pred_prices_rf,
         label='RF Predicted', linewidth=2, color='green', linestyle='--')
plt.legend()
plt.title("Random Forest: Actual vs Predicted AAPL Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Overlapping Price Comparison
plt.subplot(2, 1, 2)
plt.plot(df_ridge['Date'].iloc[split_ridge:], actual_prices_ridge,
         label='Actual Price', linewidth=3, color='black')
plt.plot(df_ridge['Date'].iloc[split_ridge:], pred_prices_ridge,
         label='Ridge Predicted', linewidth=2, color='red', linestyle='--')
plt.plot(df_rf['Date'].iloc[split_rf:], pred_prices_rf,
         label='Random Forest Predicted', linewidth=2, color='green', linestyle='-.')
plt.legend()
plt.title("AAPL Price Prediction Comparison: Ridge vs Random Forest")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print performance metrics
print("="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

rmse_ridge = np.sqrt(mean_squared_error(y_test_ridge, preds_ridge))
dir_acc_ridge = np.mean(np.sign(preds_ridge) == np.sign(y_test_ridge))

rmse_rf = np.sqrt(mean_squared_error(y_test_rf, preds_rf))
dir_acc_rf = np.mean(np.sign(preds_rf) == np.sign(y_test_rf))

print("Ridge Regression:")
print(f"  RMSE: {rmse_ridge:.6f}")
print(f"  Directional Accuracy: {dir_acc_ridge:.3f}")
print()
print("Random Forest:")
print(f"  RMSE: {rmse_rf:.6f}")
print(f"  Directional Accuracy: {dir_acc_rf:.3f}")
print()
print(f"Test period: {len(y_test_ridge)} trading days")
print(f"Price range: ${actual_prices_ridge[0]:.2f} to ${actual_prices_ridge[-1]:.2f}")