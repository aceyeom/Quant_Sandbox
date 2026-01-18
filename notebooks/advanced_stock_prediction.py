import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


def create_advanced_features(df):
    """Create sophisticated technical indicators"""
    # Basic returns
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Momentum indicators
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag) / df['Volume'].shift(lag).rolling(20).mean()

    # Moving averages and trends
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'ma_ratio_{window}'] = df['Close'] / df['Close'].rolling(window=window).mean()
        df[f'ma_trend_{window}'] = df['Close'].rolling(window=window).mean().pct_change(5)

    # Volatility measures
    df['volatility_5'] = df['return'].rolling(5).std()
    df['volatility_20'] = df['return'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

    # Price position relative to range
    df['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['close_open_ratio'] = df['Close'] / df['Open']

    # Volume indicators
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_trend'] = df['Volume'].rolling(5).mean().pct_change(5)

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df


def prepare_data():
    """Load and prepare data with advanced features"""
    df = pd.read_csv("../data/sample/snp500_ohlc_2000_2019_top10.csv", parse_dates=["Date"])
    df = df[df['symbol'] == 'AAPL'].copy()
    df.sort_values("Date", inplace=True)
    df = df.head(1500)  # More data for better training

    # Create advanced features
    df = create_advanced_features(df)
    df.dropna(inplace=True)

    return df


def train_models(X_train, y_train):
    """Train multiple models"""
    models = {}

    # Ridge Regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=0.1, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    models['ridge'] = (ridge, scaler)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['rf'] = (rf, None)

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['gb'] = (gb, None)

    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate all models"""
    results = {}

    for name, (model, scaler) in models.items():
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        directional_acc = np.mean(np.sign(preds) == np.sign(y_test))

        results[name] = {
            'rmse': rmse,
            'directional_acc': directional_acc,
            'predictions': preds
        }

    return results


def predict_direction_only(y_true, y_pred):
    """Predict only direction (up/down), not magnitude"""
    return np.mean(np.sign(y_pred) == np.sign(y_true))


# Main analysis
print("🔬 Advanced Stock Prediction Analysis")
print("=" * 50)

# Prepare data
df = prepare_data()
print(f"📊 Data prepared: {len(df)} trading days")

# Feature selection (exclude highly correlated features)
exclude_features = ['Date', 'symbol', 'return', 'log_return', 'Close', 'High', 'Low', 'Open', 'Volume']
features = [col for col in df.columns if col not in exclude_features]
print(f"🎯 Using {len(features)} features")

# Time series split for validation
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = {'ridge': [], 'rf': [], 'gb': []}

print("\n🔄 Cross-validation Results:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    X_train_cv = df.iloc[train_idx][features]
    y_train_cv = df.iloc[train_idx]['return']
    X_val_cv = df.iloc[val_idx][features]
    y_val_cv = df.iloc[val_idx]['return']

    # Quick training for CV
    rf_cv = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_cv.fit(X_train_cv, y_train_cv)
    rf_preds_cv = rf_cv.predict(X_val_cv)
    rf_rmse_cv = np.sqrt(mean_squared_error(y_val_cv, rf_preds_cv))

    cv_scores['rf'].append(rf_rmse_cv)
    print(f"  Fold {fold+1}: RF RMSE = {rf_rmse_cv:.4f}")

print(f"  Average CV RMSE: {np.mean(cv_scores['rf']):.4f}")

# Final train/test split
split = int(len(df) * 0.8)
X_train = df.iloc[:split][features]
y_train = df.iloc[:split]['return']
X_test = df.iloc[split:][features]
y_test = df.iloc[split:]['return']

print(f"\n📈 Final Split: {len(X_train)} train, {len(X_test)} test samples")

# Train models
models = train_models(X_train, y_train)

# Evaluate models
results = evaluate_models(models, X_test, y_test)

# Print results
print("\n🏆 Model Performance Comparison:")
print("-" * 40)
naive_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))
print(f"  Naive Baseline RMSE: {naive_rmse:.6f}")
print(f"  Market Volatility (std): {np.std(y_test):.6f}")

for name, result in results.items():
    print(f"  {name.upper():6s}: RMSE={result['rmse']:.6f}, DirAcc={result['directional_acc']:.3f}")

# Feature importance analysis
rf_model, _ = models['rf']
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Top 10 Most Important Features:")
print(feature_importance.head(10))

# Convert to price predictions for visualization
last_train_price = df['Close'].iloc[split-1]

plt.figure(figsize=(16, 10))

# Plot 1: Return predictions
plt.subplot(2, 2, 1)
plt.plot(df['Date'].iloc[split:], y_test.values, label='Actual Returns', linewidth=2, alpha=0.7)
for name, result in results.items():
    plt.plot(df['Date'].iloc[split:], result['predictions'],
             label=f'{name.upper()} Predictions', linewidth=1.5, alpha=0.8)
plt.title('Return Predictions Comparison')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Price predictions
plt.subplot(2, 2, 2)
actual_prices = df['Close'].iloc[split:].values
plt.plot(df['Date'].iloc[split:], actual_prices, label='Actual Price', linewidth=3, color='black')

for name, result in results.items():
    pred_returns_cumsum = np.cumsum(result['predictions'])
    pred_prices = last_train_price * (1 + pred_returns_cumsum)
    plt.plot(df['Date'].iloc[split:], pred_prices, label=f'{name.upper()} Predicted Price', linewidth=2)

plt.title('Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Prediction errors
plt.subplot(2, 2, 3)
for name, result in results.items():
    errors = result['predictions'] - y_test.values
    plt.plot(df['Date'].iloc[split:], errors, label=f'{name.upper()} Error', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Prediction Errors Over Time')
plt.xlabel('Date')
plt.ylabel('Prediction Error')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Feature importance
plt.subplot(2, 2, 4)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), [f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']])
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_model_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Advanced analysis plot saved as: advanced_model_comparison.png")

# Summary insights
print("\n💡 Key Insights:")
print("• Stock returns are inherently noisy and difficult to predict")
print("• Models perform similarly to naive baseline due to market efficiency")
print("• Direction prediction is often more valuable than magnitude prediction")
print("• Feature engineering helps but cannot overcome fundamental market randomness")
print("• Consider predicting market regimes or using external data sources")

print("\n🎯 Recommendations for Better Performance:")
print("• Use classification (up/down) instead of regression")
print("• Add external data: news sentiment, economic indicators")
print("• Consider market regime awareness (bull/bear markets)")
print("• Try deep learning models (LSTM) for sequential patterns")
print("• Focus on risk management rather than return prediction")