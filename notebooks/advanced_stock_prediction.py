import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
HORIZON = 100  # Predict 30-day returns (configurable)


def create_advanced_features(df, use_predicted_returns=False):
    """Create sophisticated technical indicators
    
    Args:
        df: DataFrame with OHLCV data (and optionally 'predicted_return' column)
        use_predicted_returns: If True, use predicted_return col for lags; else use actual returns
    """
    # Determine which return series to use for lags
    if use_predicted_returns and 'predicted_return' in df.columns:
        return_series = df['predicted_return'].copy()
    else:
        return_series = df['Close'].pct_change()
    
    df['return'] = return_series

    # Momentum indicators (use return_series which may be predicted)
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'return_lag_{lag}'] = return_series.shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag) / df['Volume'].shift(lag).rolling(20).mean()

    # Moving averages and trends (always use actual Close)
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'ma_ratio_{window}'] = df['Close'] / df['Close'].rolling(window=window).mean()
        df[f'ma_trend_{window}'] = df['Close'].rolling(window=window).mean().pct_change(5)

    # Volatility measures
    df['volatility_5'] = return_series.rolling(5).std()
    df['volatility_20'] = return_series.rolling(20).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

    # Price position relative to range
    df['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['close_open_ratio'] = df['Close'] / df['Open']

    # Volume indicators
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_trend'] = df['Volume'].rolling(5).mean().pct_change(5)

    # RSI (Relative Strength Index)
    delta = return_series.diff()
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


def prepare_data(horizon=HORIZON):
    """Load and prepare data with advanced features and horizon-based target
    
    Creates future_return_Nd which is the cumulative return over next N days.
    Only uses data before the horizon window for features (no look-ahead bias).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../data/sample/snp500_ohlc_2000_2019_top10.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[df['symbol'] == 'AAPL'].copy()
    df.sort_values("Date", inplace=True)
    df = df.head(1500)

    # Create basic returns for feature engineering
    df['return'] = df['Close'].pct_change()
    
    # Create the HORIZON-based target: cumulative return over next N days
    # This is what we actually want to predict
    df[f'future_return_{horizon}d'] = df['Close'].shift(-horizon).pct_change(horizon)
    
    # Create advanced features (only from historical data, no future data)
    df = create_advanced_features(df, use_predicted_returns=False)
    
    # Drop rows where we don't have enough past data for features or future data for target
    df = df.dropna()
    
    return df, horizon


def train_models(X_train, y_train):
    """Train multiple models on training set (before cutoff date)"""
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


def predict_with_rolling_features(df, models, cutoff_idx, features, horizon):
    """
    Predict on test set (after cutoff) using rolling feature calculation.
    
    For each prediction in the test set:
    1. Use actual/predicted data available up to that point
    2. Make prediction for next HORIZON days
    3. Use that prediction to construct features for the next position
    
    Args:
        df: Full dataframe with training + test data
        models: Dict of trained models
        cutoff_idx: Index where training ends and test begins
        features: List of feature column names
        horizon: Number of days ahead to predict
    
    Returns:
        Dict with predictions and metadata for each model
    """
    results = {name: {'predictions': [], 'dates': []} for name in models.keys()}
    
    # For test set predictions, we'll predict one window at a time
    # and use predicted returns to update features
    test_df = df.iloc[cutoff_idx:].copy().reset_index(drop=True)
    
    # Keep track of whether we're using predicted or actual returns for features
    full_df = df.iloc[:cutoff_idx + len(test_df)].copy()
    
    for test_pos in range(len(test_df)):
        current_idx = cutoff_idx + test_pos
        
        # For feature calculation at this point, use actual data up to cutoff
        # then predicted data for anything after cutoff
        if current_idx >= len(full_df):
            # Extend full_df with predicted data for feature engineering
            full_df = full_df.iloc[:current_idx].copy()
        
        # Get features from the data point at current_idx - horizon
        # (we want to predict the next horizon days starting from current_idx)
        feat_idx = current_idx - 1
        
        if feat_idx < 0:
            continue
            
        feat_row = full_df.iloc[feat_idx]
        X_point = feat_row[features].values.reshape(1, -1)
        
        # Predict with all models
        for name, (model, scaler) in models.items():
            if scaler:
                X_point_scaled = scaler.transform(X_point)
                pred = model.predict(X_point_scaled)[0]
            else:
                pred = model.predict(X_point)[0]
            
            results[name]['predictions'].append(pred)
            results[name]['dates'].append(full_df.iloc[feat_idx]['Date'])
        
        # Update full_df with predicted return for next feature calculation
        # Convert horizon-based prediction to approximate daily return for features
        if test_pos < len(test_df) - 1:
            # Average daily return approximation from horizon prediction
            avg_daily_return = (1 + results['rf']['predictions'][-1]) ** (1/horizon) - 1
            
            # Add a new row with predicted return for feature calculation
            next_row = full_df.iloc[current_idx].copy() if current_idx < len(full_df) else test_df.iloc[test_pos].copy()
            next_row['predicted_return'] = avg_daily_return
            next_row['Close'] = full_df.iloc[current_idx-1]['Close'] * (1 + avg_daily_return) if current_idx > 0 else test_df.iloc[test_pos]['Close']
            
            # Recalculate features for next iteration using predicted data
            if current_idx < len(full_df):
                # Update the actual dataframe for next feature calc
                pass
    
    return results


def evaluate_models(df, models, cutoff_idx, features, horizon):
    """Evaluate all models on test set"""
    target_col = f'future_return_{horizon}d'
    test_df = df.iloc[cutoff_idx:].copy()
    y_test = test_df[target_col].values
    dates_test = test_df['Date'].values
    
    results = {}
    
    for name, (model, scaler) in models.items():
        # Get features for test set
        X_test = test_df[features]
        
        # Make predictions
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        directional_acc = np.mean(np.sign(preds) == np.sign(y_test))
        
        results[name] = {
            'rmse': rmse,
            'directional_acc': directional_acc,
            'predictions': preds,
            'actual': y_test,
            'dates': dates_test
        }
    
    return results


# Main analysis
print("🔬 Advanced Stock Prediction Analysis (Horizon-Based)")
print("=" * 60)

# Prepare data with horizon-based target
df, horizon = prepare_data(horizon=HORIZON)
print(f"📊 Data prepared: {len(df)} trading days")
print(f"🎯 Prediction horizon: {horizon} days")

# Feature selection (exclude target and non-feature columns)
target_col = f'future_return_{horizon}d'
exclude_features = ['Date', 'symbol', 'return', 'Close', 'High', 'Low', 'Open', 'Volume', 
                    target_col, 'predicted_return']
features = [col for col in df.columns if col not in exclude_features]
print(f"🎯 Using {len(features)} features")

# Train/test split at a specific cutoff date (80/20 split)
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['Date']
print(f"\n📅 Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"   Training: {cutoff_idx} samples (up to cutoff)")
print(f"   Testing:  {len(df) - cutoff_idx} samples (after cutoff)")

# Training data (BEFORE cutoff - no future data used)
X_train = df.iloc[:cutoff_idx][features]
y_train = df.iloc[:cutoff_idx][target_col]

# Test data (AFTER cutoff)
X_test = df.iloc[cutoff_idx:][features]
y_test = df.iloc[cutoff_idx:][target_col]
dates_test = df.iloc[cutoff_idx:]['Date'].values

print(f"\n📊 Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"                X_test={X_test.shape}, y_test={y_test.shape}")

# Train models
print("\n🤖 Training models on training set only...")
models = train_models(X_train, y_train)
print("   ✓ Ridge Regression trained")
print("   ✓ Random Forest trained")
print("   ✓ Gradient Boosting trained")

# Evaluate models (using actual test data for now)
print("\n📈 Evaluating on test set...")
results = evaluate_models(df, models, cutoff_idx, features, horizon)

# Print results
print("\n🏆 Model Performance Comparison (Predicting {}-day returns)".format(horizon))
print("-" * 60)

for name, result in results.items():
    print(f"  {name.upper():6s}: RMSE={result['rmse']:.6f}, DirAcc={result['directional_acc']:.3f}")

# Feature importance analysis
rf_model, _ = models['rf']
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Select a representative window - show 10 actual days with multiple 10-day predictions
window_start_idx = cutoff_idx + 50  # Start 50 days into test set for variety
window_dates = df['Date'].iloc[window_start_idx:window_start_idx+horizon].values
actual_prices_window = df['Close'].iloc[window_start_idx:window_start_idx+horizon].values

# Plot 1: Show multiple 10-day predictions starting from different dates within the window
ax = axes[0, 0]
# Show overlapping predictions: each starts at different dates, all predict 10 days ahead
for start_offset in range(0, horizon, 2):  # Every 2 days, show a prediction
    pred_start_idx = window_start_idx - horizon + start_offset
    pred_end_idx = pred_start_idx + horizon
    
    if pred_start_idx >= cutoff_idx and pred_end_idx < len(df):
        pred_dates = df['Date'].iloc[pred_start_idx:pred_end_idx].values
        # Use RF predictions
        rf_pred = results['rf']['predictions'][pred_start_idx - cutoff_idx]
        actual_return_10d = df[f'future_return_{horizon}d'].iloc[pred_start_idx]
        
        # Reconstruct 10-day price path from starting price
        start_price = df['Close'].iloc[pred_start_idx]
        end_price = start_price * (1 + actual_return_10d)
        
        ax.plot(pd.to_datetime(pred_dates), 
               np.linspace(start_price, end_price, horizon),
               alpha=0.5, linewidth=1, label=f'Actual from {pd.Timestamp(pred_dates[0]).strftime("%Y-%m-%d")}')

ax.axhline(y=df['Close'].iloc[window_start_idx], color='black', linestyle='-', linewidth=3, alpha=0.7, label='Reference Price')
ax.set_title(f'Multiple {horizon}-Day Predictions Overlaid\n(Each starts at different date, covers {horizon} days)')
ax.set_xlabel(f'Date (spans {horizon} calendar days)')
ax.set_ylabel('Price ($)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Show what a single 30-day horizon prediction looks like - ALL MODELS
ax = axes[0, 1]
single_pred_idx = cutoff_idx + 30  # Pick a single prediction point
actual_10d_return = df[f'future_return_{horizon}d'].iloc[single_pred_idx]

start_price = df['Close'].iloc[single_pred_idx]
actual_end_price = start_price * (1 + actual_10d_return)

pred_dates_10d = df['Date'].iloc[single_pred_idx:single_pred_idx+horizon].values
actual_prices_10d = df['Close'].iloc[single_pred_idx:single_pred_idx+horizon].values

# Plot the actual 30-day price path
ax.plot(pd.to_datetime(pred_dates_10d), actual_prices_10d, 
        marker='o', linewidth=3, markersize=6, label='Actual price path', color='black')

# Plot predictions from all three models
colors = {'ridge': 'blue', 'rf': 'orange', 'gb': 'green'}
for model_name in ['ridge', 'rf', 'gb']:
    pred_return = results[model_name]['predictions'][single_pred_idx - cutoff_idx]
    pred_end_price = start_price * (1 + pred_return)
    
    ax.plot(pd.to_datetime(pred_dates_10d), 
            np.linspace(start_price, pred_end_price, horizon),
            marker='s', linewidth=2, markersize=4, linestyle='--', 
            label=f'{model_name.upper()} predicted path', 
            color=colors[model_name], alpha=0.7)

ax.axvline(x=pd.Timestamp(pred_dates_10d[0]), color='green', linestyle='--', alpha=0.5, linewidth=2, label='Prediction Made Here')
ax.axvline(x=pd.Timestamp(pred_dates_10d[-1]), color='red', linestyle='--', alpha=0.5, linewidth=2, label='Horizon Endpoint')
ax.set_title(f'Example: Single {horizon}-Day Prediction - All Models Compared\n(Prediction made on {pd.Timestamp(pred_dates_10d[0]).strftime("%Y-%m-%d")}, covers {horizon} calendar days)')
ax.set_xlabel(f'Date (spans exactly {horizon} days)')
ax.set_ylabel('Price ($)')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Aggregate performance across all predictions - ALL MODELS
ax = axes[1, 0]
# Show predicted vs actual 30-day returns scatter for all three models
colors = {'ridge': 'blue', 'rf': 'orange', 'gb': 'green'}

for model_name in ['ridge', 'rf', 'gb']:
    ax.scatter(results[model_name]['actual'], results[model_name]['predictions'], 
              alpha=0.5, s=40, label=f'{model_name.upper()}', color=colors[model_name])

ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', linewidth=1, alpha=0.5, label='Perfect Prediction')
ax.set_xlabel(f'Actual {horizon}-Day Return')
ax.set_ylabel(f'Predicted {horizon}-Day Return')
ax.set_title(f'All {horizon}-Day Return Predictions vs Actual\n(All three models, {len(results["ridge"]["actual"])} predictions each)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 4: Feature importance
ax = axes[1, 1]
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features['feature']])
ax.set_title('Top 10 Feature Importance (Random Forest)')
ax.set_xlabel('Importance Score')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('advanced_model_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Advanced analysis plot saved as: advanced_model_comparison.png")

# Summary insights
print("\n💡 Key Insights:")
print(f"• Predicting {horizon}-day returns is harder than 1-day returns")
print("• Longer horizons reduce momentum-based predictability")
print("• Models trained only on pre-cutoff data avoid look-ahead bias")
print("• For production: use time-series cross-validation with rolling windows")
print("• Consider ensemble methods combining multiple predictions")

print("\n✅ Analysis complete!")
print(f"   • Models trained on {cutoff_idx} days of history")
print(f"   • Predictions made for {len(dates_test)} test days")
print(f"   • Each prediction covers {horizon} days ahead")