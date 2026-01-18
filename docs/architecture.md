### Full-Stack Architecture ###
-------------------------------

Main Responsibility:
- UI manages state, controls, and visualization ()
- Backend handles data loading, feature construction, training, evaluation
- Models are encapsulated and reproducible
- Caching prevents redundant computation



## Dataset
- File: `sp500_2000_2019_top10.csv`
- Columns:
  - Date
  - Symbol
  - Open, High, Low, Close
  - Volume



## Prediction Target
The model predicts returns over a user-selected future horizon which is configurable through a UI slider.
All evaluation metrics are computed strictly on the prediction period corresponding to the chosen horizon.
Look-ahead bias is not allowed.



## Feature Selections
Users control model inputs through intuitive feature sliders.
Each feature summarizes past market behavior. (no future information is used)
Changing a feature window affects how much past data the model considers.
These feature windows do NOT change the prediction horizon.

| User Control | Internal Name | What It Measures | Typical Range |
|-------------|---------------|------------------|---------------|
| Previous Day Return | `return_lag_1` | How much the price changed yesterday | Fixed |
| Past Return (N days ago) | `return_lag_N` | Price change from N trading days ago | 2–20 days |
| Momentum Window | `momentum_N` | Total price change over the last N days | 5–30 days |
| Trend Strength | `ma_ratio_N` | How far price is above or below its N-day average | 5–50 days |
| Market Volatility | `volatility_N` | How unstable price has been recently | 5–30 days |
| Volatility Shift | `volatility_ratio` | Short-term volatility vs long-term volatility | (5 / 20), (10 / 50) |
| Volume Surge | `volume_ratio` | How today’s trading volume compares to normal | 5–30 days |
| Price Position | `bb_position` | Whether price is near the top or bottom of its recent range | 20 days |
| Intraday Pressure | `close_open_ratio` | Whether buyers or sellers dominated today | Fixed |

Feature window sizes are adjustable through the UI and help users explore
how short-term vs long-term market behavior affects predictions.

Notes:
- `N` is chosen via sliders but passed as integers internally
- Features are dropped automatically if insufficient history exists
- Feature engineering must be deterministic and stateless
- Features must be intuitive for beginner friendly user



## Machine Learning Models
Ridge Regression:
- Linear model with L2 regularization
- Used for interpretability
- Coefficients are exposed
- Regularization controlled via `alpha`

Random Forest:
- Non-linear ensemble model
- Used for comparison
- Feature importance exposed
- Fixed random seed by default for reproducibility



## User Interaction Rules
- Changing UI controls does not retrain models automatically
- Models train only when the user clicks "Train"
- Identical parameter combinations should return cached results
- Heavy jobs may be queued



## Train and Prediction Split
Models are trained and evaluated using a time-based split.
A single cutoff date separates historical training data from future prediction data.

The cutoff date:
- Defines the last day used for model training
- Marks the start of the prediction period
- Is visualized as a shaded region on all charts
- Training data is never used for evaluation.



### Primary Metrics
Each training run returns the following outputs:
- Time-indexed predicted returns for the prediction period
- Paired with actual returns for comparison
- Used for chart rendering

Metrics:
- RMSE
- Directional Accuracy

Ridge Regression:
- Feature coefficients
- Coefficient magnitude indicates influence strength
- Coefficient sign indicates positive or negative relationship

Random Forest:
- Feature importance scores
- Indicates relative contribution of each feature

Plain-English summary of model behavior and performance



## API Flow
1. Client requests available datasets and symbols
2. Client submits a training request with model and feature parameters
3. Backend validates inputs and checks cache
4. If cached, results are returned immediately
5. If not cached, a training job is executed
6. Results are stored and returned to the client



## Project Structure

The codebase is organized by responsibility so that data loading,
feature generation, model logic, evaluation, and API orchestration
remain explicit, interpretable, and reproducible.

src/
  data/
    loader.py            # Load CSV data, filter by symbol and date range,
                         # enforce time ordering and basic validation

  features/
    returns.py           # Return computation and lagged return features
    momentum.py          # Momentum and cumulative return features
    rolling.py           # Rolling statistics (mean, volatility, ratios)
    volume.py            # Volume-based features
    indicators.py        # Optional technical indicators (RSI, Bollinger, etc.)

  models/
    ridge.py             # Ridge regression (linear, interpretable, coefficients)
    random_forest.py     # Random Forest regressor (non-linear baseline)
    base.py              # Shared model interface (fit, predict, explain)

  evaluation/
    metrics.py           # RMSE, directional accuracy
    baselines.py         # Naive baselines (e.g. zero-return predictor)

  api/
    train.py             # /train endpoint:
                         # - request validation
                         # - feature construction
                         # - model training and evaluation
    status.py            # /status endpoint for long-running jobs
    results.py           # /results endpoint returning predictions and metrics

  cache/
    keys.py              # Deterministic cache key construction
    store.py             # Cache interface (in-memory or Redis)

  jobs/
    runner.py            # Optional background job execution logic
    queue.py             # Job queue abstraction (if enabled)

  config.py              # Global configuration (paths, seeds, limits)
  schemas.py             # Request / response schemas for API payloads

docs/
  architecture.md        # System design, data flow, and interaction mapping
  api.md                 # API behavior and endpoint documentation


## Reference
See `docs/architecture.md` for the full system design and interaction mapping.