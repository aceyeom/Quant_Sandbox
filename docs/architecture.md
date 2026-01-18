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

### Data Management & Freshness
- The dataset is static and covers 2000–2019 (historical reference data)
- On each training request, the API loads the full dataset into memory
- The cutoff date determines the train/test boundary and must be strictly enforced:
  - Training data: all rows with `Date <= cutoff_date`
  - Prediction data: all rows with `Date > cutoff_date`
- Feature engineering always uses data strictly before the cutoff date
- Time ordering is enforced; any gaps or out-of-order dates trigger validation failure



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
- **Critical:** Feature windows do NOT shift with the cutoff date. They always look backward from each date in the training period.
- Features are dropped automatically if insufficient history exists (e.g., if a 30-day momentum is requested but only 20 days of prior data exist before the cutoff, that feature is excluded)
- Feature engineering must be deterministic and stateless
- Features must be intuitive for beginner-friendly users
- **Validation Rule:** If after dropping insufficient-history features, fewer than 2 features remain, the training request is rejected



## Machine Learning Models

### Training Methodology
- **Train/Test Split:** Single cutoff date (user-provided or default)
  - Training set: historical data up to cutoff (fitted on known returns)
  - Test set: data after cutoff (evaluated on actual returns, never used for training)
- **Hyperparameter Handling:**
  - Ridge: `alpha` (L2 regularization strength) is configurable via UI or defaults to 1.0
  - Random Forest: `n_estimators` defaults to 100, `max_depth` defaults to None (full growth), `random_state` fixed to ensure reproducibility
- **No nested cross-validation:** Given the time-series nature of stock data, k-fold CV would introduce look-ahead bias. A single train/test split respects temporal ordering.

### Ridge Regression
- Linear model with L2 regularization
- Used for interpretability and explainability
- Coefficients are exposed to users (positive/negative relationships)
- Regularization strength controlled via `alpha` parameter

### Random Forest
- Non-linear ensemble model
- Used as a non-linear baseline for comparison
- Feature importance scores are exposed (relative contribution of each feature)
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

### Request Validation
Before training begins, inputs are validated:
- Cutoff date is within the dataset's date range
- Selected symbol exists in the dataset
- Feature windows are positive integers within reasonable bounds (e.g., 2–250 days)
- At least 2 valid features remain after dropping those with insufficient history
- Prediction horizon is positive and leaves enough test data

### Training Request Handling
1. Client requests available datasets and symbols
2. Client submits a training request with model type, feature parameters, and cutoff date
3. Backend constructs a deterministic cache key from all parameters
4. Backend validates inputs (see above); returns error immediately if invalid
5. Cache is checked:
   - **If hit:** Results are returned immediately
   - **If miss:** A training job is queued
6. **Async Behavior:** `/train` endpoint returns a job ID immediately (non-blocking)
   - Client polls `/status/{job_id}` to check progress
   - `/results/{job_id}` retrieves completed results (returns 202 Accepted if not ready)
7. Results are stored in cache and returned to client
8. Cached results are reused for identical parameter combinations until cache is cleared



## Project Structure

The codebase is organized by responsibility so that data loading,
feature generation, model logic, evaluation, and API orchestration
remain explicit, interpretable, and reproducible.

```
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
    metrics.py           # RMSE, directional accuracy, baseline comparisons
    baselines.py         # Naive baselines (e.g., zero-return or mean-return predictor)

  api/
    train.py             # POST /train endpoint:
                         # - request validation
                         # - feature construction
                         # - model training and evaluation
                         # - returns job_id immediately
    status.py            # GET /status/{job_id} endpoint for polling long-running jobs
    results.py           # GET /results/{job_id} endpoint returning predictions and metrics

  cache/
    keys.py              # Deterministic cache key construction from parameters
    store.py             # Cache interface abstraction (in-memory or Redis)

  jobs/
    runner.py            # Background job execution logic (training pipeline)
    queue.py             # Job queue abstraction (FIFO or priority-based)

  config.py              # Global configuration (paths, seeds, limits, defaults)
  schemas.py             # Pydantic schemas for request/response validation

docs/
  architecture.md        # System design, data flow, and interaction mapping
  api.md                 # API endpoint documentation and examples
```


## Reference
See `docs/architecture.md` for the full system design and interaction mapping.