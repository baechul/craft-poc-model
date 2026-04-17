# craft-poc-model

A collection of Jupyter notebooks demonstrating an end-to-end ML pipeline for sales revenue forecasting — from raw data cleaning through feature engineering to model training and evaluation.

## Overview

This repository is the model development component of the [craft-poc](https://github.com/baechul/craft-poc) application. It covers:

- **Data cleaning & preprocessing** — Standardizing columns, handling missing values, deduplication, and type coercion
- **Feature engineering** — Temporal features (year, month, week, day of week) and lag features (1, 7, 14, 30-day revenue lags) aggregated at the `(date, product_category)` level
- **Model training & evaluation** — Comparing LightGBM and XGBoost regressors using MAE and RMSE metrics

## Notebooks

| Notebook                                                               | Description                                                                             |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [`notebooks/rawdata-cleaning.ipynb`](notebooks/rawdata-cleaning.ipynb) | Loads `Online Sales Data.csv`, cleans and validates the data, and exports `sales.csv`   |
| [`notebooks/train-model.ipynb`](notebooks/train-model.ipynb)           | Builds the feature table, trains LightGBM and XGBoost models, and evaluates predictions |

## Prediction Algorithm

The model forecasts `total_revenue` per `(date, product_category)` using **gradient boosting regression**.

### Input Features

| Feature            | Type        | Description                         |
| ------------------ | ----------- | ----------------------------------- |
| `category_encoded` | categorical | Label-encoded product category      |
| `year`             | temporal    | Calendar year                       |
| `month`            | temporal    | Month of year (1–12)                |
| `week`             | temporal    | ISO week number (1–53)              |
| `day_of_week`      | temporal    | Day of week (0=Monday, 6=Sunday)    |
| `revenue_lag_1`    | lag         | Revenue 1 day prior (same category) |
| `revenue_lag_7`    | lag         | Revenue 7 days prior                |
| `revenue_lag_14`   | lag         | Revenue 14 days prior               |
| `revenue_lag_30`   | lag         | Revenue 30 days prior               |

**Temporal Features** - To have the model learn the time-based seasonal patterns:
- Example: April tends to have higher/lower revenue than January.

**Lag Features** - To have the model learn what happens before:
- Example: If sale was high 7 days ago, it's likely to be high today too.

**Exogenous Features**
In this demo, I didn't add exogenous features but in a real world product, sales could be correlated with the 
interest rates, consumer index or regional holidays.

### Target

`total_revenue` — sum of revenue for a given category on a given date.

### Models Evaluated

| Model                          | Notes                                                                       |
| ------------------------------ | --------------------------------------------------------------------------- |
| **LightGBM** (`LGBMRegressor`) | In general — faster training, better accuracy on this dataset           |
| **XGBoost** (`XGBRegressor`)   | Compared baseline — `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` |

Both models are trained with an 80/20 chronological train/test split (sorted by date to prevent data leakage) and evaluated using **MAE** and **RMSE**.

### Training Pipeline

```
Raw CSV
  └─► Data Cleaning (rawdata-cleaning.ipynb)
        └─► sales.csv
              └─► Feature Engineering (train-model.ipynb)
                    ├─► Encode categories (LabelEncoder)
                    ├─► Add temporal features
                    ├─► Add lag features → drop NaN rows
                    └─► Train/Test Split (80/20, chronological)
                          └─► Model Training & Evaluation (with various model hyperparameters)
```

## Tech Stack

- **Python** 3.14
- **pandas**, **numpy** — Data manipulation
- **scikit-learn** — Preprocessing (`LabelEncoder`) and evaluation metrics
- **LightGBM**, **XGBoost** — Gradient boosting regressors
- **uv** — Dependency and environment management

## Getting Started

```bash
git clone https://github.com/baechul/craft-poc-model.git
cd craft-poc-model
uv sync
uv run jupyter lab
```

> `uv sync` installs all dependencies from the locked `uv.lock` file, ensuring a reproducible environment.

## App Screenshots

<img src="./images/screen1.png" width="600" height="auto">

<img src="./images/screen2.png" width="600" height="auto">

<img src="./images/screen3.png" width="600" height="auto">

## Related

- Demo application: [craft-poc](https://github.com/baechul/craft-poc)
