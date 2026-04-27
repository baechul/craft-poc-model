# craft-poc-model

A set of Jupyter notebooks for sales-model experimentation, from raw-data cleaning to feature engineering, model training, evaluation, and artifact export.

## Overview

This repository is the model-development side of [craft-poc](https://github.com/baechul/craft-poc). It currently includes:

- Data cleaning and preprocessing from raw CSV input
- Two forecasting model tracks:
  - Category-level revenue prediction (`total_revenue`)
  - Product-level demand prediction (`units_sold`)
- Gradient-boosting model comparison (LightGBM vs XGBoost)
- Saved model artifacts for downstream inference

## Notebooks

| Notebook                                                                     | Description                                                             |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [notebooks/rawdata-cleaning.ipynb](notebooks/rawdata-cleaning.ipynb)         | Cleans and validates source sales data, then outputs `sales.csv`.       |
| [notebooks/category-sales-model.ipynb](notebooks/category-sales-model.ipynb) | Builds category-level time-series features and trains revenue models.   |
| [notebooks/product-sales-model.ipynb](notebooks/product-sales-model.ipynb)   | Builds product-level time-series features and trains unit-sales models. |

## Modeling Approach

Both model notebooks follow the same structure:

1. Aggregate transactional records to a daily forecasting grain.
2. Encode categorical identifiers using `LabelEncoder`.
3. Build temporal features (`year`, `month`, `week`, `day_of_week`).
4. Build lag features (1, 7, 14, 30 days).
5. Split chronologically (`80%` train, `20%` test).
6. Train and compare `LGBMRegressor` and `XGBRegressor`.
7. Evaluate and print metrics.

## Evaluation Metrics

The notebooks report:

- `MAE` (Mean Absolute Error)
- `RMSE` (Root Mean Squared Error)
- `R2` (coefficient of determination)

Notes:

- MAE and RMSE are in target units (`total_revenue` or `units_sold`).
- R2 is unitless and indicates explained variance relative to a mean baseline.

## Features by Model

### Category Model (`notebooks/category-sales-model.ipynb`)

Target: `total_revenue` per `(date, product_category)`

Core features:

- `category_encoded`
- `year`, `month`, `week`, `day_of_week`
- `revenue_lag_1`, `revenue_lag_7`, `revenue_lag_14`, `revenue_lag_30`

### Product Model (`notebooks/product-sales-model.ipynb`)

Target: `units_sold` per `(date, product_category, product_name)`

Core features:

- `category_encoded`, `product_encoded`
- `year`, `month`, `week`, `day_of_week`
- `units_lag_1`, `units_lag_7`, `units_lag_14`, `units_lag_30`

## Saved Artifacts

Current notebook outputs include:

- `notebooks/models/top_sales_prediction_lgb.joblib`
- `notebooks/models/top_sales_prediction_xgb.joblib`
- `notebooks/models/top_products_prediction_lgb.joblib`

Each artifact stores the trained model and required encoders/feature metadata used during training.

## Tech Stack

- Python 3.14
- pandas, numpy
- scikit-learn (LabelEncoder + metrics)
- LightGBM, XGBoost
- joblib
- uv

## Getting Started

```bash
git clone https://github.com/baechul/craft-poc-model.git
cd craft-poc-model
uv sync
uv run jupyter lab
```

`uv sync` installs dependencies from the lockfile for reproducible environments.

## App Screenshots

<img src="./images/screen1.png" width="600" height="auto">

<img src="./images/screen2.png" width="600" height="auto">

<img src="./images/screen3.png" width="600" height="auto">

## Related

- Demo application: [craft-poc](https://github.com/baechul/craft-poc)
