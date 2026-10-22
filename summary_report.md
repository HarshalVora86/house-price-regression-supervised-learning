# Summary Report: House Price Prediction (Regression)

## 1. Business Problem and Dataset
A PropTech startup (similar to NoBroker or MagicBricks) wants an automated estimator so buyers and sellers can benchmark property values. I used the Ames Housing dataset (Kaggle, `train.csv`): 1,460 houses, 79 features (numeric and categorical) and a continuous target, `SalePrice` (USD). The task is supervised regression.

## 2. Preprocessing and Feature Engineering
- **Target:** `SalePrice` is right-skewed, so I applied `log1p` and converted predictions back with `expm1`. All metrics are reported in USD.
- **Missing values:** columns with more than 80% missing were dropped. Categorical NaNs were filled with `'None'` (the feature is absent, e.g. no pool or garage) and numeric NaNs were filled with a numeric value, leaving no missing data.
- **Outliers:** two rows with `GrLivArea > 4000` and `SalePrice < 300,000` were removed because they distort a linear fit.
- **New features:** `TotalSF`, `HouseAge`, `RemodAge`, `HasGarage`, `HasPool`, which capture size, age and amenities more directly than the raw columns.
- **Encoding and scaling:** ordinal encoding for quality columns, one-hot encoding for nominal columns with 10 or fewer categories, label encoding for high-cardinality columns, `log1p` for numeric features with skew above 0.75, and `StandardScaler`. Everything is chained in a `ColumnTransformer` (190 features after processing).
- **Split:** 80/20 train-test, `random_state=42`.

## 3. Best Model
| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| **XGBoost** | **19,150** | **13,807** | **0.934** |
| Lasso | 20,073 | 14,720 | 0.927 |
| Ridge | 20,489 | 14,647 | 0.924 |
| Linear Regression | 22,620 | 15,774 | 0.907 |
| Random Forest | 23,248 | 16,309 | 0.902 |

XGBoost performed best. It builds trees one after another, each correcting the previous errors, so it captures non-linear effects and feature interactions that linear models miss. Five-fold CV RMSE was 0.1232 ± 0.0077 (log scale) against a test RMSE of 0.1240, so the result is stable and not a lucky split. Residual analysis showed that errors are largest for expensive houses. RandomizedSearchCV did not beat the default settings (tuned RMSE $21,079), probably because the search space allowed very deep trees that overfit a small dataset.

## 4. Top 3 Predictive Features
1. **OverallQual:** build and finish quality is the strongest driver. Renovations that raise perceived quality are likely to return the most value.
2. **TotalSF:** total area (basement plus both floors) matters more than any single room count, so price per square foot is a sound benchmark.
3. **ExterQual:** exterior material quality shapes first impressions. Curb appeal and facade upkeep are worth investing in before listing.

Location (neighbourhood), garage capacity and central air also contribute.

## 5. Next Steps for Production
- **More data:** 1,460 rows from one city is small. Adding recent, local (Indian market) listings would improve reliability.
- **Better tuning:** use Bayesian optimisation (e.g. Optuna) with a tighter search space instead of random search.
- **Ensembling:** blend XGBoost with Lasso/Ridge (stacking) to reduce variance.
- **Neighbourhood features:** add target encoding or geographic data (distance to metro, schools).
- **Monitoring:** track RMSE on live predictions and retrain regularly as prices drift.
