# NASDAQ Volatility Forecasting: XGBoost with PDP, ALE & SHAP

Forecasts 30-day realized volatility of the NASDAQ Composite Index using XGBoost, with full global interpretability via Partial Dependence Plots, Accumulated Local Effects, and SHAP values.

## Motivation
Volatility drives portfolio risk, leverage constraints, margin calls, and derivative pricing. A model that is accurate but opaque is difficult to trust or justify — global interpretation tools provide the transparency required for real financial decisions.

## Data
- **Source:** Yahoo Finance via `quantmod` — 10 years of daily NASDAQ data (2014–2024), downloaded live
- **Target:** 30-day rolling realized volatility (rolling SD of daily returns)
- **Raw features:** open, high, low, close, adjusted close, volume

## Feature Engineering
| Feature | Description |
|---|---|
| `daily_return` | Adjusted close return |
| `abs_return` | Absolute return |
| `sq_return` | Squared return (proxy for realized variance) |
| `ma7`, `ma30` | 7- and 30-day moving averages |
| `volume_change` | Day-over-day volume ratio |
| `lag_vol_1/5/10` | 1-, 5-, 10-day lagged volatility — captures **volatility clustering** |

Lagged volatility features were the key addition: volatility is path-dependent, and these lags allow the model to recognize persistence rather than reacting to noise.

## Modelling
- Chronological 80/20 train/test split (no data leakage)
- XGBoost with conservative settings: `eta = 0.05`, `max_depth = 4`, `subsample = 0.8`, early stopping
- Conservative tuning chosen deliberately — volatility is noisy and prone to overfitting

## Interpretability (via DALEX)
| Method | What it shows |
|---|---|
| **Permutation Importance** | Which features matter most globally (Δ RMSE) |
| **PDP** | Average marginal effect of each feature on predicted volatility |
| **ALE** | Unbiased version of PDP — accounts for feature correlations |
| **SHAP** | Additive feature contributions per observation; aggregated as mean \|SHAP\| for global view |

## Tech Stack
- R
- `quantmod` · `xgboost` · `DALEX`
- `zoo` · `dplyr` · `ggplot2` · `lubridate`
