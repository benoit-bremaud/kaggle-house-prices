# Submission Log

Track all Kaggle submissions for this competition.

| # | Date | Score | Model | Features | Notes |
|---|------|-------|-------|----------|-------|
| 1 | 2026-03-26 | CV: 0.1363 / LB: 0.14507 | RandomForestRegressor(100) | Engineered (TotalSF, TotalBath, HouseAge, ordinal + one-hot) | Baseline — default params, log(SalePrice), 5-fold CV |
| 2 | 2026-03-27 | CV: 0.1266 / LB: 0.13138 | LGBMRegressor(500, lr=0.05) | Same as #1 | LightGBM beats RF by 0.0097 — same features, better model |
