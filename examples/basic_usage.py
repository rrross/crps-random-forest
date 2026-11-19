"""Run this after installing the package: python examples/basic_usage.py"""

import numpy as np

from crps_random_forest import CRPSRandomForestRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 3))
y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(scale=0.4, size=100)

model = CRPSRandomForestRegressor(
    n_estimators=50,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X[:80], y[:80])

X_new = X[80:85]
print("Point predictions:", model.predict(X_new))
print("10%, 50%, 90% quantiles:\n", model.predict_quantiles(X_new))
print("5 distribution samples per row:\n", model.sample(X_new, n_samples=5, random_state=0))
print("Split-frequency feature importance:", model.feature_importances_)
