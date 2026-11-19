# CRPS Random Forest

A Python package providing `CRPSRandomForestRegressor`, a Numba-accelerated
random forest that selects tree splits using a **CRPS-based criterion** and
exposes empirical predictive distributions.


## Requirements

- Python 3.10 or newer
- NumPy, Numba, joblib, scikit-learn (installed automatically by pip)

## Install from GitHub


```bash
python -m pip install "crps-random-forest @ git+https://github.com/rrross/crps-random-forest.git"
```


## Local development

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/basic_usage.py
```

The first model fit may take longer because Numba compiles the splitting
kernels on first use; later fits can reuse compiled kernels.

## Quick start

```python
import numpy as np
from crps_random_forest import CRPSRandomForestRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(60, 4))
y = 3 * X[:, 0] - X[:, 1] + rng.normal(size=60)

model = CRPSRandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    bootstrap=True,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
model.fit(X[:50], y[:50])

X_test = X[50:]
point_predictions = model.predict(X_test)              # shape (10,)
quantiles = model.predict_quantiles(X_test, (0.1, 0.5, 0.9))  # (10, 3)
members = model.predict_members(X_test)                  # list of 10 arrays
draws = model.sample(X_test, n_samples=20, random_state=1)  # (10, 20)
importances = model.feature_importances_                 # shape (4,)
```

## API

| Method or attribute | Meaning |
| --- | --- |
| `fit(X, y)` | Fit the forest; `X` is `(n_samples, n_features)` and `y` contains numerical targets. |
| `predict(X)` | Mean of pooled terminal-leaf targets for each observation. |
| `predict_members(X)` | List of arrays containing pooled leaf targets for each observation. |
| `predict_quantiles(X, quantiles)` | Empirical quantiles, returning `(n_samples, n_quantiles)`. |
| `sample(X, n_samples, random_state)` | Samples with replacement from the pooled target values. |
| `feature_importances_` | Fraction of **split nodes** selecting each feature, not gain-based importance. |
| `score(X, y)` | scikit-learn regression score (R-squared). |

### Parameters

- `n_estimators=100`: number of trees.
- `min_samples_leaf=5`: smallest allowed leaf size.
- `bootstrap=True`: sample rows with replacement for each tree.
- `max_samples=None`: bootstrap size; `None` means the entire training size,
  an integer specifies a count, and a float in `(0, 1]` specifies a fraction.
- `random_state=None`: random seed (set this for reproducibility).
- `max_depth=None`: unlimited depth, or a positive integer.
- `n_jobs=-1`: use all available joblib workers (`1` for serial execution).
- `max_features="sqrt"`: sample `floor(sqrt(n_features))` candidate predictors
  at each node; set to `None` to consider all features.

### Predictive distribution convention

`predict_members` concatenates the observed target values in **every tree's
terminal leaf** for each prediction row. Larger leaves therefore contribute
more values to the pooled empirical distribution. It does **not** weight every
tree equally, and it is not a kernel-smoothed or parametric distribution.
Consequently, `predict(X)` is the mean of this pooled distribution (which
may differ from the mean of per-tree leaf means).

This release requires finite numerical `X` and `y` (no NaN or infinity).
For a single observation, you may pass a 1D feature vector. All prediction
inputs must have the same feature count as training data.

