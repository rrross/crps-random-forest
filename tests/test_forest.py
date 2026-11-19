"""Behavioral and API tests for the CRPS random forest."""

import numpy as np
import pytest
from sklearn.base import clone, is_regressor

from crps_random_forest import CRPSRandomForestRegressor


@pytest.fixture
def example_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = 2.0 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=40)
    return X, y


def test_fit_predict_and_distribution_shapes(example_data):
    X, y = example_data
    model = CRPSRandomForestRegressor(
        n_estimators=5, min_samples_leaf=2, random_state=42, n_jobs=1
    ).fit(X, y)

    assert model.n_features_in_ == X.shape[1]
    assert len(model.trees_) == 5
    assert model.n_samples_bootstrap_ == len(X)
    assert model.predict(X[:5]).shape == (5,)
    assert model.predict_quantiles(X[:5]).shape == (5, 3)
    assert model.sample(X[:5], n_samples=10).shape == (5, 10)
    members = model.predict_members(X[:5])
    assert len(members) == 5
    assert all(len(v) >= 5 for v in members)
    np.testing.assert_allclose(model.predict(X[:5]), [v.mean() for v in members])
    assert np.isfinite(model.predict(X[:5])).all()
    assert model.feature_importances_.shape == (X.shape[1],)
    assert np.all(model.feature_importances_ >= 0)
    assert np.isclose(model.feature_importances_.sum(), 1.0) or np.all(
        model.feature_importances_ == 0
    )


def test_separates_simple_step_function():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 10, 10, 10], dtype=float)
    model = CRPSRandomForestRegressor(
        n_estimators=1,
        min_samples_leaf=1,
        bootstrap=False,
        max_features=None,
        random_state=17,
        n_jobs=1,
    ).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y)
    np.testing.assert_allclose(model.predict_quantiles([[0], [5]], [0.1, 0.9]), [[0, 0], [10, 10]])
    np.testing.assert_array_equal(model.feature_importances_, [1.0])


def test_constant_targets_have_no_splits(example_data):
    X, _ = example_data
    y = np.full(len(X), 7.0)
    model = CRPSRandomForestRegressor(n_estimators=3, random_state=1, n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(model.predict(X[:2]), [7.0, 7.0])
    np.testing.assert_array_equal(model.feature_importances_, np.zeros(X.shape[1]))


def test_reproducible_across_parallel_jobs(example_data):
    X, y = example_data
    opts = dict(n_estimators=6, min_samples_leaf=2, random_state=123)
    serial = CRPSRandomForestRegressor(**opts, n_jobs=1).fit(X, y)
    parallel = CRPSRandomForestRegressor(**opts, n_jobs=2).fit(X, y)
    np.testing.assert_array_equal(serial.predict(X), parallel.predict(X))
    np.testing.assert_array_equal(serial.feature_importances_, parallel.feature_importances_)
    for a, b in zip(serial.predict_members(X[:4]), parallel.predict_members(X[:4])):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        serial.sample(X[:4], n_samples=7, random_state=5),
        serial.sample(X[:4], n_samples=7, random_state=5),
    )


def test_sklearn_compatible_constructor_and_clone(example_data):
    X, y = example_data
    model = CRPSRandomForestRegressor(
        n_estimators=4, max_features=None, min_samples_leaf=2, n_jobs=1
    )
    assert is_regressor(model)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned.fit(X, y).score(X, y) <= 1.0 + 1e-12


def test_max_samples_options(example_data):
    X, y = example_data
    for max_samples, expected in ((None, 40), (7, 7), (0.25, 10)):
        model = CRPSRandomForestRegressor(
            n_estimators=2, max_samples=max_samples, random_state=1, n_jobs=1
        ).fit(X, y)
        assert model.n_samples_bootstrap_ == expected
    no_bootstrap = CRPSRandomForestRegressor(
        n_estimators=2, bootstrap=False, random_state=1, n_jobs=1
    ).fit(X, y)
    assert no_bootstrap.n_samples_bootstrap_ == len(X)


@pytest.mark.parametrize(
    "params",
    [
        {"n_estimators": 0},
        {"n_estimators": 2.5},
        {"min_samples_leaf": 0},
        {"min_samples_leaf": 2.5},
        {"max_depth": 0},
        {"max_features": "invalid"},
        {"n_jobs": 0},
        {"bootstrap": 1},
        {"bootstrap": False, "max_samples": 2},
        {"max_samples": 0.0},
        {"max_samples": 41},
        {"max_samples": True},
    ],
)
def test_invalid_hyperparameters(example_data, params):
    X, y = example_data
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor(**params).fit(X, y)


def test_invalid_training_and_prediction_inputs(example_data):
    X, y = example_data
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor().fit(X[0], y)
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor().fit(X[:0], y[:0])
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor().fit(np.empty((40, 0)), y)
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor().fit(X, y[:2])
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        CRPSRandomForestRegressor().fit(X_bad, y)
    model = CRPSRandomForestRegressor(n_estimators=2, random_state=5, n_jobs=1).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])
    with pytest.raises(ValueError):
        model.predict(np.empty((0, 3)))
    with pytest.raises(ValueError):
        model.predict(np.full((1, 3), np.inf))
    with pytest.raises(ValueError):
        model.predict(np.ones((2, 2, 3)))
    assert model.predict(X[0]).shape == (1,)


def test_invalid_quantiles_and_sampling(example_data):
    X, y = example_data
    model = CRPSRandomForestRegressor(n_estimators=2, random_state=5, n_jobs=1).fit(X, y)
    for quantiles in ([], [1.1], [-0.2], [np.nan], [[0.5]]):
        with pytest.raises(ValueError):
            model.predict_quantiles(X[:2], quantiles)
    for size in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.sample(X[:2], n_samples=size)


def test_predict_before_fit(example_data):
    X, _ = example_data
    model = CRPSRandomForestRegressor()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X[:1])
