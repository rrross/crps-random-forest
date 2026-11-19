"""CRPS-splitting random forest with empirical predictive distributions.

The split criterion and bootstrap RNG draw order are preserved from the
original implementation supplied by the package authors.
"""

import numpy as np

from numba import njit

@njit(inline="always", cache=True, nogil=True)
def _fen_add(bit, i, delta):
    n = bit.shape[0] - 1
    while i <= n:
        bit[i] += delta
        i += i & (-i)

@njit(inline="always", cache=True, nogil=True)
def _fen_sum(bit, i):
    out = 0.0
    while i > 0:
        out += bit[i]
        i -= i & (-i)
    return out

@njit(inline="always", cache=True, nogil=True)
def _dist_to_set(bit_count, bit_sum, total_count, total_sum, idx, val):
    count_le = _fen_sum(bit_count, idx)
    sum_le = _fen_sum(bit_sum, idx)
    count_gt = total_count - count_le
    sum_gt = total_sum - sum_le
    return val * count_le - sum_le + sum_gt - val * count_gt


@njit(cache=True, nogil=True)
def _scan_feature(x_sorted, y_sorted, yi_sorted, m, n,
                  parent_T, total_sum, min_leaf):
    """One feature scan. Returns (best_score, best_threshold, found)."""
    lc = np.zeros(m + 1)
    ls = np.zeros(m + 1)
    rc = np.zeros(m + 1)
    rs = np.zeros(m + 1)

    for k in range(n):
        _fen_add(rc, yi_sorted[k], 1.0)
        _fen_add(rs, yi_sorted[k], y_sorted[k])

    n_left = 0
    n_right = n
    total_left = 0.0
    total_right = total_sum
    T_left = 0.0
    T_right = parent_T

    best_score = np.inf
    best_threshold = np.nan
    found = False

    pos = 0
    while pos < n:
        x_val = x_sorted[pos]
        next_pos = pos
        while next_pos < n and x_sorted[next_pos] == x_val:
            val = y_sorted[next_pos]
            idx = yi_sorted[next_pos]

            T_right -= _dist_to_set(rc, rs, n_right, total_right, idx, val)
            _fen_add(rc, idx, -1.0)
            _fen_add(rs, idx, -val)
            n_right -= 1
            total_right -= val

            T_left += _dist_to_set(lc, ls, n_left, total_left, idx, val)
            _fen_add(lc, idx, 1.0)
            _fen_add(ls, idx, val)
            n_left += 1
            total_left += val

            next_pos += 1

        if next_pos < n and n_left >= min_leaf and n_right >= min_leaf:
            score = T_left / n_left + T_right / n_right
            if score < best_score:
                next_x = x_sorted[next_pos]
                best_threshold = x_val + 0.5 * (next_x - x_val)
                best_score = score
                found = True

        pos = next_pos

    return best_score, best_threshold, found


@njit(cache=True, nogil=True)
def _best_split_kernel(X, y, yi, feature_ids, parent_T, total_sum, m, min_leaf):
    """
    Loop over candidate features (already chosen outside).
    X: (n, p) full node matrix; we index columns by feature_ids.
    yi: compressed 1-based y ranks for this node.
    Returns (best_feature, best_threshold, best_score, found).
    """
    n = X.shape[0]
    best_score = np.inf
    best_feature = -1
    best_threshold = np.nan
    found = False

    for f in range(feature_ids.shape[0]):
        feature = feature_ids[f]
        col = X[:, feature]
        order = np.argsort(col, kind="mergesort")
        x_sorted = col[order]
        if x_sorted[0] == x_sorted[-1]:
            continue
        y_sorted = y[order]
        yi_sorted = yi[order]

        sc, thr, fnd = _scan_feature(
            x_sorted, y_sorted, yi_sorted, m, n,
            parent_T, total_sum, min_leaf
        )
        if fnd and sc < best_score:
            best_score = sc
            best_feature = feature
            best_threshold = thr
            found = True

    return best_feature, best_threshold, best_score, found


@njit(cache=True, nogil=True)
def _pairwise_abs_sum(y_sorted):
    """T = sum_{i<j}|y_i - y_j|, y_sorted ascending."""
    total = 0.0
    csum = 0.0
    for i in range(y_sorted.shape[0]):
        total += y_sorted[i] * i - csum
        csum += y_sorted[i]
    return total


from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin



def _compress_y(y):
    """Return (yi_1based, m) where yi are dense ranks of y."""
    uniq, inv = np.unique(y, return_inverse=True)
    return (inv + 1).astype(np.int64), int(uniq.shape[0])


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "values")
    def __init__(self, feature=None, threshold=None, left=None, right=None, values=None):
        self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right; self.values = values
    @property
    def is_leaf(self):
        return self.values is not None
    def leaf_values(self, x):
        node = self
        while node.values is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.values


class _CRPSTree:
    def __init__(self, min_samples_leaf=5, random_state=None, max_depth=None,
                 max_features="sqrt"):
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.root_ = None
        self.n_features_in_ = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.root_ = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        n, p = X.shape
        ml = self.min_samples_leaf
        if n < 2 * ml:
            return _Node(values=y.copy())
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(values=y.copy())
        if np.all(y == y[0]):
            return _Node(values=y.copy())

        y_sorted = np.sort(y)
        parent_T = _pairwise_abs_sum(y_sorted)
        parent_score = parent_T / n
        total_sum = float(y_sorted.sum())

        yi, m = _compress_y(y)

        # Candidate predictors are sampled independently at each node, matching
        # the top-level sqrt(p) vs all-features toggle used by sklearn RF.
        if self.max_features is None:
            feature_ids = np.arange(p, dtype=np.int64)
        elif self.max_features == "sqrt":
            n_candidates = max(1, int(np.sqrt(p)))
            feature_ids = self.rng_.choice(
                p, size=n_candidates, replace=False
            ).astype(np.int64)
        else:
            raise ValueError('max_features must be "sqrt" or None.')

        feat, thr, score, found = _best_split_kernel(
            X, y, yi, feature_ids, parent_T, total_sum, m, ml
        )

        if not found:
            return _Node(values=y.copy())
        if score >= parent_score - 1e-12:
            return _Node(values=y.copy())

        left_mask = X[:, feat] <= thr
        nl = left_mask.sum()
        if nl < ml or (n - nl) < ml:
            return _Node(values=y.copy())

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=int(feat), threshold=float(thr), left=left, right=right)

    def predict_members_one(self, x):
        return self.root_.leaf_values(x)


def _fit_one_tree(X, y, idx, min_samples_leaf, max_depth, max_features, seed):
    tree = _CRPSTree(min_samples_leaf=min_samples_leaf,
                     random_state=seed, max_depth=max_depth,
                     max_features=max_features)
    tree.fit(X[idx], y[idx])
    return tree


class CRPSRandomForestRegressor(RegressorMixin, BaseEstimator):
    """Random forest trained using a CRPS-based split criterion.

    Each tree retains observed response values in its terminal leaves.
    ``predict_members`` pools these values across trees (so larger leaves
    contribute more members); ``predict`` averages the pooled members and
    ``predict_quantiles`` calculates their empirical quantiles.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    min_samples_leaf : int, default=5
        Minimum samples in either child after a split.
    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample.
    max_samples : int, float, or None, default=None
        Per-tree bootstrap sample size (only valid when bootstrap=True).
        Integers specify counts, floats specify a fraction of training rows.
    random_state : int, numpy.random.Generator, or None, default=None
        Seed or generator used for bootstrap sampling and tree seeds.
    max_depth : int or None, default=None
        Maximum depth for an individual tree.
    n_jobs : int or None, default=-1
        Number of parallel jobs (joblib threading backend).
    max_features : {"sqrt", None}, default="sqrt"
        Number of candidate features at each node. None tries all features.

    Attributes
    ----------
    trees_ : list
        Fitted CRPS trees.
    feature_importances_ : ndarray, shape (n_features,)
        Fraction of internal-node splits selecting each feature, not impurity
        reduction or permutation importance.
    n_features_in_ : int
        Number of training features.
    n_samples_bootstrap_ : int
        Number of samples drawn per tree when bootstrapping (or training rows).
    """

    def __init__(self, n_estimators=100, min_samples_leaf=5, bootstrap=True,
                 max_samples=None, random_state=None, max_depth=None, n_jobs=-1,
                 max_features="sqrt"):
        # Keep constructor parameters unchanged for sklearn.clone compatibility.
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.max_features = max_features

    def fit(self, X, y):
        """Fit the forest to numeric features ``X`` and continuous targets ``y``."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths.")
        if not isinstance(self.n_estimators, (int, np.integer)) or isinstance(self.n_estimators, (bool, np.bool_)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or isinstance(self.min_samples_leaf, (bool, np.bool_)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be True or False.")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, (int, np.integer))
            or isinstance(self.max_depth, (bool, np.bool_))
            or self.max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer.")
        if self.max_features not in ("sqrt", None):
            raise ValueError('max_features must be "sqrt" or None.')
        if self.n_jobs is not None and (
            not isinstance(self.n_jobs, (int, np.integer))
            or isinstance(self.n_jobs, (bool, np.bool_))
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be None or a nonzero integer.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("This implementation requires finite numeric X and y.")

        self.n_features_in_ = X.shape[1]

        if self.bootstrap:
            n_boot = self._resolve_max_samples(X.shape[0])
        else:
            if self.max_samples is not None:
                raise ValueError("max_samples cannot be set if bootstrap=False.")
            n_boot = X.shape[0]
        self.n_samples_bootstrap_ = n_boot

        rng = np.random.default_rng(self.random_state)
        # Match the original's RNG consumption order EXACTLY: for each tree,
        # draw its seed first, then its bootstrap indices. This keeps results
        # reproducible against the reference serial implementation.
        seeds = []
        idxs = []
        base = np.arange(X.shape[0])
        for _ in range(self.n_estimators):
            seeds.append(int(rng.integers(0, np.iinfo(np.int32).max)))
            if self.bootstrap:
                idxs.append(rng.integers(0, X.shape[0], size=n_boot))
            else:
                idxs.append(base)

        self.trees_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_one_tree)(X, y, idxs[i], self.min_samples_leaf,
                                   self.max_depth, self.max_features, seeds[i])
            for i in range(self.n_estimators)
        )

        # Cheap model-native importance: fraction of all internal-node splits
        # assigned to each predictor across the fitted forest.
        split_counts = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.trees_:
            stack = [tree.root_]
            while stack:
                node = stack.pop()
                if node is None or node.is_leaf:
                    continue
                split_counts[node.feature] += 1.0
                stack.append(node.left)
                stack.append(node.right)
        total_splits = split_counts.sum()
        self.feature_importances_ = (
            split_counts / total_splits if total_splits > 0 else split_counts
        )
        return self

    def _resolve_max_samples(self, n_samples):
        ms = self.max_samples
        if ms is None:
            return n_samples
        if isinstance(ms, (bool, np.bool_)):
            raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")
        if isinstance(ms, (int, np.integer)):
            ms = int(ms)
            if not 1 <= ms <= n_samples:
                raise ValueError("max_samples int must be in [1, n_samples].")
            return ms
        if isinstance(ms, (float, np.floating)):
            ms = float(ms)
            if not 0.0 < ms <= 1.0:
                raise ValueError("max_samples float must be in (0.0, 1.0].")
            return max(1, int(round(n_samples * ms)))
        raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")

    def predict_members(self, X):
        """Return pooled terminal-leaf targets (one array per observation)."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 1- or 2-dimensional.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values.")
        out = []
        for x in X:
            members = [t.predict_members_one(x) for t in self.trees_]
            out.append(np.concatenate(members))
        return out

    def predict(self, X):
        """Return means of pooled leaf members (shape: n_samples)."""
        return np.array([v.mean() for v in self.predict_members(X)], dtype=float)

    def predict_quantiles(self, X, quantiles=(0.1, 0.5, 0.9)):
        """Return predictive quantiles (shape: n_samples, n_quantiles)."""
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("quantiles must be a nonempty 1D sequence in [0, 1].")
        return np.vstack([np.quantile(v, q) for v in self.predict_members(X)])

    def sample(self, X, n_samples=1, random_state=None):
        """Sample with replacement from the pooled empirical distribution."""
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, (bool, np.bool_)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(random_state)
        draws = []
        for v in self.predict_members(X):
            draws.append(v[rng.integers(0, len(v), size=n_samples)])
        return np.vstack(draws)

    def _check_is_fitted(self):
        if not hasattr(self, "trees_"):
            raise RuntimeError("Estimator is not fitted yet. Call fit(X, y) first.")
