"""
Single frozen structural model M: a semiparametric ("extended rank
likelihood") Gaussian copula over the generation-panel coordinates.

Fitting (fixed algorithm, no tuning):
  1. Each coordinate's marginal is represented, with no distributional
     assumption, by its empirical training CDF (numeric columns) or by a
     fixed alphabetical category ordering plus empirical category
     frequencies (categorical columns). This is a nonparametric *fitted*
     marginal (ordinary fitted parameters), not a tuned hyperparameter.
  2. Every observed raw coordinate is mapped, via its own fitted marginal,
     to a standard-normal latent score z. Missing coordinates are latent
     and unobserved.
  3. The p x p latent Gaussian correlation matrix Sigma is estimated by a
     plug-in EM algorithm for the multivariate-normal missing-data problem
     (Little & Rubin): E-step imputes each row's missing latent
     coordinates by their conditional Gaussian mean/covariance given that
     row's observed latent coordinates under the current Sigma; M-step
     recomputes Sigma from the completed second-moment matrix and
     re-normalizes to unit diagonal. A fixed number of iterations
     (config.EM_ITERATIONS) is run; a small fixed ridge
     (config.COVARIANCE_RIDGE) is added to every inverted sub-block for
     numerical invertibility. Neither is chosen by comparing candidate
     values.

Once Sigma is frozen this one object supplies BOTH:
  - geometry: d_G(a,b), a whitened Mahalanobis distance between each
    material's fully-completed latent vector (missing coordinates filled
    in by the model's own conditional-mean imputation -- the same
    machinery used for generation);
  - generation: unconditional sampling from N(0, Sigma) followed by the
    per-coordinate inverse marginal transform, and conditional sampling
    of any hidden subset given any observed subset via the standard
    Gaussian conditional distribution.

No quantization/discretization is used anywhere in this model.
"""
import json
import os

import numpy as np
from scipy.stats import norm

from . import config


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return False


class _MarginalNumeric:
    """Fitted (nonparametric) empirical-CDF marginal for one numeric column."""

    def __init__(self, values):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        v = np.sort(v)
        self.sorted_train = v
        self.n = len(v)
        # rank positions used for BOTH forward (u from value) and inverse
        # (value from u) directions, consistent van-der-Waerden convention.
        self.rank_u = (np.arange(self.n) + 0.5) / self.n

    def to_u(self, values):
        v = np.asarray(values, dtype=float)
        lo = np.searchsorted(self.sorted_train, v, side="left")
        hi = np.searchsorted(self.sorted_train, v, side="right")
        u = (lo + hi) / (2.0 * self.n)
        eps = 0.5 / self.n
        return np.clip(u, eps, 1 - eps)

    def from_u(self, u):
        u = np.clip(np.asarray(u, dtype=float), self.rank_u[0], self.rank_u[-1])
        return np.interp(u, self.rank_u, self.sorted_train)


class _MarginalCategorical:
    """Fitted empirical-frequency marginal for one categorical column, using
    a fixed (alphabetical) category ordering chosen before fitting."""

    def __init__(self, values):
        vals = [v for v in values if not _is_missing(v)]
        cats = sorted(set(vals))  # fixed, deterministic ordering
        counts = np.array([vals.count(c) for c in cats], dtype=float)
        probs = counts / counts.sum()
        cum = np.concatenate([[0.0], np.cumsum(probs)])
        self.categories = cats
        self.cum_edges = cum  # length K+1, cum_edges[k]..cum_edges[k+1] -> category k
        # representative u for each category = midpoint of its cumulative interval
        self.rep_u = (cum[:-1] + cum[1:]) / 2.0
        self.index = {c: i for i, c in enumerate(cats)}

    def to_u(self, values):
        u = np.full(len(values), np.nan)
        for i, v in enumerate(values):
            if _is_missing(v):
                continue
            idx = self.index.get(v)
            if idx is None:
                # unseen category at eval time: fall back to nearest by
                # insertion position in the fixed alphabetical ordering.
                idx = np.searchsorted(self.categories, v)
                idx = min(max(idx, 0), len(self.categories) - 1)
            u[i] = self.rep_u[idx]
        return u

    def from_u(self, u):
        u = np.clip(np.asarray(u, dtype=float), 1e-9, 1 - 1e-9)
        k = np.clip(np.searchsorted(self.cum_edges, u, side="right") - 1, 0, len(self.categories) - 1)
        return np.array([self.categories[i] for i in k], dtype=object)


def _group_by_pattern(obs_mask):
    """obs_mask: (n,p) bool. Returns dict: pattern_bytes -> row indices array,
    and dict: pattern_bytes -> tuple of observed column indices."""
    groups = {}
    patt_cols = {}
    packed = np.packbits(obs_mask, axis=1)
    keys = [row.tobytes() for row in packed]
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    for k, idxs in groups.items():
        patt_cols[k] = tuple(np.flatnonzero(obs_mask[idxs[0]]))
    return {k: np.array(v) for k, v in groups.items()}, patt_cols


class GaussianCopulaModel:
    def __init__(self, numeric_cols, categorical_cols,
                 em_iterations=config.EM_ITERATIONS, ridge=config.COVARIANCE_RIDGE):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.columns = self.numeric_cols + self.categorical_cols
        self.p = len(self.columns)
        self.em_iterations = em_iterations
        self.ridge = ridge
        self.marginals = {}
        self.Sigma = None
        self.Sigma_inv_full = None
        self._cond_cache = {}

    # ---------------- marginal fitting + latent transform ----------------

    def fit_marginals(self, df_train):
        for c in self.numeric_cols:
            self.marginals[c] = _MarginalNumeric(df_train[c].to_numpy())
        for c in self.categorical_cols:
            self.marginals[c] = _MarginalCategorical(df_train[c].tolist())

    def to_latent(self, df):
        """df: dataframe with self.columns. Returns Z (n,p) with np.nan for missing."""
        n = len(df)
        Z = np.full((n, self.p), np.nan)
        for j, c in enumerate(self.columns):
            m = self.marginals[c]
            if c in self.categorical_cols:
                vals = df[c].tolist()
            else:
                vals = df[c].to_numpy()
            u = m.to_u(vals)
            u = np.asarray(u, dtype=float)
            with np.errstate(invalid="ignore"):
                z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
            z[~np.isfinite(u)] = np.nan
            Z[:, j] = z
        return Z

    def from_latent(self, Z):
        """Inverse transform: Z (n,p) -> dataframe-like dict of raw columns."""
        out = {}
        u_all = norm.cdf(Z)
        for j, c in enumerate(self.columns):
            m = self.marginals[c]
            out[c] = m.from_u(u_all[:, j])
        return out

    # -------------------------- EM fitting of Sigma --------------------------

    def fit_sigma(self, Z):
        n, p = Z.shape
        obs_mask = np.isfinite(Z)
        Zf = np.where(obs_mask, Z, 0.0)
        # initialize Sigma at identity (independence) -- a fixed, non-data-peeked start.
        Sigma = np.eye(p)
        groups, patt_cols = _group_by_pattern(obs_mask)

        for _ in range(self.em_iterations):
            T = np.zeros((p, p))
            Zc = Zf.copy()
            for key, rows in groups.items():
                obs_idx = np.array(patt_cols[key], dtype=int)
                mis_idx = np.array([j for j in range(p) if j not in set(obs_idx)], dtype=int)
                Zo = Zf[np.ix_(rows, obs_idx)]  # (g, |O|)
                if len(obs_idx) == 0:
                    if len(mis_idx) > 0:
                        cond_cov = Sigma[np.ix_(mis_idx, mis_idx)]
                        cond_mean = np.zeros((len(rows), len(mis_idx)))
                    else:
                        continue
                else:
                    Soo = Sigma[np.ix_(obs_idx, obs_idx)] + self.ridge * np.eye(len(obs_idx))
                    Soo_inv = np.linalg.inv(Soo)
                    if len(mis_idx) > 0:
                        Smo = Sigma[np.ix_(mis_idx, obs_idx)]
                        coef = Smo @ Soo_inv  # (|M|,|O|)
                        cond_mean = Zo @ coef.T  # (g, |M|)
                        cond_cov = Sigma[np.ix_(mis_idx, mis_idx)] - coef @ Sigma[np.ix_(obs_idx, mis_idx)]
                    else:
                        cond_mean = None
                        cond_cov = None

                if len(mis_idx) > 0:
                    Zc[np.ix_(rows, mis_idx)] = cond_mean

                # accumulate E[Z Z^T | obs] for this group
                Zg = Zc[rows]  # (g, p) completed (mean) values for this pattern
                block = Zg.T @ Zg  # (p,p), correct for observed-observed and observed-missing (mean x mean)
                if len(mis_idx) > 0 and cond_cov is not None:
                    block[np.ix_(mis_idx, mis_idx)] += len(rows) * cond_cov
                T += block

            Sigma = T / n
            d = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
            Sigma = Sigma / d[:, None] / d[None, :]
            np.fill_diagonal(Sigma, 1.0)
            Sigma = (Sigma + Sigma.T) / 2.0

        self.Sigma = Sigma
        self.Sigma_inv_full = np.linalg.inv(Sigma + self.ridge * np.eye(p))
        return Sigma

    # -------------------------- conditional-Gaussian cache --------------------------

    def _cond_params(self, obs_idx_tuple):
        cached = self._cond_cache.get(obs_idx_tuple)
        if cached is not None:
            return cached
        p = self.p
        obs_idx = np.array(obs_idx_tuple, dtype=int)
        mis_idx = np.array([j for j in range(p) if j not in set(obs_idx)], dtype=int)
        if len(obs_idx) == 0:
            coef = np.zeros((len(mis_idx), 0))
            cond_cov = self.Sigma[np.ix_(mis_idx, mis_idx)].copy()
        else:
            Soo = self.Sigma[np.ix_(obs_idx, obs_idx)] + self.ridge * np.eye(len(obs_idx))
            Soo_inv = np.linalg.inv(Soo)
            if len(mis_idx) > 0:
                Smo = self.Sigma[np.ix_(mis_idx, obs_idx)]
                coef = Smo @ Soo_inv
                cond_cov = self.Sigma[np.ix_(mis_idx, mis_idx)] - coef @ self.Sigma[np.ix_(obs_idx, mis_idx)]
            else:
                coef = np.zeros((0, len(obs_idx)))
                cond_cov = np.zeros((0, 0))
        cond_cov = (cond_cov + cond_cov.T) / 2.0
        result = (obs_idx, mis_idx, coef, cond_cov)
        self._cond_cache[obs_idx_tuple] = result
        return result

    def complete_latent_mean(self, Z):
        """Fill every missing latent coordinate with its conditional mean
        given that row's observed coordinates (frozen Sigma). Used for
        geometry. Returns a fully-populated (n,p) array."""
        n, p = Z.shape
        obs_mask = np.isfinite(Z)
        out = np.where(obs_mask, Z, 0.0).copy()
        groups, patt_cols = _group_by_pattern(obs_mask)
        for key, rows in groups.items():
            obs_idx, mis_idx, coef, _ = self._cond_params(patt_cols[key])
            if len(mis_idx) == 0:
                continue
            Zo = out[np.ix_(rows, obs_idx)]
            out[np.ix_(rows, mis_idx)] = Zo @ coef.T
        return out

    def sample_conditional_latent(self, Z_obs_template, rng, n_draws=1):
        """Z_obs_template: (n,p) array, np.nan where hidden/unobserved.
        Returns array (n_draws, n, p) with hidden coords sampled from the
        model's conditional Gaussian and observed coords held fixed."""
        n, p = Z_obs_template.shape
        obs_mask = np.isfinite(Z_obs_template)
        base = np.where(obs_mask, Z_obs_template, 0.0)
        groups, patt_cols = _group_by_pattern(obs_mask)
        draws = np.repeat(base[None, :, :], n_draws, axis=0)
        for key, rows in groups.items():
            obs_idx, mis_idx, coef, cond_cov = self._cond_params(patt_cols[key])
            if len(mis_idx) == 0:
                continue
            Zo = base[np.ix_(rows, obs_idx)]
            mean = Zo @ coef.T  # (g,|M|)
            if cond_cov.size == 0:
                continue
            cond_cov_r = cond_cov + self.ridge * np.eye(len(mis_idx))
            L = np.linalg.cholesky(cond_cov_r)
            g = len(rows)
            for d in range(n_draws):
                eps = rng.standard_normal(size=(g, len(mis_idx)))
                samp = mean + eps @ L.T
                draws[d][np.ix_(rows, mis_idx)] = samp
        return draws

    def sample_unconditional_latent(self, n, rng):
        L = np.linalg.cholesky(self.Sigma + self.ridge * np.eye(self.p))
        eps = rng.standard_normal(size=(n, self.p))
        return eps @ L.T

    # -------------------------- geometry --------------------------

    def geometry_distance_matrix(self, Z_full):
        """Z_full: (n,p) fully completed latent matrix. Returns (n,n) symmetric
        whitened-Mahalanobis distance using the frozen precision matrix,
        normalized by p so it is on a comparable scale to d_E/d_M."""
        W = self.Sigma_inv_full
        # d^2(a,b) = (za-zb)^T W (za-zb) = za^T W za - 2 za^T W zb + zb^T W zb
        WZ = Z_full @ W
        diag = np.sum(WZ * Z_full, axis=1)
        cross = WZ @ Z_full.T
        sq = diag[:, None] + diag[None, :] - 2 * cross
        sq = np.maximum(sq, 0.0)
        d = np.sqrt(sq / self.p)
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        return d

    # -------------------------- persistence (frozen artifacts) --------------------------

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "sigma.npy"), self.Sigma)
        meta = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "em_iterations": self.em_iterations,
            "ridge": self.ridge,
        }
        marg = {}
        for c in self.numeric_cols:
            m = self.marginals[c]
            marg[c] = {"type": "numeric", "sorted_train": m.sorted_train.tolist()}
        for c in self.categorical_cols:
            m = self.marginals[c]
            marg[c] = {
                "type": "categorical",
                "categories": list(m.categories),
                "cum_edges": m.cum_edges.tolist(),
            }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(out_dir, "marginals.json"), "w") as f:
            json.dump(marg, f)

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        model = cls(meta["numeric_cols"], meta["categorical_cols"],
                     em_iterations=meta["em_iterations"], ridge=meta["ridge"])
        with open(os.path.join(out_dir, "marginals.json")) as f:
            marg = json.load(f)
        for c, d in marg.items():
            if d["type"] == "numeric":
                m = _MarginalNumeric([])
                m.sorted_train = np.array(d["sorted_train"], dtype=float)
                m.n = len(m.sorted_train)
                m.rank_u = (np.arange(m.n) + 0.5) / m.n
            else:
                m = _MarginalCategorical([])
                m.categories = d["categories"]
                m.cum_edges = np.array(d["cum_edges"], dtype=float)
                m.rep_u = (m.cum_edges[:-1] + m.cum_edges[1:]) / 2.0
                m.index = {cat: i for i, cat in enumerate(m.categories)}
            model.marginals[c] = m
        model.Sigma = np.load(os.path.join(out_dir, "sigma.npy"))
        model.Sigma_inv_full = np.linalg.inv(model.Sigma + model.ridge * np.eye(model.p))
        return model
