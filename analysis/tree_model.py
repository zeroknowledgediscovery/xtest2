"""The single frozen structural model: a Chow-Liu tree-structured categorical
Bayesian network over the discretized material-state coordinates.

From the same fitted object (tree structure + conditional probability
tables) this module provides:
  - exact log-likelihood (used only for the permitted quantization-resolution
    selection on the inner development split),
  - a sampler that draws X ~ P_M(X | X_O = x_O) for ANY evidence set O,
    including O = empty (unconditional generation) and O = arbitrary partial
    observation (conditional completion), via exact tree belief propagation,
  - a Fisher-score embedding of each material, from which the submitted
    geometry d_G is computed (Euclidean distance between normalized Fisher
    scores). Geometry and generation both read the same `edges`, `parent`,
    `root_marginal`, and `cpt` fitted arrays: no independently-fitted side
    model is used for either capability.
"""
from dataclasses import dataclass, field

import numpy as np

import config


def _pairwise_mi(discrete: np.ndarray, n_states: np.ndarray, alpha: float) -> np.ndarray:
    """discrete: (N, D) int array of state codes. n_states: (D,) array.
    Returns (D, D) symmetric mutual-information matrix (Laplace-smoothed)."""
    n_rows, n_cols = discrete.shape
    mi = np.zeros((n_cols, n_cols), dtype=float)
    marginals = []
    for j in range(n_cols):
        counts = np.bincount(discrete[:, j], minlength=n_states[j]).astype(float)
        marginals.append((counts + alpha) / (n_rows + alpha * n_states[j]))
    for i in range(n_cols):
        ni = n_states[i]
        xi = discrete[:, i]
        for j in range(i + 1, n_cols):
            nj = n_states[j]
            xj = discrete[:, j]
            joint_idx = xi.astype(np.int64) * nj + xj.astype(np.int64)
            joint_counts = np.bincount(joint_idx, minlength=ni * nj).astype(float).reshape(ni, nj)
            pij = (joint_counts + alpha) / (n_rows + alpha * ni * nj)
            pi = marginals[i].reshape(-1, 1)
            pj = marginals[j].reshape(1, -1)
            ratio = pij / (pi * pj)
            term = pij * np.log(ratio)
            val = float(np.sum(term))
            mi[i, j] = val
            mi[j, i] = val
    return mi


def _max_spanning_tree_edges(mi: np.ndarray):
    """Kruskal maximum-spanning-tree over the complete MI graph. Returns list
    of (i, j, weight) tree edges."""
    n = mi.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((mi[i, j], i, j))
    edges.sort(key=lambda e: e[0], reverse=True)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree_edges = []
    for w, i, j in edges:
        if union(i, j):
            tree_edges.append((i, j, w))
        if len(tree_edges) == n - 1:
            break
    return tree_edges


def _root_and_orient(tree_edges, n, mi):
    adj = {i: [] for i in range(n)}
    for i, j, w in tree_edges:
        adj[i].append((j, w))
        adj[j].append((i, w))
    strength = np.zeros(n)
    for i, j, w in tree_edges:
        strength[i] += w
        strength[j] += w
    root = int(np.argmax(strength))

    parent = {root: None}
    order = [root]
    visited = {root}
    queue = [root]
    children = {i: [] for i in range(n)}
    while queue:
        u = queue.pop(0)
        for v, _w in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                order.append(v)
                queue.append(v)
    return root, parent, children, order


@dataclass
class ChowLiuTreeModel:
    columns: list
    n_states: dict
    alpha: float = config.LAPLACE_ALPHA
    fisher_eps: float = config.FISHER_VAR_EPS

    root: int = None
    parent: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    root_marginal: np.ndarray = None
    cpt: dict = field(default_factory=dict)   # node -> (n_states[parent], n_states[node]) row-normalized
    mi_matrix: np.ndarray = None

    panel_mask: np.ndarray = None  # bool per node: restrict sampling away from missing state
    missing_state: dict = field(default_factory=dict)  # node -> missing-state index or None

    def col_index(self, col):
        return self.columns.index(col)

    def fit_structure_and_params(self, discrete: np.ndarray):
        n = len(self.columns)
        n_states_arr = np.array([self.n_states[c] for c in self.columns])
        mi = _pairwise_mi(discrete, n_states_arr, self.alpha)
        self.mi_matrix = mi
        tree_edges = _max_spanning_tree_edges(mi)
        root, parent, children, order = _root_and_orient(tree_edges, n, mi)
        self.root, self.parent, self.children, self.order = root, parent, children, order
        self._fit_params(discrete)

    def _fit_params(self, discrete: np.ndarray):
        n_rows = discrete.shape[0]
        root_counts = np.bincount(discrete[:, self.root], minlength=self.n_states[self.columns[self.root]]).astype(float)
        nr = self.n_states[self.columns[self.root]]
        self.root_marginal = (root_counts + self.alpha) / (n_rows + self.alpha * nr)

        self.cpt = {}
        for node in self.order:
            if node == self.root:
                continue
            p = self.parent[node]
            np_ = self.n_states[self.columns[p]]
            nc = self.n_states[self.columns[node]]
            xp = discrete[:, p].astype(np.int64)
            xc = discrete[:, node].astype(np.int64)
            joint_idx = xp * nc + xc
            counts = np.bincount(joint_idx, minlength=np_ * nc).astype(float).reshape(np_, nc)
            cpt = (counts + self.alpha) / (counts.sum(axis=1, keepdims=True) + self.alpha * nc)
            self.cpt[node] = cpt

    def set_panel_restriction(self, panel_columns, missing_state_by_col):
        panel_cols = set(panel_columns)
        self.panel_mask = np.array([self.columns[i] in panel_cols for i in range(len(self.columns))])
        self.missing_state = {i: missing_state_by_col.get(self.columns[i]) for i in range(len(self.columns))}

    # ---------------- log-likelihood (for quantization selection only) -----
    def mean_log_likelihood(self, discrete: np.ndarray) -> float:
        n_rows = discrete.shape[0]
        ll = np.log(self.root_marginal[discrete[:, self.root]])
        for node in self.order:
            if node == self.root:
                continue
            p = self.parent[node]
            xp = discrete[:, p]
            xc = discrete[:, node]
            probs = self.cpt[node][xp, xc]
            ll = ll + np.log(np.clip(probs, 1e-300, None))
        return float(np.mean(ll))

    # ---------------- unconditional generation (vectorized, O = empty) -----
    def sample_unconditional(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n_cols = len(self.columns)
        out = np.empty((n, n_cols), dtype=np.int64)
        out[:, self.root] = _sample_categorical_batch(
            np.tile(self._apply_panel_restriction(self.root, self.root_marginal), (n, 1)), rng
        )
        for node in self.order:
            if node == self.root:
                continue
            p = self.parent[node]
            rows = self.cpt[node][out[:, p]]
            rows = self._apply_panel_restriction_batch(node, rows)
            out[:, node] = _sample_categorical_batch(rows, rng)
        return out

    # ---------------- conditional generation given arbitrary evidence ------
    def sample_conditional(self, evidence: dict, n_replicates: int, rng: np.random.Generator) -> np.ndarray:
        """evidence: {col_index: state}. Returns (n_replicates, n_cols) int array."""
        n_cols = len(self.columns)
        u = {}  # node -> unnormalized belief vector over its own states
        for node in reversed(self.order):
            ns = self.n_states[self.columns[node]]
            if node in evidence:
                local = np.zeros(ns)
                local[evidence[node]] = 1.0
            else:
                local = np.ones(ns)
            acc = local.copy()
            for c in self.children[node]:
                m = self.cpt[c] @ u[c]  # shape (n_states[node],)
                acc = acc * m
            s = acc.sum()
            if s <= 0:
                acc = local if local.sum() > 0 else np.ones(ns)
                s = acc.sum()
            u[node] = acc / s

        out = np.empty((n_replicates, n_cols), dtype=np.int64)
        root_belief = self.root_marginal * u[self.root]
        root_belief = self._apply_panel_restriction(self.root, root_belief)
        out[:, self.root] = _sample_categorical_batch(np.tile(root_belief, (n_replicates, 1)), rng)
        for node in self.order:
            if node == self.root:
                continue
            p = self.parent[node]
            if node in evidence:
                out[:, node] = evidence[node]
                continue
            row = self.cpt[node][out[:, p]] * u[node][None, :]
            row_sum = row.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0] = 1.0
            row = row / row_sum
            row = self._apply_panel_restriction_batch(node, row)
            out[:, node] = _sample_categorical_batch(row, rng)
        return out

    def _apply_panel_restriction(self, node: int, dist: np.ndarray) -> np.ndarray:
        if self.panel_mask is None or not self.panel_mask[node]:
            return dist
        ms = self.missing_state.get(node)
        if ms is None:
            return dist
        d = dist.copy()
        if d.sum() - d[ms] > 1e-12:
            d[ms] = 0.0
            d = d / d.sum()
        return d

    def _apply_panel_restriction_batch(self, node: int, rows: np.ndarray) -> np.ndarray:
        if self.panel_mask is None or not self.panel_mask[node]:
            return rows
        ms = self.missing_state.get(node)
        if ms is None:
            return rows
        rows = rows.copy()
        rest_mass = rows.sum(axis=1) - rows[:, ms]
        ok = rest_mass > 1e-12
        rows[ok, ms] = 0.0
        rows[ok] = rows[ok] / rows[ok].sum(axis=1, keepdims=True)
        return rows

    # ---------------- Fisher-score embedding (drives d_G) ------------------
    def fisher_scores(self, discrete: np.ndarray) -> np.ndarray:
        n_rows = discrete.shape[0]
        blocks = []
        for node in self.order:
            ns = self.n_states[self.columns[node]]
            xi = discrete[:, node]
            onehot = np.zeros((n_rows, ns))
            onehot[np.arange(n_rows), xi] = 1.0
            if node == self.root:
                p_k = np.tile(self.root_marginal, (n_rows, 1))
            else:
                p = self.parent[node]
                p_k = self.cpt[node][discrete[:, p]]
            score = onehot - p_k
            denom = np.sqrt(np.clip(p_k * (1 - p_k), 0, None) + self.fisher_eps)
            blocks.append(score / denom)
        return np.concatenate(blocks, axis=1)


def _sample_categorical_batch(prob_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """prob_rows: (n, k) nonnegative rows, not necessarily normalized.
    Returns (n,) int draws from the row-normalized categorical distribution."""
    row_sum = prob_rows.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    normalized = prob_rows / row_sum
    cdf = np.cumsum(normalized, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(prob_rows.shape[0])
    return (u[:, None] > cdf).sum(axis=1).astype(np.int64)
