"""Single frozen structural model M: a quantized-state Chow-Liu tree.

Each of the 24 generation-panel coordinates is discretized into a small,
fixed number of states (quantile bins for numeric coordinates, native
levels for categorical coordinates, plus one explicit "missing" state for
every coordinate -- missingness is never silently replaced by a physical
value). A maximum-spanning-tree dependency structure (Chow-Liu, 1968) over
pairwise mutual information between quantized coordinates is learned from
the training data, with Laplace-smoothed conditional probability tables on
each edge.

The SAME frozen (bin edges + tree structure + CPTs) bundle provides:
  - distance(M, a, b): normalized Hamming distance between quantized state
    vectors, restricted to jointly-observed coordinates. Because it depends
    only on coarse bin membership, not within-bin magnitude, this geometry
    is deliberately decorrelated from raw per-coordinate Euclidean distance.
  - sample(M, observed, mask, seed): ancestral tree sampling (unconditional)
    or exact two-pass sum-product belief propagation for per-node posterior
    marginals given partial evidence (conditional), both read-only
    transformations of the same frozen tables.

The only empirically compared setting is the number of quantile bins B
("quantization resolution"), selected among a small declared candidate set
by an internal, held-out-free criterion: cross-validated training
log-likelihood using an internal split of the 50,000 training rows only.
This is the single comparison explicitly permitted by the task's
no-hyperparameter-tuning rule ("comparative selection only among the
declared quantization resolutions").
"""
import numpy as np
import pandas as pd
import pickle

QUANT_CANDIDATES = [3, 4, 6]
FIT_SEED = 20260917
LAPLACE_ALPHA = 0.5
CV_HOLD_FRACTION = 0.2


class StructuralModel:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, panel_df, numeric_cols, categorical_cols,
            quant_candidates=QUANT_CANDIDATES, fit_seed=FIT_SEED, verbose=True):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.panel_cols = self.numeric_cols + self.categorical_cols
        self.p = len(self.panel_cols)
        self.fit_seed = fit_seed

        df = panel_df[self.panel_cols].reset_index(drop=True)
        n = len(df)

        num_arr = df[self.numeric_cols].to_numpy(dtype=float)
        self.num_global_min = np.nanmin(num_arr, axis=0)
        self.num_global_max = np.nanmax(num_arr, axis=0)

        self.cat_categories = {}
        for col in self.categorical_cols:
            self.cat_categories[col] = sorted(df[col].dropna().unique().tolist())

        rng = np.random.default_rng(fit_seed)
        perm = rng.permutation(n)
        n_hold = int(round(n * CV_HOLD_FRACTION))
        hold_idx = perm[:n_hold]
        fit_idx = perm[n_hold:]

        trace_rows = []
        best = None
        for B in quant_candidates:
            edges_cv = self._fit_bin_edges(df.iloc[fit_idx], B)
            n_states_cv = self._n_states_vec(edges_cv)
            states_cv_fit = self._encode_states(df.iloc[fit_idx], edges_cv)
            tree_cv = self._learn_tree(states_cv_fit, n_states_cv)
            cpts_cv = self._fit_cpts(states_cv_fit, tree_cv, n_states_cv)
            states_cv_hold = self._encode_states(df.iloc[hold_idx], edges_cv)
            ll = self._avg_log_likelihood(states_cv_hold, tree_cv, cpts_cv)
            trace_rows.append({"quant_resolution": B, "cv_avg_log_likelihood": ll})
            if verbose:
                print(f"[quant={B}] CV avg log-likelihood = {ll:.4f}")
            if best is None or ll > best[0]:
                best = (ll, B)

        self.selection_trace = pd.DataFrame(trace_rows)
        _, self.quant_resolution = best

        # refit final model on the FULL training set at the selected resolution
        self.bin_edges = self._fit_bin_edges(df, self.quant_resolution)
        self.n_states = self._state_counts()
        states_full = self._encode_states(df, self.bin_edges)
        self.tree = self._learn_tree(states_full)
        self.cpts = self._fit_cpts(states_full, self.tree)

        # Per-bin empirical value pools (training values actually observed
        # in each bin), used to reconstruct continuous values from a
        # sampled bin so that within-bin marginal shape (skew, heavy
        # tails, etc.) matches the real data rather than assuming a
        # uniform distribution across the bin's raw interval.
        self.bin_values = {}
        for j, col in enumerate(self.numeric_cols):
            n_bins = len(self.bin_edges[col]) + 1
            vals = df[col].to_numpy(dtype=float)
            codes = states_full[:, j]
            pools = []
            for k in range(n_bins):
                pool = vals[(codes == k) & ~np.isnan(vals)]
                pools.append(pool)
            self.bin_values[col] = pools

        # Deterministic coordinate weighting for the distance metric, read
        # directly off the frozen tree's own edge structure: each
        # coordinate is weighted by its Chow-Liu mutual information with
        # its tree parent (or, for the root, its strongest neighbor MI).
        # This is a fixed post-processing of the single fitted model, not
        # an additional comparison across alternative geometries.
        MI = self.tree["MI"]
        weight = np.zeros(self.p)
        for v in range(self.p):
            par = self.tree["parent"][v]
            weight[v] = MI[v, :].max() if par < 0 else MI[v, par]
        self.coord_weight = weight
        return self

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------
    def _fit_bin_edges(self, df, B):
        edges = {}
        for col in self.numeric_cols:
            v = df[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            qs = np.quantile(v, np.linspace(0, 1, B + 1)[1:-1])
            edges[col] = np.unique(qs)
        return edges

    def _state_counts(self):
        counts = {}
        for col in self.numeric_cols:
            # len(edges)+1 real bins (codes 0..len(edges)), plus 1 missing code
            counts[col] = len(self.bin_edges[col]) + 2
        for col in self.categorical_cols:
            counts[col] = len(self.cat_categories[col]) + 1
        return counts

    def _encode_states(self, df, edges):
        """Return (n, p) int array of state codes. For column col with
        k non-missing states, code k is the missing state."""
        n = len(df)
        S = np.empty((n, self.p), dtype=np.int64)
        for j, col in enumerate(self.numeric_cols):
            v = df[col].to_numpy(dtype=float)
            e = edges[col]
            n_bins = len(e) + 1
            codes = np.searchsorted(e, v, side="right")
            codes = np.where(np.isnan(v), n_bins, codes)
            S[:, j] = codes
        for jj, col in enumerate(self.categorical_cols):
            j = len(self.numeric_cols) + jj
            cats = self.cat_categories[col]
            mapping = {c: i for i, c in enumerate(cats)}
            codes = df[col].map(mapping)
            n_cats = len(cats)
            codes = codes.where(~codes.isna(), n_cats).astype(np.int64).to_numpy()
            S[:, j] = codes
        return S

    def _n_states_vec(self, edges=None):
        if edges is None:
            return np.array([self.n_states[c] for c in self.panel_cols])
        out = []
        for col in self.numeric_cols:
            out.append(len(edges[col]) + 2)
        for col in self.categorical_cols:
            out.append(len(self.cat_categories[col]) + 1)
        return np.array(out)

    # ------------------------------------------------------------------
    # Chow-Liu structure learning
    # ------------------------------------------------------------------
    def _learn_tree(self, states, n_states_vec=None):
        p = self.p
        if n_states_vec is None:
            n_states_vec = self._n_states_vec()
        n = states.shape[0]

        MI = np.zeros((p, p))
        for i in range(p):
            ki = n_states_vec[i]
            ci = states[:, i]
            for j in range(i + 1, p):
                kj = n_states_vec[j]
                cj = states[:, j]
                joint = np.zeros((ki, kj))
                idx = ci * kj + cj
                counts = np.bincount(idx, minlength=ki * kj).reshape(ki, kj)
                joint = counts / n
                pi = joint.sum(axis=1, keepdims=True)
                pj = joint.sum(axis=0, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = joint / (pi * pj)
                    term = joint * np.log(ratio)
                term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                mi = term.sum()
                MI[i, j] = mi
                MI[j, i] = mi

        # maximum spanning tree via Prim's algorithm (p is small, ~24)
        in_tree = np.zeros(p, dtype=bool)
        in_tree[0] = True
        parent = np.full(p, -1, dtype=int)
        edges = []
        best_edge = MI[0, :].copy()
        best_from = np.zeros(p, dtype=int)
        best_edge[0] = -np.inf
        for _ in range(p - 1):
            j = np.argmax(best_edge)
            i = best_from[j]
            edges.append((i, j, MI[i, j]))
            parent[j] = i
            in_tree[j] = True
            best_edge[j] = -np.inf
            for k in range(p):
                if not in_tree[k] and MI[j, k] > best_edge[k]:
                    best_edge[k] = MI[j, k]
                    best_from[k] = j

        entropies = np.zeros(p)
        for i in range(p):
            ki = n_states_vec[i]
            counts = np.bincount(states[:, i], minlength=ki)
            probs = counts / n
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(np.where(probs > 0, probs * np.log(probs), 0.0))
            entropies[i] = ent
        root = int(np.argmax(entropies))

        children = [[] for _ in range(p)]
        adj = [[] for _ in range(p)]
        for i, j, w in edges:
            adj[i].append(j)
            adj[j].append(i)

        parent2 = np.full(p, -1, dtype=int)
        order = []
        visited = np.zeros(p, dtype=bool)
        stack = [root]
        visited[root] = True
        while stack:
            v = stack.pop()
            order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    parent2[u] = v
                    children[v].append(u)
                    stack.append(u)

        return {"root": root, "parent": parent2, "children": children,
                "topo_order": order, "adj": adj, "MI": MI}

    # ------------------------------------------------------------------
    # CPT fitting (Laplace-smoothed)
    # ------------------------------------------------------------------
    def _fit_cpts(self, states, tree, n_states_vec=None):
        if n_states_vec is None:
            n_states_vec = self._n_states_vec()
        p = self.p
        root = tree["root"]
        n = states.shape[0]

        root_counts = np.bincount(states[:, root], minlength=n_states_vec[root]).astype(float)
        root_probs = (root_counts + LAPLACE_ALPHA) / (root_counts.sum() + LAPLACE_ALPHA * n_states_vec[root])

        edge_cpt = {}
        for child in range(p):
            par = tree["parent"][child]
            if par < 0:
                continue
            kc = n_states_vec[child]
            kp = n_states_vec[par]
            idx = states[:, par] * kc + states[:, child]
            counts = np.bincount(idx, minlength=kp * kc).reshape(kp, kc).astype(float)
            counts += LAPLACE_ALPHA
            probs = counts / counts.sum(axis=1, keepdims=True)
            edge_cpt[child] = probs  # shape (kp, kc): P(child | parent)

        return {"root_probs": root_probs, "edge_cpt": edge_cpt}

    def _avg_log_likelihood(self, states, tree, cpts):
        root = tree["root"]
        ll = np.log(cpts["root_probs"][states[:, root]] + 1e-300)
        for child in range(self.p):
            par = tree["parent"][child]
            if par < 0:
                continue
            probs = cpts["edge_cpt"][child]
            ll = ll + np.log(probs[states[:, par], states[:, child]] + 1e-300)
        return float(np.mean(ll))

    # ------------------------------------------------------------------
    # Distance: normalized Hamming on quantized states
    # ------------------------------------------------------------------
    def encode(self, df):
        return self._encode_states(df[self.panel_cols], self.bin_edges)

    def distance_matrix(self, df):
        S = self.encode(df)
        n_states_vec = self._n_states_vec()
        missing_code = n_states_vec  # per-column missing code
        n = S.shape[0]
        w = self.coord_weight
        observed = S != missing_code[None, :]
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            joint_obs = observed[i][None, :] & observed
            mismatch = (S[i][None, :] != S) & joint_obs
            num = (mismatch * w[None, :]).sum(axis=1)
            den = (joint_obs * w[None, :]).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = num / den
            d[den == 0] = 1.0
            D[i, :] = d
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        return D

    # ------------------------------------------------------------------
    # Unconditional generation: ancestral tree sampling
    # ------------------------------------------------------------------
    def _state_to_value(self, col, state, rng):
        if col in self.cat_categories:
            cats = self.cat_categories[col]
            return cats[state]
        else:
            pool = self.bin_values[col][state]
            if len(pool) > 0:
                return rng.choice(pool)
            # empty bin (rare/degenerate): fall back to the bin's raw interval
            e = self.bin_edges[col]
            j = self.numeric_cols.index(col)
            lo = self.num_global_min[j] if state == 0 else e[state - 1]
            hi = self.num_global_max[j] if state == len(e) else e[state]
            return lo if hi <= lo else rng.uniform(lo, hi)

    def sample_unconditional(self, n, seed):
        rng = np.random.default_rng(seed)
        n_states_vec = self._n_states_vec()
        tree = self.tree
        cpts = self.cpts
        root = tree["root"]

        sampled_state = np.empty((n, self.p), dtype=np.int64)

        root_probs = cpts["root_probs"][:-1].copy()  # exclude missing state
        root_probs = root_probs / root_probs.sum()
        sampled_state[:, root] = rng.choice(len(root_probs), size=n, p=root_probs)

        for v in tree["topo_order"]:
            if v == root:
                continue
            par = tree["parent"][v]
            probs_table = cpts["edge_cpt"][v][:, :-1]  # exclude missing state for child
            row_sums = probs_table.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums <= 0, 1.0, row_sums)
            probs_table = probs_table / row_sums
            par_states = sampled_state[:, par]
            cum = np.cumsum(probs_table[par_states, :], axis=1)
            u = rng.random((n, 1))
            sampled_state[:, v] = (u < cum).argmax(axis=1)

        out = {}
        for j, col in enumerate(self.panel_cols):
            vals = [self._state_to_value(col, s, rng) for s in sampled_state[:, j]]
            out[col] = vals
        return pd.DataFrame(out, columns=self.panel_cols)

    # ------------------------------------------------------------------
    # Conditional generation: exact tree belief propagation
    # ------------------------------------------------------------------
    def _posterior_marginals(self, evidence_state, hidden_flags, n_states_vec):
        """evidence_state: length-p array, state code where observed, -1 if
        hidden. Returns dict node -> normalized posterior probability vector
        (over non-missing states) for every hidden node."""
        tree = self.tree
        cpts = self.cpts
        p = self.p
        root = tree["root"]
        children = tree["children"]
        parent = tree["parent"]

        def phi(v):
            k = n_states_vec[v]
            if evidence_state[v] >= 0:
                vec = np.zeros(k)
                vec[evidence_state[v]] = 1.0
                return vec
            return np.ones(k)

        upward = {}

        def collect(v):
            msg_children = np.ones(n_states_vec[v])
            for c in children[v]:
                collect(c)
                cpt = cpts["edge_cpt"][c]  # (k_v, k_c)
                mu_c = upward[c]
                msg_children = msg_children * (cpt @ mu_c)
            upward[v] = phi(v) * msg_children

        collect(root)

        downward = {}

        def distribute(v, incoming):
            belief = phi(v) * incoming
            for c in children[v]:
                others = np.ones(n_states_vec[v])
                for c2 in children[v]:
                    if c2 != c:
                        cpt2 = cpts["edge_cpt"][c2]
                        others = others * (cpt2 @ upward[c2])
                parent_side = phi(v) * incoming * others
                cpt = cpts["edge_cpt"][c]  # (k_v, k_c)
                lam = parent_side @ cpt  # -> (k_c,)
                downward[c] = lam
                distribute(c, lam)

        root_incoming = cpts["root_probs"].copy()
        distribute(root, root_incoming)

        beliefs = {}
        belief_root = phi(root) * root_incoming
        for c in children[root]:
            cpt = cpts["edge_cpt"][c]
            belief_root = belief_root * (cpt @ upward[c])
        beliefs[root] = belief_root

        def compute_belief(v):
            if v == root:
                return beliefs[root]
            lam = downward[v]
            msg_children = np.ones(n_states_vec[v])
            for c in children[v]:
                cpt = cpts["edge_cpt"][c]
                msg_children = msg_children * (cpt @ upward[c])
            return phi(v) * lam * msg_children

        result = {}
        for v in range(p):
            if hidden_flags[v]:
                b = compute_belief(v)
                b = b[:-1]  # exclude missing state
                s = b.sum()
                if s <= 0:
                    b = np.ones(len(b)) / len(b)
                else:
                    b = b / s
                result[v] = b
        return result

    def sample_conditional_batch(self, obs_df, hidden_mask_df, n_reps, seed):
        rng = np.random.default_rng(seed)
        m = len(obs_df)
        n_states_vec = self._n_states_vec()

        S_obs = self.encode(obs_df)
        hidden_arr = hidden_mask_df[self.panel_cols].to_numpy(dtype=bool)

        out_num = {c: np.empty((m, n_reps)) for c in self.numeric_cols}
        out_cat = {c: np.empty((m, n_reps), dtype=object) for c in self.categorical_cols}

        for i in range(m):
            evidence_state = S_obs[i].copy()
            hflags = hidden_arr[i].copy()
            missing_here = (evidence_state == n_states_vec) & ~hflags
            hflags = hflags | missing_here
            evidence_state = np.where(hflags, -1, evidence_state)

            if not hflags.any():
                continue
            post = self._posterior_marginals(evidence_state, hflags, n_states_vec)

            for j in np.where(hflags)[0]:
                col = self.panel_cols[j]
                probs = post[j]
                cum = np.cumsum(probs)
                u = rng.random(n_reps)
                states = np.searchsorted(cum, u, side="left")
                states = np.clip(states, 0, len(probs) - 1)
                vals = [self._state_to_value(col, s, rng) for s in states]
                if col in self.numeric_cols:
                    out_num[col][i, :] = vals
                else:
                    out_cat[col][i, :] = vals

        return out_num, out_cat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        state = self.__dict__.copy()
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.__dict__.update(state)
        return obj
