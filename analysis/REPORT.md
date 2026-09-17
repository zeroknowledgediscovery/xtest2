# Discovery report

## Result

```
analysis_status   = DISCOVERY_SUCCESS
primary_discovery = YES
```

All required gates passed on the single, prespecified, frozen structural
model, in the single one-shot outer/blind evaluation performed by this
repository (see `outer_evaluation_history.jsonl`, exactly one record).

## The model

A single frozen structural model M: a semiparametric ("extended rank
likelihood") Gaussian copula over the 24-coordinate generation panel (22
numeric measured/chemistry variables + `crys` + `dimensionality`), fit with
no quantization and no tuned hyperparameter of any kind:

- Each coordinate's marginal is its own empirical training CDF (numeric) or
  empirical category frequency under a fixed alphabetical ordering
  (categorical), giving a latent standard-normal score for every observed
  value.
- The p x p latent correlation matrix Sigma is estimated by a plug-in EM
  algorithm for the missing-data multivariate normal (30 fixed iterations,
  1e-6 ridge), fit once on the 50,000-row outer training set.
- **Geometry** `d_G(a,b)`: each material's latent vector is completed
  (missing coordinates filled by the model's own conditional-mean
  imputation under the frozen Sigma), then compared by a whitened
  Mahalanobis distance using Sigma^-1.
- **Generation** `X ~ P_M(X | X_O)`: unconditional sampling draws
  `Z ~ N(0, Sigma)` and inverts each coordinate's marginal; conditional
  completion samples the hidden latent coordinates from the standard
  Gaussian conditional distribution given the observed ones (same Sigma,
  same conditioning machinery used for geometry's imputation step) and
  inverts the marginals.

Both learned artifacts (Sigma, the per-column empirical marginals) are used
by both geometry and generation (`model_usage.json`); nothing is fit
independently on either side.

## The four scientific claims

1. **The learned geometry differs from conventional material similarity.**
   Confirmed. Mean 20-NN overlap with the standardized-numeric comparator
   d_E is 26.4% (upper 95% CI 27.2%), and with the mixed-type comparator
   d_M is 18.8% (upper 95% CI 19.7%) -- both far below the 50% novelty
   threshold. Global and anchor-wise rank correlations are likewise well
   under the 0.75 guard.

2. **It retrieves blind external functional analogs under tightly
   raw-distance-matched controls.** Confirmed. Among 1,455 evaluable
   anchors, materials retrieved as geometrically close (among the
   conventionally-*far* 10% tail) are functionally closer in the 8 blind
   transport variables than raw-distance-matched controls: retrieval ratio
   0.935 (upper 95% CI 0.952 < 1), win rate 64.9% (lower 95% CI 62.5% >
   55%), with the controls matched to within 8.4% of the far-distance scale
   (upper 95% CI 9.9% < 10%).

3. **The underlying structural model generates coherent multivariate
   material states from no observations.** Confirmed. All 10,000
   unconditional draws are fully populated and valid, 100% unique, 0% exact
   training-row matches; and a fixed ExtraTrees verifier trained on the
   generated materials retains cross-coordinate structure indistinguishable
   in kind from real materials (all 24 panel coordinates informative and
   improved over the independent-marginal baseline; retention G = 0.689,
   lower 95% CI 0.688 > 0).

4. **The same model reconstructs hidden material state from arbitrary
   partial observations.** Confirmed. At 50% and 80% coordinate masking,
   the model's completion loss is a fraction of the independent-marginal
   baseline's (ratio 0.590 and 0.771 respectively, both upper 95% CIs < 1),
   with 91.7% of individually evaluable coordinates beating the baseline at
   both levels.

No non-quantization hyperparameter, model family, geometry family, or
sampler setting was compared or selected at any point (`no_model_tuning_pass
= YES`; `selection_trace.csv` contains exactly one configuration row). The
outer/blind evaluation was performed exactly once.
