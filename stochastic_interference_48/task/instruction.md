# The Stochastic Balance Experiment

Forty-eight binary stochastic modules influence the dynamics of a shared noisy system. An excitation applied to a module can have positive or negative polarity. Under joint excitation, the modules interact according to an unknown, reproducible mechanism. The dynamics can depend on the recent symbolic history.

The laboratory has established that **exactly one polarity assignment, up to reversing every polarity**, makes the joint system emit independent, unbiased binary noise at the theoretical maximum entropy rate of one bit per emitted symbol. Every module must participate at unit magnitude.

**Determine the 48 polarities** that achieve this maximum-entropy condition. Fix module 01 at polarity +1, and assign every other module either +1 or -1.

## Measurements

All supplied files are in `/workspace/data/`. Each `.txt` file contains one consecutive binary sample path; different paths are independent experimental realizations, with no position-to-position alignment.

- `module_XX_plus.txt`, `module_XX_minus.txt`: each module recorded separately at the indicated signed calibration strength. The strength is recorded in `experiment.json`; these are not samples from the combined system.
- `calibration_XX.txt`, `calibration.json`: experiments involving multiple modules simultaneously. Their signed strengths are listed in `calibration.json`. The combination rule is not specified.
- `probe_XX.txt`: independent stochastic reference recordings available for comparative measurements.
- `experiment.json`: machine and data-format metadata.

The objective concerns full unit-strength excitation of all modules, not the calibration-strength experiment. The exact neutral configuration is guaranteed to be common to these strength scales.

Direct enumeration of all candidate configurations is not feasible within the task runtime. You may use any reproducible scientific or statistical method. You are not required to use any particular software package or mathematical representation.

## Deliverables

Write `/workspace/output/solution.py`, which must execute unattended with

```bash
cd /workspace && python3 output/solution.py
```

It must write `/workspace/output/solution.json` in this format:

```json
{"polarities": [1, -1, 1, 1]}
```

The example array is abbreviated: the submitted array must contain **exactly 48 integers** from `{-1, 1}`, and its first entry must equal `1`. The list order corresponds to modules 01 through 48.

Also write `/workspace/output/analysis.md`, briefly describing how the scientific result was obtained and checked. The grading is based on the 48 polarities, not on the explanatory narrative.
