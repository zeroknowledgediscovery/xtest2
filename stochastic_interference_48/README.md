# Stochastic interference nullspace — 48-module Honeydew candidate

This branch contains a reproducible synthetic stochastic-science challenge.

48 independently recorded binary stochastic generators exhibit a hidden signed relation. Every output has balanced lower-order word statistics, but the joint model becomes independent fair noise only at the planted sign vector (up to global sign).

## Layout

- `task/instruction.md`: blind agent-facing task
- `task/environment/data/`: generated binary streams, controls, and probes
- `task/solution/`: private reference oracle
- `task/tests/`: private exact-answer verifier
- `private/construct.py`: seeded source and recordings generator
- `private/recover.py`: differential probe recovery experiment
- `private/validate.py`: construction checks

## Build data

Run from `stochastic_interference_48`:

```bash
python -m pip install numpy scipy numba
python private/construct.py
python private/recover.py 7
python private/recover.py 7 first
python private/recover.py 7 second
WORKSPACE_DIR="$PWD/task/environment" python task/solution/solve.py
```

A branch-specific GitHub Action can automatically build and commit the generated data.

## Status and limitations

This is a **worked proof of concept**, not a validated agent-failing Honeydew task.
The reference solver uses a controlled **conditional sequence-likelihood divergence surrogate**, not the published compiled LSmash implementation. The planted equal-magnitude emission morphs may make the example easier for generic statistical solvers. No Harbor agent campaign has been run. Neither solver instructions nor the candidate grader require a specific modeling method.

Private truth and verifier belong outside the agent-visible task environment.
