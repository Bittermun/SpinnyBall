# Anchor Validation Decision

## Decision

For the current paper/demo phase, higher-fidelity validation is **not required yet**.

- Decision: `defer`
- Claim level: `L1-reduced-order`
- Higher-fidelity required now: `false`

## Rationale

The repo now contains:

- a reduced-order nonlinear anchor model
- controller comparisons
- robustness scenarios
- Sobol sensitivity analysis
- config-driven experiment execution
- standardized artifacts and a browser-readable report/dashboard

That is sufficient for the present reduced-order claim set and public-facing demo work.

## Trigger To Reopen This Decision

Revisit Newton or FEMM only if one of these becomes true:

1. Reviewer feedback challenges the credibility of the reduced-order force or damping assumptions.
2. Hardware design requires coil-field realism beyond the current reduced-order treatment.
3. Packet-level rigid-body effects become central to the argument rather than secondary.

## Next Fidelity Choice If Reopened

- Choose `Newton` for offline packet rigid-body validation.
- Choose `FEMM/pyFEMM` for coil-field realism.
- Do not pursue both simultaneously without a narrow validation question.
