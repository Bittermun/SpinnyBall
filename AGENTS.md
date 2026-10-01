# SpinnyBall — Repository Guidelines

A small scientific laboratory for motion, balance errors, and reproducible physics experiments.

## Core Invariants & Architecture
1. **Interactive Workbench (`workbench/`)**:
   - Zero-dependency, pure JavaScript ES module engine (Engine `1.0.0`, Schema `1`).
   - Runs off the main thread in a module Web Worker (`worker.mjs` $\leftrightarrow$ `physics.mjs`).
   - Local server: `python -m workbench` (serves `http://127.0.0.1:8765/`, standard library only).
2. **Strict Physical & Balance Accountability**:
   - **Balance Residual Threshold**: $<0.1\%$ of reference scale is a numerical warning.
   - **Fixed-Step Integration**: Uses fixed $h$ with a shortened final step to land exactly on requested duration.
   - **Deterministic & Unfalsifiable**: Imported JSON experiments are recomputed from parameters; uploaded/synthetic samples are never trusted as evidence.
   - **Step/Sample Limits**: At most 100,000 integration steps; default $\le 1,200$ retained samples.
3. **Research Separation**:
   - Python research modules (`dynamics/`, `control_layer/`, `scripts/`, `src/`) are historical/exploratory.
   - Never import unvalidated Python packages or dependencies into the standalone `workbench/`.

## AST & Code Graph CLI (Gauthier-Style)
```bash
node scripts/code_graph.mjs --stats
node scripts/code_graph.mjs --query <SymbolOrTerm>
```

## Primary Verification Commands
```bash
# 1. Physics Engine Analytic & Balance Tests (Pure Node, 19 tests, ~1s)
node --test workbench/tests/physics.test.mjs

# 2. Replay Verification
node workbench/run.mjs orbit orbit.json

# 3. Python Unit Tests
python -m unittest discover -s workbench/tests -p "test_*.py"
```
