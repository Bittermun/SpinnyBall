# SpinnyBall

**A small scientific laboratory for exploring motion, testing ideas, and keeping the physics accountable.**
I know few if any people read this, but if you do, please help me find the advanced concepts forum. -msunwc@gmail.com
Change a parameter, run a model, inspect the motion and balance errors, pin a comparison, and export a reproducible experiment.


| Laboratory | Question | Model |
|---|---|---|
| Orbital motion | When does an orbit become an escape trajectory? | Planar test particle in a fixed lunar-scale gravity field |
| Free spin | Why does an asymmetric body tumble around its middle axis? | Torque-free Euler equations and quaternion orientation |
| Momentum exchange | Can internal motion accelerate a system's center of mass? | Two point masses, a spring, and an optional external force |

These are idealized numerical experiments, not evidence that a magnetic mass-stream anchor is feasible. Older Python research remains available; see [research status](docs/RESEARCH_STATUS.md).

## Start the laboratory

From the repository root, with Python 3.10+ installed:

~~~sh
python -m workbench
~~~

This opens **http://127.0.0.1:8765/**. No pip install, Poetry, GPU, API key, account or internet connection is needed. A current browser with JavaScript and module workers is required. If the port is occupied, use `--port 8766` or `--port 0` for an available port. Use `--no-browser` to suppress opening a tab; Ctrl+C stops the server.

You can also serve the static `workbench/` directory over HTTP. Opening its HTML directly with a file URL does not support the module worker.

## Try an experiment

1. In **Orbital motion**, select **Circular** and press Play. Radius remains nearly constant.
2. Choose **Pin comparison**, then **Elliptical**. The amber dashed curve is the pinned run.
3. Change **Speed / circular speed** to **1.42**, then **Apply & run**. Positive specific orbital energy indicates an unbound orbit in this model.
4. Expand **Duration & numerical resolution**, halve the integration step and rerun. Compare the trajectory and reported balance errors.
5. **Save experiment** exports parameters, initial conditions, model/version information, retained samples and diagnostics. **Import** recalculates from parameters. **CSV** exports the sampled trajectory.

Playback replays computed samples: the whole run takes approximately 24 seconds at 1×. Pause, scrub or reset without changing the experiment. A comparison is held only in this tab; save it before closing.

## Reproduce and verify

Node.js 20+ is needed only for command-line experiments and tests; no npm dependencies are required:

~~~sh
node --test workbench/tests/physics.test.mjs
node --test workbench/tests/audit.test.mjs
node workbench/run.mjs orbit orbit.json
node workbench/run.mjs orbit.json replay.json
node workbench/verify-external.mjs orbit.json --mu 4.905e12
python -m unittest discover -s workbench/tests -p "test_*.py"
~~~

The worker and CLI share the same pure JavaScript engine. External projects can also use `workbench/verify-external.mjs` as an independent verification oracle to audit their own trajectory CSVs/JSONs against Keplerian conservation laws with SpinnyBall's strict `< 0.1%` balance threshold. Tests use independent analytic solutions, convergence, balance laws, invalid inputs and serialization. See [verification evidence](docs/WORKBENCH_VALIDATION.md).

## Documentation

- [Models and equations](docs/WORKBENCH_MODELS.md): units, assumptions, integrators, balance scales and sources.
- [Architecture](ARCHITECTURE.md): model, worker, interface and export boundaries.
- [Direction and research status](docs/RESEARCH_STATUS.md): what changed and what remains experimental.
- [Contributing](CONTRIBUTING.md): development and verification.

Previous interfaces are retained under [archive/interfaces](archive/interfaces/README.md), and earlier front-page documentation under [docs/archive/pre-workbench](docs/archive/pre-workbench/README.md). Historical benchmark, containment, material feasibility and mass-reduction claims are not workbench results.

## Limits and next work

The three experiments are not coupled into a mission simulator. This release has no magnetic bearings, thermal coupling, control system, collision dynamics, uncertainty inference or empirical calibration. Numerical balance diagnostics are not physical uncertainty bounds.

Useful next steps are analytic-reference overlays, uncertainty experiments, and adding verified interactions only after establishing their assumptions and independent tests.

MIT licensed; see [LICENSE](LICENSE). [GitHub repository](https://github.com/Bittermun/SpinnyBall).
