# Contributing to SpinnyBall

Read [the README](README.md), [model contracts](docs/WORKBENCH_MODELS.md) and [research status](docs/RESEARCH_STATUS.md). Keep each experiment's question, assumptions and evidence understandable.

## Development

~~~sh
python -m workbench
node --test workbench/tests/physics.test.mjs
python -m unittest discover -s workbench/tests -p "test_*.py"
~~~

Python 3.10+ serves the interface without installed packages. Node.js 20+ runs tests and CLI experiments without npm packages. Refresh after editing; the local server disables caching.

For a physics change, add independent reference cases, balance checks and timestep convergence where appropriate. Document units and validity limits. Do not tune equations to reproduce old headline results. For UI changes, exercise run, pause/reset, scenario switching, pin/clear, export/import, invalid input and narrow screens.

Changes to exported-result meaning need an engine version update and an explicit import compatibility decision. Do not trust imported samples as computed output. Keep calculations bounded and outside the UI thread.

## Older research

The root pyproject.toml and Poetry environment apply to the retained Python research. Changes there need their own verification: workbench tests do not validate them. Preserve research datasets and provenance.

Submit focused branches/PRs describing user-visible behavior, scientific assumptions, tests actually run and known limits. Credit external implementations and preserve licenses. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
