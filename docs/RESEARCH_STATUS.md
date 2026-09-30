# Direction decision and research status

## Decision

Make SpinnyBall a reproducible physics workbench for concept testing. The first release covers orbit shape/escape, rigid-body stability and momentum accounting. These preserve the interest in orbiting and spinning packets while giving each experiment a verifiable physical question.

The previous repository mixes a cislunar anchor proposal, general multiphysics framework, sweeps, control research and a browser “digital twin.” A polished interface alone would make unsupported results more convincing. The main entry point now uses a smaller core with inspectable equations and limits.

This is deliberate scope reduction, not a finding that all older ideas are impossible. Coupling models requires a specified physical question and independent validation. The workbench is not a comprehensive engineering simulator.

## Findings

The initial local baseline was commit 446cb98. Publication uses upstream 8551c5a as its base and preserves newer research.

- The previous digital_twin.html generates displayed stress, efficiency and stiffness using Math.random(). They were not measurements from a solved physical model.
- In sim/domains/mechanics_stream.py, the side-by-side force labeled repulsive points from i toward j and is added to i, making it attractive under that file's vector definition. Upstream repairs to dynamics/interball_magnetic.py are separate from this adapter.
- That adapter's T/R² times displacement expression has units N/m rather than N. Its nominal-circle restoring term also supplies no centripetal force at zero displacement. It is not adopted as the workbench orbit model.
- monte_carlo/cascade_runner.py uses different Python and accelerated stress formulas. The accelerated m(rω)²/(4πr²) expression has units kg/s² rather than Pa; the fallback includes an additional inverse length. The May 4 audit's validation label does not establish physical validity. Neither formula is adopted here.
- The four original tests in tests/test_simulation_invariants.py passed in the scheduling checkout, but do not establish validity of these formulas, randomized dashboard metrics or old feasibility claims.

These are bounded findings, not a full legacy audit.

## Status map

| Area | Status |
|---|---|
| workbench/ | Supported standalone experiment interface and engine |
| index.html, digital_twin.html | Entry pages directing users to the workbench and historical references |
| dynamics/, sim/, control_layer/, monte_carlo/, backend/, src/ | Retained Python research, not imported or validated by the workbench |
| research_data/, paper_model/, results/, sweeps and mission outputs | Historical/research evidence; inspect assumptions and provenance before reuse |
| Earlier technical specs, rigor plans, benchmark/mission/paper documents | Research documentation, not the current product specification |
| docs/archive/pre-workbench/, archive/interfaces/ | Labeled former entry points and documentation snapshots |

The root pyproject.toml continues to package the Python research. The workbench needs none of its optional ML, accelerator or backend dependencies.

## Claims and preservation

The canonical README withdraws earlier headline speedup, containment, material feasibility and mass-reduction figures as current product claims. Historical snapshots retain them, including upstream precision warnings, for provenance.

Agreement with analytic solutions supports implementation within stated assumptions. Conservation is necessary but not sufficient. Workbench results do not validate superconducting bearings, mass-stream anchors, packet capture, infrastructure mass or mission feasibility.

Existing datasets and unfinished local physics/control edits are preserved and excluded from this reform commit.

## Remaining work

1. Analytic-reference overlays and explicit convergence plots.
2. Parameter sweeps and uncertainty experiments with source provenance.
3. Versioned import migrations; currently incompatible engines are rejected.
4. Verified magnetic, thermal or control coupling only when a physical question requires it.
