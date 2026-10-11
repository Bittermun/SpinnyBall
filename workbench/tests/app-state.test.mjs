import test from 'node:test';
import assert from 'node:assert/strict';
import { nextImportGeneration, isCurrentImportGeneration, resetInspectionState, sweepWorkerFailureUi } from '../app-state.mjs';

test('sweepWorkerFailureUi resets busy controls and reports actionable error', () => {
  const ui = sweepWorkerFailureUi('Sweep worker sent unreadable data. Try running the sweep again.');
  assert.equal(ui.sweepBusy, false);
  assert.equal(ui.runSweepDisabled, false);
  assert.equal(ui.cancelSweepHidden, true);
  assert.equal(ui.statusError, true);
  assert.match(ui.statusMessage, /Try running the sweep again/);
});

test('import generation helper ignores stale imports', () => {
  const first = nextImportGeneration(0);
  const second = nextImportGeneration(first);
  assert.equal(isCurrentImportGeneration(first, second), false);
  assert.equal(isCurrentImportGeneration(second, second), true);
});

test('resetInspectionState clears inspected index and selected plot', () => {
  assert.deepEqual(resetInspectionState(), { inspectedIndex: null, inspectingPlot: null });
});
