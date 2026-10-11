export function nextImportGeneration(currentGeneration) {
  return currentGeneration + 1;
}

export function isCurrentImportGeneration(generation, currentGeneration) {
  return generation === currentGeneration;
}

export function resetInspectionState() {
  return { inspectedIndex: null, inspectingPlot: null };
}

export function sweepWorkerFailureUi(message) {
  return {
    sweepBusy: false,
    runSweepDisabled: false,
    cancelSweepHidden: true,
    statusMessage: message || 'Sweep worker failed before completion. Try running the sweep again.',
    statusError: true,
  };
}
