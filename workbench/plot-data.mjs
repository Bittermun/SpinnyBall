/**
 * Pure helper for selecting the nearest retained sample in a simulation run.
 * Does not synthesize or interpolate physical states.
 */

/**
 * Finds the nearest retained sample in an ascending time series.
 * @param {Array<{t: number}>} samples Retained samples array
 * @param {number} time Requested timestamp in seconds
 * @returns {{ index: number, sample: object } | null}
 */
export function nearestSampleAtTime(samples, time) {
  if (!Array.isArray(samples) || samples.length === 0) return null;
  if (typeof time !== 'number' || !Number.isFinite(time)) return null;

  if (time <= samples[0].t) return { index: 0, sample: samples[0] };
  if (time >= samples.at(-1).t) return { index: samples.length - 1, sample: samples.at(-1) };

  let low = 0;
  let high = samples.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const t = samples[mid].t;
    if (t === time) {
      return { index: mid, sample: samples[mid] };
    }
    if (t < time) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  // high is the earlier neighbor, low is the later neighbor
  const dHigh = Math.abs(time - samples[high].t);
  const dLow = Math.abs(time - samples[low].t);

  // Ties choose the earlier sample
  if (dHigh <= dLow) {
    return { index: high, sample: samples[high] };
  }
  return { index: low, sample: samples[low] };
}
