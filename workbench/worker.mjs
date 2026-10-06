import { simulate } from './physics.mjs';
import { runOrbitSweep } from './sweep.mjs';

self.onmessage = ({ data }) => {
  if (data.kind === 'orbit-speed-sweep') {
    try {
      const result = runOrbitSweep(data.config, data.range, (completed, total) => {
        self.postMessage({ id: data.id, kind: 'sweep-progress', completed, total });
      });
      self.postMessage({ id: data.id, kind: 'sweep-result', result });
    } catch (error) {
      self.postMessage({ id: data.id, kind: 'sweep-result', error: error.message });
    }
    return;
  }
  try { self.postMessage({ id: data.id, result: simulate(data.config) }); }
  catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
