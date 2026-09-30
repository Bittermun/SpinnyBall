import { simulate } from './physics.mjs';
self.onmessage = ({ data }) => {
  try { self.postMessage({ id: data.id, result: simulate(data.config) }); }
  catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
