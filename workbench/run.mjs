import { readFile, writeFile } from 'node:fs/promises';
import { configFor, readExperiment, simulate } from './physics.mjs';

const [source = 'orbit', destination] = process.argv.slice(2);
try {
  const config = ['orbit', 'spin', 'exchange'].includes(source) ? configFor(source) : readExperiment(JSON.parse(await readFile(source, 'utf8')));
  const result = simulate(config);
  if (destination) await writeFile(destination, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ config: result.config, diagnostics: result.diagnostics, ...(destination ? { saved: destination } : {}) }, null, 2));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
