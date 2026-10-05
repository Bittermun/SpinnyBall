#!/usr/bin/env node
/**
 * SpinnyBall Trajectory Invariant Verification Oracle CLI.
 * 
 * Audits an external orbit trajectory file or stream against Keplerian physical conservation laws.
 * 
 * Usage:
 *   node workbench/verify-external.mjs <file> [--mu <mu>] [--radius <r>] [--json] [--strict]
 *   cat trajectory.csv | node workbench/verify-external.mjs - [--mu <mu>]
 */

import { readFile } from 'node:fs/promises';
import { parseTrajectoryData, auditOrbitTrajectory } from './audit.mjs';

function parseArgs(args) {
  let file = null;
  let mu = 4.905e12; // default lunar mu
  let radius = 0;
  let jsonOutput = false;
  let strict = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--mu' && i + 1 < args.length) {
      mu = Number(args[++i]);
    } else if (arg === '--radius' && i + 1 < args.length) {
      radius = Number(args[++i]);
    } else if (arg === '--json') {
      jsonOutput = true;
    } else if (arg === '--strict') {
      strict = true;
    } else if (!arg.startsWith('-') && !file) {
      file = arg;
    }
  }

  return { file, mu, radius, jsonOutput, strict };
}

async function readInput(filePath) {
  if (!filePath || filePath === '-') {
    const chunks = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk);
    }
    return Buffer.concat(chunks).toString('utf8');
  }
  return readFile(filePath, 'utf8');
}

async function main() {
  const { file, mu, radius, jsonOutput, strict } = parseArgs(process.argv.slice(2));

  if (!file && process.stdin.isTTY) {
    console.error('Usage: node workbench/verify-external.mjs <trajectory-file | -> [--mu <mu>] [--radius <r>] [--json] [--strict]');
    process.exit(1);
  }

  try {
    const content = await readInput(file);
    const samples = parseTrajectoryData(content);
    const result = auditOrbitTrajectory(samples, { mu, r_body: radius });

    if (jsonOutput) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(`\n======================================================`);
      console.log(` SpinnyBall Trajectory Conservation Audit`);
      console.log(`======================================================`);
      console.log(`Status:            ${result.passed ? '✓ PASSED (all invariants < 0.1% drift)' : '✗ FAILED (invariants exceeded threshold)'}`);
      console.log(`Samples audited:   ${result.sampleCount}`);
      console.log(`Trajectory span:   ${result.duration.toFixed(2)} s`);
      console.log(`Gravitational μ:   ${mu.toExponential(4)} m³/s²`);
      console.log(`Closest approach:  ${(result.metrics.closestApproach / 1000).toFixed(2)} km`);
      console.log(`Max Energy Drift:  ${(result.metrics.maxEnergyRelError * 100).toFixed(5)}%`);
      console.log(`Max Ang Mom Drift: ${(result.metrics.maxAngMomRelError * 100).toFixed(5)}%`);
      console.log(`Eccentricity Drift:${result.metrics.maxEccentricityDrift.toExponential(4)}`);
      console.log(`Max Res. Accel:    ${result.metrics.maxParasiticAccel.toExponential(4)} m/s²`);
      
      if (result.warnings.length > 0) {
        console.log(`\nWarnings & Violations:`);
        for (const w of result.warnings) {
          console.log(` [!] ${w}`);
        }
      }
      console.log(`======================================================\n`);
    }

    if (strict && !result.passed) {
      process.exitCode = 1;
    }
  } catch (err) {
    if (jsonOutput) {
      console.error(JSON.stringify({ error: err.message }));
    } else {
      console.error(`Audit Error: ${err.message}`);
    }
    process.exitCode = 1;
  }
}

main();
