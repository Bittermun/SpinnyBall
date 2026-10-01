#!/usr/bin/env node
/**
 * SpinnyBall Antigravity Stop Gate Hook
 * Runs when the agent completes its turn.
 * - If workbench .mjs files were modified, runs `node --test workbench/tests/physics.test.mjs`
 * - If Python .py files were modified, runs Python py_compile syntax check.
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

function findPython() {
  const candidates = [
    process.env.PYTHON_EXECUTABLE,
    'C:\\Users\\msunw\\Downloads\\TheFoundationProtocol\\.dist_verify\\runtime\\Scripts\\python.exe',
    `${process.env.LOCALAPPDATA || ''}\\Programs\\Python\\Python312\\python.exe`,
    `${process.env.LOCALAPPDATA || ''}\\Programs\\Python\\Python311\\python.exe`,
  ];
  for (const c of candidates) {
    if (c && fs.existsSync(c)) return c;
  }
  return null;
}

function main() {
  const cwd = process.cwd();
  let statusOut = '';
  try {
    statusOut = execFileSync('git', ['status', '--porcelain'], {
      cwd,
      encoding: 'utf8',
      timeout: 5000,
    });
  } catch {
    process.exit(0);
  }

  const modified = statusOut
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 3 && !l.startsWith('D '))
    .map((l) => {
      const raw = l.slice(2).trim();
      const arrow = raw.indexOf(' -> ');
      return arrow >= 0 ? raw.slice(arrow + 4).trim() : raw;
    });

  const modifiedMjs = modified.some(f => f.endsWith('.mjs') || f.endsWith('.js'));
  const modifiedPy = modified.some(f => f.endsWith('.py'));

  // 1. If JS/MJS files modified, run physics tests
  if (modifiedMjs) {
    try {
      execFileSync('node', ['--test', 'workbench/tests/physics.test.mjs'], {
        cwd,
        encoding: 'utf8',
        timeout: 10000,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (err) {
      const errorMsg = (err.stderr || err.stdout || err.message || 'Unknown error').slice(0, 1500);
      console.log(JSON.stringify({
        decision: 'continue',
        reason: `SpinnyBall physics test failed before completion:\n${errorMsg}`,
      }));
      process.exit(0);
    }
  }

  // 2. If Python files modified, run py_compile
  if (modifiedPy) {
    const pythonExe = findPython();
    if (pythonExe) {
      for (const rel of modified.filter(f => f.endsWith('.py'))) {
        const full = path.resolve(cwd, rel);
        if (fs.existsSync(full)) {
          try {
            execFileSync(pythonExe, ['-m', 'py_compile', full], {
              cwd,
              encoding: 'utf8',
              timeout: 5000,
              stdio: ['ignore', 'pipe', 'pipe'],
            });
          } catch (err) {
            console.log(JSON.stringify({
              decision: 'continue',
              reason: `Python py_compile syntax error in ${rel}:\n${err.stderr || err.stdout || err.message}`,
            }));
            process.exit(0);
          }
        }
      }
    }
  }

  console.log(JSON.stringify({ decision: 'allow' }));
  process.exit(0);
}

main();
