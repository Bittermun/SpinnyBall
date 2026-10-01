#!/usr/bin/env node
/**
 * SpinnyBall AST & Code Graph CLI (Gauthier-style Repo-Map)
 * Scans both JavaScript (workbench/ interactive engine) and Python (numerical research modules).
 * Usage:
 *   node scripts/code_graph.mjs --stats
 *   node scripts/code_graph.mjs --query <SymbolOrTerm>
 *   node scripts/code_graph.mjs --all
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, '..');

const SCAN_DIRS = [
  'workbench',
  'dynamics',
  'control_layer',
  'scripts',
  'src',
  'tests',
  'simulations',
  'scenarios',
];

const EXCLUDE_DIRS = new Set([
  'node_modules',
  '__pycache__',
  '.git',
  '.dist_verify',
  '.venv',
  'archive',
  'docs',
  'scratch',
]);

function walkDir(dir) {
  const full = join(ROOT, dir);
  if (!existsSync(full)) return [];
  const entries = [];

  function recurse(curr) {
    const list = readdirSync(curr, { withFileTypes: true });
    for (const ent of list) {
      if (ent.isDirectory()) {
        if (!EXCLUDE_DIRS.has(ent.name)) {
          recurse(join(curr, ent.name));
        }
      } else if (ent.isFile()) {
        const ext = ent.name.split('.').pop()?.toLowerCase();
        if (['mjs', 'js', 'py', 'ts'].includes(ext)) {
          entries.push(join(curr, ent.name));
        }
      }
    }
  }

  recurse(full);
  return entries;
}

function extractSymbols(filePath) {
  const content = readFileSync(filePath, 'utf-8');
  const relPath = relative(ROOT, filePath).replace(/\\/g, '/');
  const lines = content.split('\n');
  const symbols = [];

  const isPython = filePath.endsWith('.py');
  const isJs = filePath.endsWith('.mjs') || filePath.endsWith('.js') || filePath.endsWith('.ts');

  lines.forEach((line, idx) => {
    const lineNum = idx + 1;
    const trimmed = line.trim();

    if (isJs) {
      // export function foo(...)
      let m = trimmed.match(/^export\s+(?:async\s+)?function\s+([A-Za-z0-9_$]+)/);
      if (m) symbols.push({ name: m[1], type: 'function', line: lineNum, file: relPath, exported: true });

      // export const / let foo =
      m = trimmed.match(/^export\s+(?:const|let)\s+([A-Za-z0-9_$]+)/);
      if (m) symbols.push({ name: m[1], type: 'constant', line: lineNum, file: relPath, exported: true });

      // export class Foo
      m = trimmed.match(/^export\s+class\s+([A-Za-z0-9_$]+)/);
      if (m) symbols.push({ name: m[1], type: 'class', line: lineNum, file: relPath, exported: true });

      // non-exported function foo(...)
      m = trimmed.match(/^function\s+([A-Za-z0-9_$]+)\s*\(/);
      if (m && !trimmed.startsWith('export')) symbols.push({ name: m[1], type: 'function', line: lineNum, file: relPath, exported: false });

      // test('name', ...)
      m = trimmed.match(/^test\(\s*['"`]([^'"`]+)['"`]/);
      if (m) symbols.push({ name: m[1], type: 'test', line: lineNum, file: relPath, exported: false });
    } else if (isPython) {
      // def foo(...)
      let m = trimmed.match(/^def\s+([A-Za-z0-9_]+)\s*\(/);
      if (m) symbols.push({ name: m[1], type: 'function', line: lineNum, file: relPath, exported: !m[1].startsWith('_') });

      // class Foo
      m = trimmed.match(/^class\s+([A-Za-z0-9_]+)/);
      if (m) symbols.push({ name: m[1], type: 'class', line: lineNum, file: relPath, exported: true });
    }
  });

  return { relPath, lines: lines.length, symbols };
}

function buildGraph() {
  const allFiles = SCAN_DIRS.flatMap(walkDir);
  const fileData = allFiles.map(extractSymbols);
  const symbolIndex = new Map();

  for (const f of fileData) {
    for (const sym of f.symbols) {
      const lower = sym.name.toLowerCase();
      if (!symbolIndex.has(lower)) symbolIndex.set(lower, []);
      symbolIndex.get(lower).push(sym);
    }
  }

  return { fileData, symbolIndex };
}

// CLI Command Handling
const args = process.argv.slice(2);
const graph = buildGraph();

if (args.includes('--stats') || args.length === 0) {
  const totalFiles = graph.fileData.length;
  const totalLoc = graph.fileData.reduce((acc, f) => acc + f.lines, 0);
  const totalSymbols = graph.fileData.reduce((acc, f) => acc + f.symbols.length, 0);

  const jsFiles = graph.fileData.filter(f => f.relPath.endsWith('.mjs') || f.relPath.endsWith('.js'));
  const pyFiles = graph.fileData.filter(f => f.relPath.endsWith('.py'));

  console.log(`\n=== SpinnyBall Code Graph Stats ===`);
  console.log(`Total Indexed Files:   ${totalFiles}`);
  console.log(`  - JavaScript Engine: ${jsFiles.length} files (${jsFiles.reduce((a, f) => a + f.lines, 0)} LOC)`);
  console.log(`  - Python Research:   ${pyFiles.length} files (${pyFiles.reduce((a, f) => a + f.lines, 0)} LOC)`);
  console.log(`Total Lines of Code:   ${totalLoc.toLocaleString()}`);
  console.log(`Total Indexed Symbols: ${totalSymbols.toLocaleString()}`);
  console.log(`====================================\n`);
  process.exit(0);
}

const queryIdx = args.indexOf('--query');
if (queryIdx !== -1 && args[queryIdx + 1]) {
  const term = args[queryIdx + 1].toLowerCase();
  console.log(`\n--- Symbol Query: "${args[queryIdx + 1]}" ---`);
  let matches = [];

  for (const [key, syms] of graph.symbolIndex.entries()) {
    if (key.includes(term)) {
      matches.push(...syms);
    }
  }

  if (matches.length === 0) {
    console.log(`No symbols matching "${args[queryIdx + 1]}" found.`);
  } else {
    for (const sym of matches) {
      console.log(`[${sym.type}] ${sym.name} -> ${sym.file}:${sym.line} (exported: ${sym.exported})`);
    }
  }
  console.log(`------------------------------------\n`);
  process.exit(0);
}

if (args.includes('--all')) {
  for (const f of graph.fileData) {
    if (f.symbols.length > 0) {
      console.log(`\nFile: ${f.relPath} (${f.lines} LOC)`);
      for (const s of f.symbols) {
        console.log(`  [${s.type}] ${s.name} (L${s.line})`);
      }
    }
  }
  process.exit(0);
}

console.log(`Usage: node scripts/code_graph.mjs [--stats | --query <term> | --all]`);
