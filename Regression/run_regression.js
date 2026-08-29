// Usage:
//   node run_regression.js <path-to-data-file> [--degree N] [--x colName] [--y colName]
//
// Reads a data file — one or more groups of lists, separated by blank
// lines — converts each group into the X/Y format the regression program
// needs, runs a fit on each group, and prints the results.

const fs = require('fs');
const path = require('path');
const { parseGroups } = require('./data_parser');
const { runRegression, formatEquation } = require('./regression');

function parseArgs(argv) {
  const args = { file: null, degree: null, x: null, y: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--degree') args.degree = parseInt(argv[++i], 10);
    else if (argv[i] === '--x') args.x = argv[++i];
    else if (argv[i] === '--y') args.y = argv[++i];
    else if (!args.file) args.file = argv[i];
  }
  return args;
}

function runOnGroup(group, index, total, args) {
  const { headers, columns, source, title } = group;
  const label = title || (total > 1 ? `Group ${index + 1}` : 'Data');

  console.log(`--- ${label} ---`);
  console.log(`${columns[0].length} rows (detected format: ${source})`);
  console.log(`Columns found: ${headers.join(', ')}`);

  const xIndex = args.x ? headers.indexOf(args.x) : 0;
  const yIndex = args.y ? headers.indexOf(args.y) : 1;
  if (xIndex === -1) { console.log(`  (skipped — no column named "${args.x}")\n`); return; }
  if (yIndex === -1) { console.log(`  (skipped — no column named "${args.y}")\n`); return; }

  const xs = columns[xIndex];
  const ys = columns[yIndex];
  console.log(`Using X = "${headers[xIndex]}", Y = "${headers[yIndex]}"`);

  try {
    const fit = runRegression(xs, ys, { degree: args.degree });
    console.log(`Best fit: degree ${fit.degree} ${fit.degree === 1 ? '(straight line)' : '(curve)'}`);
    console.log(`Equation: y = ${formatEquation(fit.coeffs)}`);
    console.log(`R² (fit quality, 0-1, closer to 1 is better): ${fit.r2.toFixed(4)}`);
  } catch (e) {
    console.log(`  (could not fit: ${e.message})`);
  }
  console.log('');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.file) {
    console.error('Usage: node run_regression.js <path-to-data-file> [--degree N] [--x colName] [--y colName]');
    process.exit(1);
  }

  const filePath = path.resolve(args.file);
  const rawText = fs.readFileSync(filePath, 'utf8');

  const { groups, errors } = parseGroups(rawText);
  console.log(`Found ${groups.length} data group${groups.length === 1 ? '' : 's'} in ${path.basename(filePath)}\n`);

  groups.forEach((group, i) => runOnGroup(group, i, groups.length, args));

  if (errors.length) {
    console.log('Some groups could not be read:');
    errors.forEach(e => console.log(`  ${e}`));
  }
}

main();
