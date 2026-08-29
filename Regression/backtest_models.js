// Usage:
//   node backtest_models.js <path-to-data-file> [output-file]
//
// For every dataset, the last 30% of its rows is set aside as the final
// test set and never touched until the very end. The remaining 70% (the
// "training pool") is itself split in half:
//   - FIT half (first 50% of the dataset)   — every candidate model is fit
//     on this alone.
//   - CHECK half (next 20% of the dataset)  — used to score how well each
//     fitted model extrapolates into data just past what it was fit on.
//     A model that overfits the fit half (chases its noise instead of its
//     shape) will predict the check half badly — that's the whole point:
//     the penalty for "poor extrapolation" is simply that bad score, no
//     separate formula needed. This directly targets the failure mode a
//     parameter-count-based penalty like adjusted R² can miss: a model can
//     be a great interpolator and a terrible extrapolator at the same time,
//     and adjusted R² has no way to see that coming (see the note in
//     model_candidates.js for what happened when this script used that instead).
// Whichever model extrapolates best onto the check half is then re-fit on
// the FULL training pool (fit + check halves together, i.e. the whole 70%)
// — no reason to throw that data away once the model type is chosen — and
// that final fit is scored against the true held-out last 30%.
//
// Writes a report file with each dataset's final test R² plus the median
// across all of them, and prints the same summary to the console.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { selectByExtrapolation } = require('./model_candidates');

// The model list itself now lives in model_candidates.js, shared with the
// /regression page's "Auto (best by extrapolation check)" option — both
// pick models exactly the same way. See that file for the full candidate
// list and why quartic/quintic are included rather than banned outright.

function median(nums) {
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function evaluateDataset(group) {
  const xs = group.columns[0];
  const ys = group.columns[1];
  const n = xs.length;

  const trainPoolLen = Math.floor(n * 0.7);
  const testLen = n - trainPoolLen;

  if (testLen < 2) {
    return { title: group.title, n, skipped: true, reason: `only ${n} rows — not enough for a fit/check/test three-way split` };
  }

  const trainXs = xs.slice(0, trainPoolLen), trainYs = ys.slice(0, trainPoolLen);
  const testXs = xs.slice(trainPoolLen), testYs = ys.slice(trainPoolLen);

  // Model TYPE gets picked by extrapolating from the first 45% of the
  // dataset onto the next 25% (ending exactly at the 70% mark, i.e. the
  // edge of the training pool) — the true last-30% TEST set stays
  // untouched until after that. fitFrac needs to be generous enough that a
  // real curve's shape (e.g. a parabola's turning point) actually falls
  // inside the FIT window — an earlier 35/35 split cut some datasets off
  // before their defining feature was visible, making a flat/monotonic
  // model look like the safer extrapolator even when the data was
  // genuinely, say, a parabola.
  let selection;
  try {
    selection = selectByExtrapolation(xs, ys, { fitFrac: 0.5, checkFrac: 0.2 });
  } catch (e) {
    return { title: group.title, n, skipped: true, reason: e.message };
  }

  // Now that the model TYPE is chosen, re-fit it on the full training pool
  // (fit half + check half together) — no reason to leave that data on the
  // table for the final model — and score that against the true held-out
  // last 30%.
  const finalFit = selection.candidate.fit(trainXs, trainYs);
  const testR2 = Regression.computeR2(testXs, testYs, finalFit.predict);

  return {
    title: group.title, n, fitLen: selection.fitLen, checkLen: selection.checkLen, testLen,
    model: selection.name, fitR2: selection.fitR2, checkR2: selection.checkR2, testR2,
  };
}

function main() {
  const [, , inputPath, outputArg] = process.argv;
  if (!inputPath) {
    console.error('Usage: node backtest_models.js <path-to-data-file> [output-file]');
    process.exit(1);
  }

  const filePath = path.resolve(inputPath);
  const outputPath = path.resolve(outputArg || 'backtest_results.txt');

  const rawText = fs.readFileSync(filePath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);

  const results = groups.map(evaluateDataset);
  const evaluated = results.filter(r => !r.skipped);
  const skipped = results.filter(r => r.skipped);
  const testR2s = evaluated.map(r => r.testR2);
  const medianTestR2 = testR2s.length ? median(testR2s) : null;

  const lines = [];
  lines.push(`Backtest of ${path.basename(filePath)}`);
  lines.push('Each dataset: fit on the first 50% (FIT half), model type chosen by how well it');
  lines.push('extrapolates onto the next 20% (CHECK half), then re-fit on that full 70% and');
  lines.push('scored against the held-out last 30% (TEST).');
  lines.push(`${evaluated.length} of ${groups.length} datasets evaluated (${skipped.length} skipped — too few rows)`);
  lines.push('');
  lines.push(`MEDIAN TEST R² ACROSS ALL DATASETS: ${medianTestR2 === null ? 'n/a' : medianTestR2.toFixed(4)}`);
  lines.push('');
  lines.push('Per-dataset results:');
  evaluated
    .slice()
    .sort((a, b) => b.testR2 - a.testR2)
    .forEach(r => {
      lines.push(`  ${r.title}`);
      lines.push(`    rows: ${r.n} (fit ${r.fitLen} / check ${r.checkLen} / test ${r.testLen})   model: ${r.model}`);
      lines.push(`    fit R²: ${r.fitR2.toFixed(4)}   check (extrapolation) R²: ${r.checkR2.toFixed(4)}   test R²: ${r.testR2.toFixed(4)}`);
    });

  if (skipped.length) {
    lines.push('');
    lines.push('Skipped:');
    skipped.forEach(r => lines.push(`  ${r.title} — ${r.reason}`));
  }

  const report = lines.join('\n') + '\n';
  fs.writeFileSync(outputPath, report);
  console.log(report);
  console.log(`Full report written to ${outputPath}`);
}

main();
