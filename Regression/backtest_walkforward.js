// Usage:
//   node backtest_walkforward.js <path-to-data-file> [output-file]
//
// Compares two model-SELECTION methods head to head, using the exact same
// final holdout for both so the only thing that differs is how the model
// type gets chosen:
//
//   - single-split (what backtest_models.js / the /regression page use):
//     pick a model type from one fit/check cutoff inside the first 70% of
//     the data, then refit on that full 70% and score against the true
//     final 30%.
//   - walk-forward: pick a model type by its AVERAGE test R² across several
//     cutoffs walking forward through that same first 70% (train on 40%,
//     test on the next 10%; train on 50%, test on the next 10%; ...), then
//     refit on that full 70% and score against the same true final 30%.
//     Falls back to a single train/test pair when there's too little data
//     for multiple cutoffs, rather than failing outright.
//
// Writes a report with each dataset's result under both methods, plus the
// aggregate medians, so the comparison is apples-to-apples.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { selectByExtrapolation, selectByWalkForward } = require('./model_candidates');

function median(nums) {
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
function average(nums) { return nums.reduce((sum, v) => sum + v, 0) / nums.length; }

const FINAL_HOLDOUT_FRAC = 0.3; // matches backtest_models.js's usual 70% train pool / 30% held-out test

function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;

  const trainPoolLen = Math.floor(n * (1 - FINAL_HOLDOUT_FRAC));
  const testLen = n - trainPoolLen;
  if (testLen < 2) return { title: group.title, n, skipped: true, reason: `only ${n} rows` };

  const trainXs = xs.slice(0, trainPoolLen), trainYs = ys.slice(0, trainPoolLen);
  const testXs = xs.slice(trainPoolLen), testYs = ys.slice(trainPoolLen);

  // Single-split selection, run on the SAME 70% train pool.
  let singleResult = null;
  try {
    const sel = selectByExtrapolation(trainXs, trainYs, { fitFrac: 5/7, checkFrac: 2/7 }); // 50%/20% of the full data, i.e. 5/7 : 2/7 of this 70% pool
    const fit = sel.candidate.fit(trainXs, trainYs);
    singleResult = { name: sel.name, testR2: Regression.computeR2(testXs, testYs, fit.predict) };
  } catch (e) {
    singleResult = { name: null, error: e.message };
  }

  // Walk-forward selection, cutoffs built from the full series but capped
  // (via finalHoldoutFrac) to never see past the same 70% boundary. Falls
  // back to a single train/test pair internally when there's too little
  // data for multiple cutoffs.
  let walkResult = null;
  try {
    const sel = selectByWalkForward(xs, ys, { finalHoldoutFrac: FINAL_HOLDOUT_FRAC });
    const fit = sel.candidate.fit(trainXs, trainYs);
    walkResult = { name: sel.name, testR2: Regression.computeR2(testXs, testYs, fit.predict), cutoffCount: sel.cutoffCount, avgScore: sel.avgScore };
  } catch (e) {
    walkResult = { name: null, error: e.message };
  }

  return { title: group.title, n, testLen, single: singleResult, walk: walkResult };
}

function main() {
  const [, , inputPath, outputArg] = process.argv;
  if (!inputPath) {
    console.error('Usage: node backtest_walkforward.js <path-to-data-file> [output-file]');
    process.exit(1);
  }
  const filePath = path.resolve(inputPath);
  const outputPath = path.resolve(outputArg || 'backtest_walkforward_results.txt');

  const rawText = fs.readFileSync(filePath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);

  const results = groups.map(evaluateDataset);
  const usable = results.filter(r => !r.skipped && r.single.testR2 !== undefined && r.walk.testR2 !== undefined);
  const skipped = results.filter(r => r.skipped || r.single.testR2 === undefined || r.walk.testR2 === undefined);

  const singleR2s = usable.map(r => r.single.testR2);
  const walkR2s = usable.map(r => r.walk.testR2);
  const medianSingle = median(singleR2s), avgSingle = average(singleR2s);
  const medianWalk = median(walkR2s), avgWalk = average(walkR2s);
  const walkWins = usable.filter(r => r.walk.testR2 > r.single.testR2).length;
  const singleWins = usable.filter(r => r.single.testR2 > r.walk.testR2).length;
  const ties = usable.length - walkWins - singleWins;

  const lines = [];
  lines.push(`Walk-forward vs single-split selection — ${path.basename(filePath)}`);
  lines.push(`Both methods score against the SAME final ${(FINAL_HOLDOUT_FRAC*100).toFixed(0)}% holdout; only the`);
  lines.push(`selection method inside the remaining ${((1-FINAL_HOLDOUT_FRAC)*100).toFixed(0)}% differs.`);
  lines.push(`${usable.length} of ${groups.length} datasets usable by both methods (${skipped.length} skipped)`);
  lines.push('');
  lines.push(`MEDIAN test R² — single-split:   ${medianSingle.toFixed(4)}`);
  lines.push(`MEDIAN test R² — walk-forward:   ${medianWalk.toFixed(4)}`);
  lines.push(`AVERAGE test R² — single-split:  ${avgSingle.toFixed(4)}`);
  lines.push(`AVERAGE test R² — walk-forward:  ${avgWalk.toFixed(4)}`);
  lines.push('');
  lines.push(`Walk-forward better: ${walkWins} datasets | Single-split better: ${singleWins} | Tied: ${ties}`);
  lines.push('');
  lines.push('Per-dataset results:');
  usable
    .slice()
    .sort((a, b) => (b.walk.testR2 - b.single.testR2) - (a.walk.testR2 - a.single.testR2))
    .forEach(r => {
      lines.push(`  ${r.title}  (n=${r.n})`);
      lines.push(`    single-split: ${r.single.name}  test R²=${r.single.testR2.toFixed(4)}`);
      lines.push(`    walk-forward: ${r.walk.name}  test R²=${r.walk.testR2.toFixed(4)}  (${r.walk.cutoffCount} cutoffs, avg walk-forward score ${r.walk.avgScore.toFixed(4)})`);
    });

  if (skipped.length) {
    lines.push('');
    lines.push('Skipped:');
    skipped.forEach(r => lines.push(`  ${r.title} — ${r.reason || (r.single && r.single.error) || (r.walk && r.walk.error) || 'unknown'}`));
  }

  const report = lines.join('\n') + '\n';
  fs.writeFileSync(outputPath, report);
  console.log(report);
  console.log(`Full report written to ${outputPath}`);
}

main();
