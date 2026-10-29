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

function average(nums) {
  return nums.reduce((sum, v) => sum + v, 0) / nums.length;
}

// Plain average is dominated by rare catastrophic outliers (we've seen a
// single dataset drag it to -49.5 billion). This version drops outliers
// before averaging what's left — a robust "typical performance" number
// that still reflects more than one point, unlike the median alone.
//
// The fence used to be a flat +/-0.17 around the median, picked by eye. It
// turned out to equal 1.5 * 1.4826 * MAD (the classic "1.5 robust standard
// deviations" outlier rule, with 1.4826 the constant that makes MAD track
// the standard deviation on normal data) — but only using a SINGLE MAD
// computed across the whole set, which assumes the spread is the same on
// both sides of the median. Test R² isn't: good fits cluster tightly just
// under 1.0 (a small, well-behaved right-hand spread), while bad fits
// scatter arbitrarily far below the median (a huge, heavy left-hand
// spread) — see [[feedback-bootstrap-methodology]]. A single symmetric MAD
// either overcorrects on the tight side or undercorrects on the scattered
// side. This computes MAD separately for the below-median and
// above-median halves ("double MAD") and fences each side with its own —
// verified against this corpus to flag far fewer good-but-unlucky fits as
// outliers than the old flat threshold did (7/56 vs 18/56 on the
// full-variance metric), while still catching the real blowups.
function doubleMADFence(nums, med, k = 1.5) {
  const below = nums.filter(v => v <= med).map(v => med - v);
  const above = nums.filter(v => v > med).map(v => v - med);
  // A side with too few points (or all identical to the median) can't
  // support its own MAD estimate — don't trim that side at all rather than
  // collapse the fence onto the median and flag everything on it.
  const leftMAD = below.length >= 2 ? median(below) : 0;
  const rightMAD = above.length >= 2 ? median(above) : 0;
  const lower = leftMAD > 0 ? med - k * 1.4826 * leftMAD : -Infinity;
  const upper = rightMAD > 0 ? med + k * 1.4826 * rightMAD : Infinity;
  return { lower, upper };
}

function trimmedAverage(nums) {
  if (!nums.length) return { value: null, kept: 0, total: 0 };
  const med = median(nums);
  const { lower, upper } = doubleMADFence(nums, med);
  const kept = nums.filter(v => v >= lower && v <= upper);
  return { value: kept.length ? average(kept) : null, kept: kept.length, total: nums.length };
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
  // last 30%. This CAN fail even though selection just succeeded: a
  // candidate's .fit() is only run on the FIT half during selection (the
  // CHECK half is only ever used through .predict(), which doesn't
  // validate anything) — so a model requiring e.g. every Y > 0 can pass
  // selection on an all-positive FIT half, then blow up here once the
  // CHECK half's points (which might dip to zero or negative) are folded
  // into the full training pool it's refit on.
  let finalFit;
  try {
    finalFit = selection.candidate.fit(trainXs, trainYs);
  } catch (e) {
    return { title: group.title, n, skipped: true, reason: `${selection.name} passed model selection but failed on the final refit: ${e.message}` };
  }
  // Sequential (walk-forward): every candidate except AR(1) ignores the
  // extra "real previous y" argument and scores exactly as before; AR(1)
  // uses it to forecast one real step at a time instead of a blind
  // multi-step recursion from the end of training (see regression.js's
  // fitAR1 comment — this was the whole point of adding it).
  const lastTrainY = trainYs[trainYs.length - 1];
  const testR2 = Regression.computeR2Sequential(lastTrainY, testXs, testYs, finalFit.predict);

  // Same held-out score, but with the R² denominator (ssTot) computed from
  // the FULL dataset's variance instead of just the test slice's own. Plain
  // computeR2 re-centers on the test slice's own mean — if that slice
  // happens to have near-zero variance (e.g. the tail of a curve that's
  // already plateaued), even a small absolute miss gets divided by a
  // near-zero denominator and produces a wildly exaggerated bad score. This
  // version scores the same miss against a fixed, stable per-dataset
  // baseline instead. See computeR2FullVariance's own comment in
  // regression.js for the "PSI vs mean Polsby-Popper score" example that
  // motivated it (-9408 by the old metric on a ~0.03-0.07 absolute miss).
  const testR2FullVariance = Regression.computeR2FullVarianceSequential(ys, lastTrainY, testXs, testYs, finalFit.predict);

  // Persistence baselines, fit on the same training pool and scored against
  // the same held-out test set — the standard "did the model beat doing
  // nothing" check from time-series forecasting. Neither one looks at the
  // selected model type at all, so a dataset where the selected model can't
  // beat these is a sign the selection (or the fitted model itself) isn't
  // adding anything over a naive forecast.
  let persistenceR2 = null, persistenceDriftR2 = null;
  try {
    const persistenceFit = Regression.fitPersistence(trainXs, trainYs);
    persistenceR2 = Regression.computeR2(testXs, testYs, persistenceFit.predict);
  } catch (e) { /* leave null — shouldn't happen with >=1 training point, but don't crash the run over it */ }
  try {
    const driftFit = Regression.fitPersistenceDrift(trainXs, trainYs);
    persistenceDriftR2 = Regression.computeR2(testXs, testYs, driftFit.predict);
  } catch (e) { /* leave null — e.g. every training X identical */ }

  return {
    title: group.title, n, fitLen: selection.fitLen, checkLen: selection.checkLen, testLen,
    model: selection.name, fitR2: selection.fitR2, checkR2: selection.checkR2, testR2, testR2FullVariance,
    persistenceR2, persistenceDriftR2,
    beatsPersistence: persistenceR2 === null ? null : testR2 > Math.max(persistenceR2, persistenceDriftR2 ?? -Infinity),
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
  const averageTestR2 = testR2s.length ? average(testR2s) : null;
  const trimmedAverageTestR2 = trimmedAverage(testR2s);

  const testR2FullVariances = evaluated.map(r => r.testR2FullVariance);
  const medianTestR2FullVariance = testR2FullVariances.length ? median(testR2FullVariances) : null;
  const averageTestR2FullVariance = testR2FullVariances.length ? average(testR2FullVariances) : null;
  const trimmedAverageTestR2FullVariance = trimmedAverage(testR2FullVariances);

  const withPersistence = evaluated.filter(r => r.persistenceR2 !== null);
  const persistenceR2s = withPersistence.map(r => r.persistenceR2);
  const persistenceDriftR2s = withPersistence.filter(r => r.persistenceDriftR2 !== null).map(r => r.persistenceDriftR2);
  const medianPersistenceR2 = persistenceR2s.length ? median(persistenceR2s) : null;
  const medianPersistenceDriftR2 = persistenceDriftR2s.length ? median(persistenceDriftR2s) : null;
  const beatCount = withPersistence.filter(r => r.beatsPersistence).length;

  const lines = [];
  lines.push(`Backtest of ${path.basename(filePath)}`);
  lines.push('Each dataset: fit on the first 50% (FIT half), model type chosen by how well it');
  lines.push('extrapolates onto the next 20% (CHECK half), then re-fit on that full 70% and');
  lines.push('scored against the held-out last 30% (TEST).');
  lines.push(`${evaluated.length} of ${groups.length} datasets evaluated (${skipped.length} skipped — too few rows)`);
  lines.push('');
  lines.push(`MEDIAN TEST R² ACROSS ALL DATASETS (Auto-selection): ${medianTestR2 === null ? 'n/a' : medianTestR2.toFixed(4)}`);
  lines.push(`AVERAGE TEST R² ACROSS ALL DATASETS (Auto-selection): ${averageTestR2 === null ? 'n/a' : averageTestR2.toFixed(4)}`);
  lines.push(`TRIMMED AVERAGE TEST R² (excludes double-MAD outliers from the median above): ${trimmedAverageTestR2.value === null ? 'n/a' : trimmedAverageTestR2.value.toFixed(4)} (${trimmedAverageTestR2.kept}/${trimmedAverageTestR2.total} datasets kept)`);
  lines.push('');
  lines.push('Same held-out predictions, scored with computeR2FullVariance instead — the');
  lines.push('R² denominator comes from the FULL dataset\'s variance, not just the test');
  lines.push('slice\'s own (which can be a near-zero, unstable baseline on its own, e.g. the');
  lines.push('tail of an already-plateaued curve):');
  lines.push(`  MEDIAN TEST R² (full-variance denominator): ${medianTestR2FullVariance === null ? 'n/a' : medianTestR2FullVariance.toFixed(4)}`);
  lines.push(`  AVERAGE TEST R² (full-variance denominator): ${averageTestR2FullVariance === null ? 'n/a' : averageTestR2FullVariance.toFixed(4)}`);
  lines.push(`  TRIMMED AVERAGE TEST R² (full-variance denominator, excludes double-MAD outliers): ${trimmedAverageTestR2FullVariance.value === null ? 'n/a' : trimmedAverageTestR2FullVariance.value.toFixed(4)} (${trimmedAverageTestR2FullVariance.kept}/${trimmedAverageTestR2FullVariance.total} datasets kept)`);
  lines.push('');
  lines.push('Persistence baselines (standard time-series sanity check — do nothing but');
  lines.push('carry the last training value forward, or continue its trend, and see how');
  lines.push('that scores on the same held-out test set):');
  lines.push(`  MEDIAN TEST R², flat persistence (last value):     ${medianPersistenceR2 === null ? 'n/a' : medianPersistenceR2.toFixed(4)}`);
  lines.push(`  MEDIAN TEST R², drift persistence (trend continued): ${medianPersistenceDriftR2 === null ? 'n/a' : medianPersistenceDriftR2.toFixed(4)}`);
  lines.push(`  Auto-selection beat BOTH persistence baselines on ${beatCount} of ${withPersistence.length} datasets`);
  lines.push('');
  lines.push('Per-dataset results:');
  evaluated
    .slice()
    .sort((a, b) => b.testR2 - a.testR2)
    .forEach(r => {
      lines.push(`  ${r.title}`);
      lines.push(`    rows: ${r.n} (fit ${r.fitLen} / check ${r.checkLen} / test ${r.testLen})   model: ${r.model}`);
      lines.push(`    fit R²: ${r.fitR2.toFixed(4)}   check (extrapolation) R²: ${r.checkR2.toFixed(4)}   test R²: ${r.testR2.toFixed(4)}   test R² (full-variance): ${r.testR2FullVariance.toFixed(4)}`);
      const persistStr = r.persistenceR2 === null ? 'n/a' : r.persistenceR2.toFixed(4);
      const driftStr = r.persistenceDriftR2 === null ? 'n/a' : r.persistenceDriftR2.toFixed(4);
      const flag = r.beatsPersistence === false ? '  <-- LOSES to a persistence baseline' : '';
      lines.push(`    persistence (flat) R²: ${persistStr}   persistence (drift) R²: ${driftStr}${flag}`);
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
