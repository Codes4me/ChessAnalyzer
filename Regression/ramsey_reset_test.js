// Does Ramsey RESET help model selection on this project's own corpus?
// Same two-part validation this project runs on every new diagnostic before
// adopting it (see autocorrelation_sensitivity_test.js, size_adjust_null_
// calibration_check.js): (1) is it correctly calibrated on data where the
// null is TRUE by construction, and (2) does it actually catch something
// real on this corpus -- here, "real" means a selected candidate that goes
// on to underperform a persistence baseline out of sample.
//
// RESET test (Ramsey 1969): fit y ~ X, take the fitted values yhat, add
// yhat^2 and yhat^3 as extra regressors, refit, and F-test whether those
// two added terms are jointly zero. H0: the model's functional form is
// correctly specified (no omitted nonlinearity). Rejecting H0 means the
// fitted values themselves have predictive power beyond the model's own
// functional form -- a sign something nonlinear got left out.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { selectByExtrapolation } = require('./model_candidates');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

// --- F-distribution CDF, via the regularized incomplete beta function
// (same log-gamma/continued-fraction machinery as regression.js's own
// Student-t CDF, just not exported from there, so reimplemented here --
// same convention as the other one-off scripts in this folder). ---
function logGamma(x) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function betaContinuedFraction(x, a, b) {
  const MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? bt * betaContinuedFraction(x, a, b) / a
    : 1 - bt * betaContinuedFraction(1 - x, b, a) / b;
}
function fDistCDF(f, d1, d2) {
  if (f <= 0) return 0;
  const x = (d1 * f) / (d1 * f + d2);
  return regularizedIncompleteBeta(x, d1 / 2, d2 / 2);
}

// --- Ramsey RESET, general form: works on ANY fitted candidate as long as
// we have (x, y, predict). Restricted model = the candidate's own fit
// (SSE from its residuals). Unrestricted model = OLS of y on [x, yhat^2,
// yhat^3] (x kept in, same convention as the textbook RESET auxiliary
// regression, so the test is about adding power terms of yhat ON TOP OF
// the existing specification, not replacing it). ---
function ramseyReset(xs, ys, predict, { powers = [2, 3] } = {}) {
  const n = xs.length;
  const yhat = xs.map(x => predict(x));
  const residuals = ys.map((y, i) => y - yhat[i]);
  const sseRestricted = residuals.reduce((s, r) => s + r * r, 0);

  const X = xs.map((x, i) => [x, ...powers.map(p => yhat[i] ** p)]);
  let unrestricted;
  try { unrestricted = Regression.fitMultiLinear(X, ys); } catch (e) { return null; }
  const ssTotal = ys.reduce((s, y) => { const m = ys.reduce((a, b) => a + b, 0) / n; return s + (y - m) ** 2; }, 0);
  const sseUnrestricted = ssTotal * (1 - unrestricted.r2);

  const q = powers.length; // number of restrictions
  const kUnrestricted = 1 + powers.length + 1; // x, power terms, intercept
  const df2 = n - kUnrestricted;
  if (df2 < 1 || sseUnrestricted <= 0) return null;
  const F = ((sseRestricted - sseUnrestricted) / q) / (sseUnrestricted / df2);
  if (!Number.isFinite(F) || F < 0) return { F: 0, pValue: 1 };
  const pValue = 1 - fDistCDF(F, q, df2);
  return { F, pValue };
}

// =====================================================================
// PART 1: null calibration -- correctly-specified synthetic data (plain
// linear + Gaussian noise, same convention used by every other null-
// calibration check in this project), across a range of n. If RESET is
// well-calibrated, the false-positive rate at p<0.05 should sit near 5%
// (some drift is expected/tolerable -- this is the same bar the other
// checks use) at every n, with no collapse or inflation as n grows (that
// collapse/inflation pattern is exactly what killed the old logistic-
// offset size-adjustment, per regression.js's SIZE_ADJUST history).
// =====================================================================
function checkNullCalibration() {
  console.log('=== Part 1: null calibration on genuinely correctly-specified synthetic data ===\n');
  console.log('  n      false-positive rate at p<0.05   (target: ~5%, flat across n)');
  const SIZES = [15, 30, 75, 200, 500, 1300];
  for (const n of SIZES) {
    const xs = Array.from({ length: n }, (_, i) => i + 1);
    let flagged = 0, total = 0;
    for (let t = 0; t < 1000; t++) {
      const rand = mulberry32(800000 + n * 7919 + t);
      const ys = xs.map(x => 3 + 2 * x + randNormal(rand) * 4);
      let fit; try { fit = Regression.runRegression(xs, ys, { degree: 1 }); } catch (e) { continue; }
      const reset = ramseyReset(xs, ys, fit.predict);
      if (!reset) continue;
      total++;
      if (reset.pValue < 0.05) flagged++;
    }
    console.log(`  ${n.toString().padStart(5)}  ${(100 * flagged / total).toFixed(1)}%  (${total} converged)`);
  }
  console.log('');
}
checkNullCalibration();

// =====================================================================
// PART 2: real-corpus relevance -- same method as autocorrelation_
// sensitivity_test.js. For each real dataset: run the project's own
// model selector, fit RESET on the SELECTED candidate's residuals, then
// separately compute the REAL held-out backtest edge (test R² minus the
// best persistence baseline, 70/30 split) and correlate RESET's p-value
// against it. If RESET is catching real missing nonlinearity -- not just
// noise -- low p-values (rejecting "correctly specified") should line up
// with datasets whose selected model later struggled to extrapolate.
// =====================================================================
function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  if (n < 20) return { skipped: true, reason: 'too few rows' };

  let selectionFull; try { selectionFull = selectByExtrapolation(xs, ys, { fitFrac: 0.5, checkFrac: 0.2 }); } catch (e) { return { skipped: true, reason: e.message }; }
  let fullFit; try { fullFit = selectionFull.candidate.fit(xs, ys); } catch (e) { return { skipped: true, reason: `refit failed: ${e.message}` }; }
  const reset = ramseyReset(xs, ys, fullFit.predict);
  if (!reset) return { skipped: true, reason: 'RESET auxiliary regression failed' };

  const trainPoolLen = Math.floor(n * 0.7);
  const testLen = n - trainPoolLen;
  if (testLen < 2) return { skipped: true, reason: 'too few rows for a held-out backtest' };
  const trainXs = xs.slice(0, trainPoolLen), trainYs = ys.slice(0, trainPoolLen);
  const testXs = xs.slice(trainPoolLen), testYs = ys.slice(trainPoolLen);
  let selection, finalFit;
  try { selection = selectByExtrapolation(trainXs, trainYs, { fitFrac: 0.5, checkFrac: 0.2 }); finalFit = selection.candidate.fit(trainXs, trainYs); }
  catch (e) { return { skipped: true, reason: `backtest fit failed: ${e.message}` }; }
  const testR2 = Regression.computeR2(testXs, testYs, finalFit.predict);
  let persistenceR2 = -Infinity, persistenceDriftR2 = -Infinity;
  try { persistenceR2 = Regression.computeR2(testXs, testYs, Regression.fitPersistence(trainXs, trainYs).predict); } catch (e) {}
  try { persistenceDriftR2 = Regression.computeR2(testXs, testYs, Regression.fitPersistenceDrift(trainXs, trainYs).predict); } catch (e) {}
  const edge = testR2 - Math.max(persistenceR2, persistenceDriftR2);

  return { skipped: false, title: group.title, n, resetP: reset.pValue, edge, candidateType: selectionFull.candidate.name };
}

function rankArray(arr) {
  const idx = arr.map((_, i) => i).sort((a, b) => arr[a] - arr[b]);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < idx.length) { let j = i; while (j + 1 < idx.length && arr[idx[j + 1]] === arr[idx[i]]) j++; const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) ranks[idx[k]] = avg; i = j + 1; }
  return ranks;
}
function pearson(xs, ys) { const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; } return sxx === 0 || syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy); }
function spearman(xs, ys) { return pearson(rankArray(xs), rankArray(ys)); }
function median(nums) { const sorted = [...nums].sort((a, b) => a - b); const mid = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2; }

console.log('=== Part 2: does RESET\'s p-value line up with real held-out forecast failure? ===\n');
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));
const results = groups.map(evaluateDataset);
const evaluated = results.filter(r => !r.skipped);
const skipped = results.filter(r => r.skipped);
console.log(`Evaluated ${evaluated.length} of ${groups.length} datasets (${skipped.length} skipped).\n`);

evaluated.slice().sort((a, b) => a.edge - b.edge).forEach(r => {
  console.log(`  ${r.title}  n=${r.n}  candidate=${r.candidateType}  edge=${r.edge.toFixed(4)}  RESET p=${r.resetP.toFixed(4)}`);
});

const edges = evaluated.map(r => r.edge);
const resetPs = evaluated.map(r => r.resetP);
const scores = resetPs.map(p => -Math.log10(Math.max(p, 1e-300)));
const corr = spearman(scores, edges);
console.log(`\nSpearman correlation (RESET significance vs. real-world edge over persistence): ${corr.toFixed(3)}`);
console.log('(negative = more significant RESET (lower p) lines up with WORSE real extrapolation, as expected if RESET is catching something real; this number does not depend on the threshold below)');

// --- Threshold comparison: the arbitrary textbook p<0.05 vs. p*=0.11468,
// this project's own DW density-crossover tipping point (regression.js's
// DW_P_STAR) -- reusing an already-validated, corpus-specific threshold
// instead of an arbitrary round number, same logic as borrowing a crossover
// threshold anywhere else in this project. ---
const THRESHOLDS = [
  { label: 'p<0.05 (textbook)', cutoff: 0.05 },
  { label: `p<${Regression.DW_P_STAR} (DW_P_STAR, this project's own tipping point)`, cutoff: Regression.DW_P_STAR },
];
for (const { label, cutoff } of THRESHOLDS) {
  const flagged = evaluated.filter(r => r.resetP < cutoff).map(r => r.edge);
  const notFlagged = evaluated.filter(r => r.resetP >= cutoff).map(r => r.edge);
  console.log(`\n--- Threshold: ${label} ---`);
  console.log(`flagged:      ${flagged.length} datasets, median edge = ${flagged.length ? median(flagged).toFixed(4) : 'n/a'}`);
  console.log(`not flagged:  ${notFlagged.length} datasets, median edge = ${notFlagged.length ? median(notFlagged).toFixed(4) : 'n/a'}`);
}

if (skipped.length) {
  console.log('\nSkipped:');
  skipped.forEach(r => console.log(`  ${r.title || '(untitled)'} — ${r.reason}`));
}

// =====================================================================
// PART 3: would RESET actually change model selection for the better?
// For every dataset where the selector's TOP candidate gets RESET-flagged,
// check whether any OTHER candidate the selector considered but didn't
// pick (a) is NOT flagged by RESET and (b) would have scored higher
// held-out R² on the same 70/30 split. If so, RESET flags would have been
// actionable, not just descriptive. Run once per threshold so the
// textbook p<0.05 and this project's own DW_P_STAR can be compared
// directly on the same 66 datasets.
// =====================================================================
const { CANDIDATES } = require('./model_candidates');
function evaluateSwitchValue(group, cutoff) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  if (n < 20) return null;
  const trainPoolLen = Math.floor(n * 0.7);
  const trainXs = xs.slice(0, trainPoolLen), trainYs = ys.slice(0, trainPoolLen);
  const testXs = xs.slice(trainPoolLen), testYs = ys.slice(trainPoolLen);
  let selection; try { selection = selectByExtrapolation(trainXs, trainYs, { fitFrac: 0.5, checkFrac: 0.2 }); } catch (e) { return null; }
  let topFit; try { topFit = selection.candidate.fit(trainXs, trainYs); } catch (e) { return null; }
  const topReset = ramseyReset(trainXs, trainYs, topFit.predict);
  if (!topReset || topReset.pValue >= cutoff) return null; // top candidate wasn't flagged -- nothing to switch away from
  const topTestR2 = Regression.computeR2(testXs, testYs, topFit.predict);

  let bestAlt = null;
  for (const cand of CANDIDATES) {
    if (cand.name === selection.candidate.name) continue;
    let fit; try { fit = cand.fit(trainXs, trainYs); } catch (e) { continue; }
    const altReset = ramseyReset(trainXs, trainYs, fit.predict);
    if (!altReset || altReset.pValue < cutoff) continue; // only count alternatives RESET does NOT flag
    const altTestR2 = Regression.computeR2(testXs, testYs, fit.predict);
    if (!bestAlt || altTestR2 > bestAlt.testR2) bestAlt = { name: cand.name, testR2: altTestR2 };
  }
  return { title: group.title, n, topType: selection.candidate.name, topTestR2, bestAlt };
}

function runPart3(label, cutoff) {
  console.log(`\n=== Part 3: would switching away from RESET-flagged candidates have helped? (threshold: ${label}) ===\n`);
  const switchResults = groups.map(g => evaluateSwitchValue(g, cutoff)).filter(Boolean);
  console.log(`${switchResults.length} datasets had a RESET-flagged top candidate.\n`);
  let helped = 0, hurt = 0, noAlt = 0;
  switchResults.forEach(r => {
    if (!r.bestAlt) { noAlt++; console.log(`  ${r.title}: top=${r.topType} (R²=${r.topTestR2.toFixed(3)}, RESET-flagged) -- no unflagged alternative available`); return; }
    const delta = r.bestAlt.testR2 - r.topTestR2;
    if (delta > 0.01) helped++; else if (delta < -0.01) hurt++;
    console.log(`  ${r.title}: top=${r.topType} (R²=${r.topTestR2.toFixed(3)}, RESET-flagged) vs best unflagged alt=${r.bestAlt.name} (R²=${r.bestAlt.testR2.toFixed(3)})  delta=${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`);
  });
  console.log(`\nSwitching away from a RESET-flagged top candidate would have HELPED (>0.01 R² gain) on ${helped}, HURT on ${hurt}, no eligible alternative on ${noAlt}.`);
  return { helped, hurt, noAlt, total: switchResults.length };
}

const part3Results = THRESHOLDS.map(({ label, cutoff }) => ({ label, cutoff, ...runPart3(label, cutoff) }));
console.log('\n=== Part 3 summary across thresholds ===');
part3Results.forEach(r => console.log(`  ${r.label}: ${r.total} flagged, helped=${r.helped} hurt=${r.hurt} noAlt=${r.noAlt}`));
