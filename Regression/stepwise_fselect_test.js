// Does stepwise (forward/backward) predictor selection via partial F-tests
// help on this project's multi-variable regression path (regression.html's
// "Multi-variable" model types -- several X columns, picked manually right
// now, fed into Regression.fitMultiLinear)?
//
// Classic stepwise regression (as in R's step(), Stata's stepwise,
// textbook SAS/SPSS): starting from either no predictors (forward) or all
// predictors (backward), add/drop one predictor at a time based on a
// PARTIAL F-TEST comparing the model with vs. without that predictor,
// using a fixed entry/exit significance threshold (0.05 here, the
// conventional default these tools ship with).
//
// Same two-part validation this project runs on every new method before
// adopting it: (1) on SYNTHETIC data with known true/noise predictors, does
// it actually recover the right predictor set; (2) on this project's real
// multi-column datasets, does the selected subset beat using every
// predictor column, out of sample.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

// --- F-distribution CDF (same machinery as ramsey_reset_test.js --
// reimplemented here rather than exported from regression.js, matching
// this folder's existing convention of small self-contained scripts). ---
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

function sse(X, y, cols) {
  if (cols.length === 0) {
    const m = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((s, v) => s + (v - m) ** 2, 0);
  }
  const Xsub = X.map(row => cols.map(c => row[c]));
  const fit = Regression.fitMultiLinear(Xsub, y);
  const ssTotal = (() => { const m = y.reduce((a, b) => a + b, 0) / y.length; return y.reduce((s, v) => s + (v - m) ** 2, 0); })();
  return ssTotal * (1 - fit.r2);
}

// Partial F-test for adding/removing ONE predictor between a reduced
// (cols without it) and full (cols with it) model.
function partialFTest(X, y, reducedCols, fullCols) {
  const n = y.length;
  const sseReduced = sse(X, y, reducedCols);
  const sseFull = sse(X, y, fullCols);
  const dfFull = n - fullCols.length - 1;
  if (dfFull < 1) return { pValue: 1 };
  const F = Math.max(0, (sseReduced - sseFull) / 1) / (sseFull / dfFull);
  return { F, pValue: 1 - fDistCDF(F, 1, dfFull) };
}

function forwardSelect(X, y, { entryP = 0.05 } = {}) {
  const p = X[0].length;
  let included = [];
  let remaining = Array.from({ length: p }, (_, i) => i);
  while (remaining.length) {
    let best = null;
    for (const c of remaining) {
      const test = partialFTest(X, y, included, [...included, c].sort((a, b) => a - b));
      if (!best || test.pValue < best.pValue) best = { c, pValue: test.pValue };
    }
    if (best.pValue < entryP) {
      included.push(best.c);
      included.sort((a, b) => a - b);
      remaining = remaining.filter(c => c !== best.c);
    } else break;
  }
  return included;
}

function backwardEliminate(X, y, { exitP = 0.05 } = {}) {
  const p = X[0].length;
  let included = Array.from({ length: p }, (_, i) => i);
  while (included.length) {
    let worst = null;
    for (const c of included) {
      const reduced = included.filter(x => x !== c);
      const test = partialFTest(X, y, reduced, included);
      if (!worst || test.pValue > worst.pValue) worst = { c, pValue: test.pValue };
    }
    if (worst.pValue > exitP) {
      included = included.filter(c => c !== worst.c);
    } else break;
  }
  return included;
}

function stepwiseSelect(X, y, { entryP = 0.05, exitP = 0.10 } = {}) {
  const p = X[0].length;
  let included = [];
  let remaining = Array.from({ length: p }, (_, i) => i);
  let guard = 0;
  while (guard++ < 2 * p + 5) {
    // forward step
    let best = null;
    for (const c of remaining) {
      const test = partialFTest(X, y, included, [...included, c].sort((a, b) => a - b));
      if (!best || test.pValue < best.pValue) best = { c, pValue: test.pValue };
    }
    let changed = false;
    if (best && best.pValue < entryP) {
      included.push(best.c); included.sort((a, b) => a - b);
      remaining = remaining.filter(c => c !== best.c);
      changed = true;
    }
    // backward step (exit threshold looser than entry -- standard stepwise convention, avoids oscillation)
    if (included.length) {
      let worst = null;
      for (const c of included) {
        const reduced = included.filter(x => x !== c);
        const test = partialFTest(X, y, reduced, included);
        if (!worst || test.pValue > worst.pValue) worst = { c, pValue: test.pValue };
      }
      if (worst.pValue > exitP) {
        included = included.filter(c => c !== worst.c);
        remaining.push(worst.c); remaining.sort((a, b) => a - b);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return included;
}

// =====================================================================
// PART 1: synthetic validation -- known true predictors vs. known noise
// predictors, across many trials. Forward/backward/stepwise should all
// recover {true predictors} and exclude {noise predictors} most of the
// time, with a false-inclusion rate for noise columns near the entry
// threshold (0.05), not wildly above it.
// =====================================================================
function runSyntheticTrial(seed, n) {
  const rand = mulberry32(seed);
  // 5 candidate predictors: columns 0,1 are real (coeffs 3, -2), columns 2,3,4 are pure noise.
  const X = Array.from({ length: n }, () => [randNormal(rand), randNormal(rand), randNormal(rand), randNormal(rand), randNormal(rand)]);
  const y = X.map(row => 5 + 3 * row[0] - 2 * row[1] + randNormal(rand) * 2);
  const fwd = forwardSelect(X, y);
  const bwd = backwardEliminate(X, y);
  const step = stepwiseSelect(X, y);
  return { fwd, bwd, step };
}

function summarizeSynthetic(label, results) {
  const n = results.length;
  const trueIncluded0 = results.filter(r => r.includes(0)).length / n;
  const trueIncluded1 = results.filter(r => r.includes(1)).length / n;
  const falseIncluded = [2, 3, 4].map(c => results.filter(r => r.includes(c)).length / n);
  console.log(`${label}:`);
  console.log(`  true predictor 0 (coef=3) recovered: ${(100 * trueIncluded0).toFixed(1)}% of trials`);
  console.log(`  true predictor 1 (coef=-2) recovered: ${(100 * trueIncluded1).toFixed(1)}% of trials`);
  console.log(`  noise predictors 2/3/4 false-inclusion rate: ${falseIncluded.map(v => (100 * v).toFixed(1) + '%').join(' / ')}  (target: near entry threshold, 5%)`);
}

console.log('=== Part 1: synthetic validation (2 true predictors, 3 noise predictors, 200 trials) ===\n');
const TRIALS = 200;
for (const n of [30, 100]) {
  console.log(`--- n=${n} ---`);
  const fwdResults = [], bwdResults = [], stepResults = [];
  for (let t = 0; t < TRIALS; t++) {
    const r = runSyntheticTrial(100000 + n * 97 + t, n);
    fwdResults.push(r.fwd); bwdResults.push(r.bwd); stepResults.push(r.step);
  }
  summarizeSynthetic('Forward selection', fwdResults);
  summarizeSynthetic('Backward elimination', bwdResults);
  summarizeSynthetic('Stepwise (both directions)', stepResults);
  console.log('');
}

// =====================================================================
// PART 2: real multi-column datasets in this project's own corpus --
// does the selected subset beat using every predictor column, out of
// sample (70/30 split, held-out R²)?
// =====================================================================
console.log('=== Part 2: real multi-column datasets -- selected subset vs. full model, held-out R² ===\n');
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));
const multiGroups = groups.filter(g => g.columns.length > 2);
console.log(`${multiGroups.length} datasets have more than one predictor column.\n`);

function evaluateReal(group) {
  const cols = group.columns;
  const yIdx = cols.length - 1; // convention: last column is the target, matches regression.html's default
  const y = cols[yIdx];
  const n = y.length;
  const predictorIdx = cols.map((_, i) => i).filter(i => i !== yIdx);
  const X = Array.from({ length: n }, (_, row) => predictorIdx.map(i => cols[i][row]));
  if (n < predictorIdx.length * 4 + 10) return { skipped: true, reason: 'too few rows relative to predictor count' };

  const trainLen = Math.floor(n * 0.7);
  const Xtrain = X.slice(0, trainLen), ytrain = y.slice(0, trainLen);
  const Xtest = X.slice(trainLen), ytest = y.slice(trainLen);
  if (ytest.length < 3) return { skipped: true, reason: 'too few rows for a held-out test' };

  const selected = stepwiseSelect(Xtrain, ytrain);
  const allCols = predictorIdx.map((_, i) => i);

  function fitAndTestR2(cols) {
    if (!cols.length) {
      const m = ytrain.reduce((a, b) => a + b, 0) / ytrain.length;
      const predict = () => m;
      return Regression.computeR2(Xtest, ytest, () => m);
    }
    const Xsub = Xtrain.map(row => cols.map(c => row[c]));
    const fit = Regression.fitMultiLinear(Xsub, ytrain);
    const predict = xRow => fit.intercept + cols.reduce((s, c, j) => s + xRow[c] * fit.coeffs[j], 0);
    let ss = 0;
    for (let i = 0; i < ytest.length; i++) ss += (ytest[i] - predict(Xtest[i])) ** 2;
    const m = ytrain.reduce((a, b) => a + b, 0) / ytrain.length;
    const sst = ytest.reduce((s, v) => s + (v - m) ** 2, 0);
    return sst === 0 ? null : 1 - ss / sst;
  }

  const selectedR2 = fitAndTestR2(selected);
  const fullR2 = fitAndTestR2(allCols);
  return { skipped: false, title: group.title, n, nPredictors: predictorIdx.length, selectedCount: selected.length, selectedR2, fullR2 };
}

const realResults = multiGroups.map(evaluateReal);
realResults.forEach(r => {
  if (r.skipped) { console.log(`  (skipped) ${r.reason}`); return; }
  const delta = r.selectedR2 - r.fullR2;
  console.log(`  ${r.title}  n=${r.n}  predictors=${r.nPredictors}  stepwise kept ${r.selectedCount}  selected R²=${r.selectedR2.toFixed(4)}  full R²=${r.fullR2.toFixed(4)}  delta=${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`);
});
