// Checks the two remaining weight-capable fit types not yet validated for
// a robust-refit gate: fitMultiLinear (multi-variable OLS) and
// fitMultiExponential (multi-variable exponential, log-linear in multiple
// predictors). Same synthetic ground-truth methodology as the
// single-variable tests: known true coefficients, five scenarios
// (Gaussian control, pure Laplace, pure Student-t, symmetric/skewed severe
// contamination), parameter-recovery error, bootstrap CI. This is a FIRST
// PASS (does an effect exist at all) — not yet a full threshold-optimization
// sweep like polynomial/log-linear got, since that's only worth doing if
// this first pass finds something.
const Regression = require('./regression.js');

function mulberry32(seed) {
  let state = seed >>> 0;
  return function () {
    state |= 0;
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function randLaplace(r, b) { const u = r() - 0.5; return -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u)); }
function randStudentT(r, df) { const z = randNormal(r); let c = 0; for (let i = 0; i < df; i++) { const n = randNormal(r); c += n * n; } return z / Math.sqrt(c / df); }
function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function bootstrapMedianCI(diffs, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = diffs.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) samples.push(median(Array.from({ length: n }, () => diffs[Math.floor(random() * n)])));
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

// Generic IRLS for fitMultiLinear — identical pattern to fitRobustPolynomial,
// just reusing fitMultiLinear's own `weights` parameter instead of
// fitPolynomial's.
function fitRobustMultiLinear(X, y, distName, nu, iterations = 20) {
  let fit = Regression.fitMultiLinear(X, y);
  let sigma2 = null;
  if (distName === 'Student-t') {
    const r0 = X.map((row, i) => y[i] - fit.predict(row));
    sigma2 = r0.reduce((s, v) => s + v * v, 0) / X.length;
    if (sigma2 <= 0) return fit;
  }
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = X.map((row, i) => y[i] - fit.predict(row));
    let weights;
    if (distName === 'Laplace') {
      const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / X.length * 0.05);
      weights = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    } else {
      weights = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    }
    let next;
    try { next = Regression.fitMultiLinear(X, y, weights); } catch (e) { break; }
    fit = next;
    if (distName === 'Student-t') {
      const newResiduals = X.map((row, i) => y[i] - fit.predict(row));
      const wSum = weights.reduce((s, v) => s + v, 0);
      const newSigma2 = newResiduals.reduce((s, r, i) => s + weights[i] * r * r, 0) / wSum;
      if (newSigma2 > 0) sigma2 = newSigma2;
    }
  }
  return fit;
}

// Same for fitMultiExponential — detect/reweight in ln(y) space (its own
// natural fitting space, same lesson as the single-variable log-linear fix).
function fitRobustMultiExponential(X, y, distName, nu, iterations = 20) {
  const lineFit = fitRobustMultiLinear(X, y.map(Math.log), distName, nu, iterations);
  const a = Math.exp(lineFit.intercept);
  const coeffs = lineFit.coeffs;
  return { a, coeffs, predict: xRow => a * Math.exp(xRow.reduce((s, v, j) => s + v * coeffs[j], 0)) };
}

function multiLinearParamError(fit, trueIntercept, trueCoeffs) {
  let e = (fit.intercept - trueIntercept) ** 2;
  trueCoeffs.forEach((c, j) => { e += (fit.coeffs[j] - c) ** 2; });
  return e;
}
function multiExpParamError(fit, trueA, trueCoeffs) {
  let e = (fit.a - trueA) ** 2;
  trueCoeffs.forEach((c, j) => { e += (fit.coeffs[j] - c) ** 2; });
  return e;
}

const N = 60;
const X2 = Array.from({ length: N }, (_, i) => [i + 1, Math.sin(i / 5) * 10 + 20]); // two predictors, one trending, one oscillating+positive (for exponential's own positivity needs elsewhere)

function runMultiLinear() {
  const trueIntercept = 3, trueCoeffs = [2, -0.5];
  const trueY = X2.map(row => trueIntercept + trueCoeffs[0] * row[0] + trueCoeffs[1] * row[1]);
  const scenarios = {
    'Gaussian (control)': rand => trueY.map(v => v + randNormal(rand) * 4),
    'Pure Laplace': rand => trueY.map(v => v + randLaplace(rand, 3)),
    'Pure Student-t df=3': rand => trueY.map(v => v + randStudentT(rand, 3) * 2.3),
    'Symmetric severe contamination (10%)': rand => trueY.map(v => { const out = rand() < 0.1; return v + randNormal(rand) * (out ? 40 : 2); }),
    'Skewed severe contamination (10%)': rand => trueY.map(v => { const out = rand() < 0.1; return v + (out ? (20 + rand() * 40) : randNormal(rand) * 2); }),
  };
  console.log(`\n===== fitMultiLinear (true intercept=${trueIntercept}, coeffs=${trueCoeffs}) =====`);
  for (const [label, gen] of Object.entries(scenarios)) {
    const diffs = [];
    let flagged = 0, n = 0;
    for (let t = 0; t < 400; t++) {
      const rand = mulberry32(1000 + t * 7919);
      const y = gen(rand);
      let olsFit;
      try { olsFit = Regression.fitMultiLinear(X2, y); } catch (e) { continue; }
      const residuals = X2.map((row, i) => y[i] - olsFit.predict(row));
      const errDist = Regression.chooseErrorDistribution(residuals);
      if (!errDist) continue;
      const nonGaussian = errDist.best.name !== 'Gaussian';
      if (nonGaussian) flagged++;
      let refit = olsFit;
      if (nonGaussian) refit = fitRobustMultiLinear(X2, y, errDist.best.name, errDist.best.params && errDist.best.params.nu) || olsFit;
      diffs.push(multiLinearParamError(refit, trueIntercept, trueCoeffs) - multiLinearParamError(olsFit, trueIntercept, trueCoeffs));
      n++;
    }
    const ci = bootstrapMedianCI(diffs);
    const verdict = ci.upper < 0 ? 'refit HELPS' : ci.lower > 0 ? 'refit HURTS' : 'no reliable effect';
    console.log(`  ${label.padEnd(36)} flagged ${flagged}/${n}  median diff=${median(diffs).toFixed(4)}  CI[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]  -> ${verdict}`);
  }
}

function runMultiExponential() {
  const trueA = 5, trueCoeffs = [0.02, -0.01];
  const trueY = X2.map(row => trueA * Math.exp(trueCoeffs[0] * row[0] + trueCoeffs[1] * row[1]));
  const scenarios = {
    'Gaussian (control)': rand => trueY.map(v => Math.max(1e-6, v + randNormal(rand) * (0.03 * v + 1))),
    'Pure Laplace': rand => trueY.map(v => Math.max(1e-6, v + randLaplace(rand, 0.02 * v + 1))),
    'Pure Student-t df=3': rand => trueY.map(v => Math.max(1e-6, v + randStudentT(rand, 3) * (0.02 * v + 1))),
    'Symmetric severe contamination (10%)': rand => trueY.map(v => { const sd = 0.03 * v + 1; const out = rand() < 0.1; return Math.max(1e-6, v + randNormal(rand) * (out ? sd * 15 : sd)); }),
    'Skewed severe contamination (10%)': rand => trueY.map(v => { const sd = 0.03 * v + 1; const out = rand() < 0.1; return Math.max(1e-6, v + (out ? (sd * 15 + rand() * sd * 15) : randNormal(rand) * sd)); }),
  };
  console.log(`\n===== fitMultiExponential (true a=${trueA}, coeffs=${trueCoeffs}) =====`);
  for (const [label, gen] of Object.entries(scenarios)) {
    const diffs = [];
    let flagged = 0, n = 0;
    for (let t = 0; t < 400; t++) {
      const rand = mulberry32(2000 + t * 7919);
      const y = gen(rand);
      let olsFit;
      try { olsFit = Regression.fitMultiExponential(X2, y); } catch (e) { continue; }
      const tResiduals = X2.map((row, i) => Math.log(y[i]) - Math.log(olsFit.predict(row)));
      const errDist = Regression.chooseErrorDistribution(tResiduals);
      if (!errDist) continue;
      const nonGaussian = errDist.best.name !== 'Gaussian';
      if (nonGaussian) flagged++;
      let refit = olsFit;
      if (nonGaussian) refit = fitRobustMultiExponential(X2, y, errDist.best.name, errDist.best.params && errDist.best.params.nu) || olsFit;
      diffs.push(multiExpParamError(refit, trueA, trueCoeffs) - multiExpParamError(olsFit, trueA, trueCoeffs));
      n++;
    }
    const ci = bootstrapMedianCI(diffs);
    const verdict = ci.upper < 0 ? 'refit HELPS' : ci.lower > 0 ? 'refit HURTS' : 'no reliable effect';
    console.log(`  ${label.padEnd(36)} flagged ${flagged}/${n}  median diff=${median(diffs).toFixed(4)}  CI[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]  -> ${verdict}`);
  }
}

runMultiLinear();
runMultiExponential();
