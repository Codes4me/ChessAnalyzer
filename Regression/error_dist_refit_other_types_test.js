// Does the robust-refit gate wired into the plain polynomial/linear branch
// (regression.js: maybeRobustRefit, ΔBIC>=45) generalize to the OTHER
// weight-capable fit types — fitProportional, fitExponential, fitPower,
// fitLogarithmic — or was that result specific to a straight line?
// Same synthetic ground-truth methodology as error_dist_refit_synthetic_test.js:
// known true (a, b), generate data with various error structures, compare
// plain OLS-style fit vs. a generic IRLS refit (reusing each fit function's
// existing `weights` parameter, same trick fitRobustPolynomial already
// uses) by parameter-recovery error, with a bootstrap CI.
//
// One real wrinkle: fitExponential/fitPower/fitLogarithmic all fit via a
// LOG-TRANSFORMED line internally (ln(y) = ln(a) + b*x, etc.) and pass
// `weights` straight into THAT transformed fit — not the original (x, y)
// scale. This test computes the detected error distribution and IRLS
// weights from the ORIGINAL scale residuals (what the user actually sees)
// and feeds them into the same `weights` parameter regardless — consistent
// with how this project already treats user-supplied weights for these fit
// types (same mismatch would exist there too), not a new issue introduced
// here, but worth flagging: it means the weighting isn't a mathematically
// pure IRLS step for these transformed fits the way it is for fitPolynomial.
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

// Generic IRLS refit for any fitFn(xs, ys, weights) -> {a, b, predict}.
function fitRobustWeighted(fitFn, xs, ys, distName, nu, iterations = 20) {
  let fit = fitFn(xs, ys);
  let sigma2 = null;
  if (distName === 'Student-t') {
    const r0 = xs.map((x, i) => ys[i] - fit.predict(x));
    sigma2 = r0.reduce((s, v) => s + v * v, 0) / xs.length;
    if (sigma2 <= 0) return fit;
  }
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    let weights;
    if (distName === 'Laplace') {
      const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / xs.length * 0.05);
      weights = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    } else {
      weights = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    }
    let next;
    try { next = fitFn(xs, ys, weights); } catch (e) { break; }
    fit = next;
    if (distName === 'Student-t') {
      const newResiduals = xs.map((x, i) => ys[i] - fit.predict(x));
      const wSum = weights.reduce((s, v) => s + v, 0);
      const newSigma2 = newResiduals.reduce((s, r, i) => s + weights[i] * r * r, 0) / wSum;
      if (newSigma2 > 0) sigma2 = newSigma2;
    }
  }
  return fit;
}

function paramError(fit, trueA, trueB) { return (fit.a - trueA) ** 2 + (fit.b - trueB) ** 2; }

function runModelFamily(familyName, fitFn, xs, genYs, trueA, trueB, trials = 400, seedBase = 1000) {
  const scenarios = [
    ['Gaussian (control)', rand => genYs(rand, { mode: 'gaussian' })],
    ['Pure Laplace', rand => genYs(rand, { mode: 'laplace' })],
    ['Pure Student-t df=3', rand => genYs(rand, { mode: 't3' })],
    ['Symmetric severe contamination (10%)', rand => genYs(rand, { mode: 'contam', frac: 0.1, skew: false })],
    ['Skewed severe contamination (10%)', rand => genYs(rand, { mode: 'contam', frac: 0.1, skew: true })],
  ];
  console.log(`\n===== ${familyName} (true a=${trueA}, b=${trueB}) =====`);
  for (const [label, gen] of scenarios) {
    const diffs = [];
    let flaggedCount = 0;
    const names = {};
    let n = 0;
    for (let t = 0; t < trials; t++) {
      const rand = mulberry32(seedBase + t * 7919);
      const ys = gen(rand);
      let olsFit;
      try { olsFit = fitFn(xs, ys); } catch (e) { continue; }
      const residuals = xs.map((x, i) => ys[i] - olsFit.predict(x));
      const errDist = Regression.chooseErrorDistribution(residuals);
      if (!errDist) continue;
      const nonGaussian = errDist.best.name !== 'Gaussian';
      if (nonGaussian) { flaggedCount++; names[errDist.best.name] = (names[errDist.best.name] || 0) + 1; }
      let refit = olsFit;
      if (nonGaussian) refit = fitRobustWeighted(fitFn, xs, ys, errDist.best.name, errDist.best.params && errDist.best.params.nu) || olsFit;
      diffs.push(paramError(refit, trueA, trueB) - paramError(olsFit, trueA, trueB));
      n++;
    }
    if (!diffs.length) { console.log(`  ${label.padEnd(36)} (no valid trials)`); continue; }
    const ci = bootstrapMedianCI(diffs);
    const verdict = ci.upper < 0 ? 'refit HELPS' : ci.lower > 0 ? 'refit HURTS' : 'no reliable effect';
    console.log(`  ${label.padEnd(36)} flagged ${flaggedCount}/${n} ${JSON.stringify(names)}  median diff=${median(diffs).toFixed(4)}  CI[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]  -> ${verdict}`);
  }
}

const N = 60;
const xsPositive = Array.from({ length: N }, (_, i) => i + 1); // needed for exponential/power (x>0)

function genNoisy(rand, trueFn, opts) {
  return xsPositive.map(x => {
    const base = trueFn(x);
    if (opts.mode === 'gaussian') return base + randNormal(rand) * (0.03 * Math.abs(base) + 1);
    if (opts.mode === 'laplace') return base + randLaplace(rand, 0.02 * Math.abs(base) + 1);
    if (opts.mode === 't3') return base + randStudentT(rand, 3) * (0.02 * Math.abs(base) + 1);
    // contamination
    const sd = 0.03 * Math.abs(base) + 1;
    const isOutlier = rand() < opts.frac;
    if (!isOutlier) return base + randNormal(rand) * sd;
    const outMag = sd * 15;
    return base + (opts.skew ? (outMag + rand() * outMag) : randNormal(rand) * outMag);
  });
}

// --- Exponential: y = a*e^(bx) ---
runModelFamily(
  'Exponential (y = a*e^(bx))',
  Regression.fitExponential,
  xsPositive,
  (rand, opts) => genNoisy(rand, x => 5 * Math.exp(0.03 * x), opts),
  5, 0.03
);

// --- Power: y = a*x^b ---
runModelFamily(
  'Power (y = a*x^b)',
  Regression.fitPower,
  xsPositive,
  (rand, opts) => genNoisy(rand, x => 2 * Math.pow(x, 1.4), opts),
  2, 1.4
);

// --- Logarithmic: y = a*ln(x) + b ---
runModelFamily(
  'Logarithmic (y = a*ln(x) + b)',
  Regression.fitLogarithmic,
  xsPositive,
  (rand, opts) => genNoisy(rand, x => 4 * Math.log(x) + 10, opts),
  4, 10
);

// --- Proportional: y = a*x ---
runModelFamily(
  'Proportional (y = a*x, no intercept — b unused, set to 0)',
  (xs, ys, w) => { const f = Regression.fitProportional(xs, ys, w); return { a: f.a, b: 0, predict: f.predict }; },
  xsPositive,
  (rand, opts) => genNoisy(rand, x => 2.5 * x, opts),
  2.5, 0
);
