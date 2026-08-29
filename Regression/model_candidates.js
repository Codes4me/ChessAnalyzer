// The list of model types the "Auto (best by extrapolation check)" option
// tries, plus the selection logic itself — shared between backtest_models.js
// (the offline CLI script) and the /regression page, so both pick models
// exactly the same way.

// Named RegressionLib, not Regression — classic (non-module) <script src>
// tags share one global lexical scope in a browser, so a top-level `const
// Regression` here would collide with regression.js's own top-level `const
// Regression` and throw a duplicate-declaration SyntaxError on the page.
const RegressionLib = (typeof module !== 'undefined' && module.exports) ? require('./regression') : window.Regression;

// `minFit` is the fewest FIT-half points a candidate needs to even be
// attempted. `params` is roughly how many fitted coefficients the model
// has — used below to break near-ties in favor of the simpler shape (e.g.
// a degree-4 curve that only barely out-extrapolates a degree-2 one is
// probably just fitting noise, not a real 4th-order feature of the data).
// Quartic/quintic are included deliberately rather than banned outright —
// earlier experiments showed adjusted R² selection alone lets them win on
// training data and then blow up on unseen data; the fit/check
// extrapolation test below is what actually keeps them honest.
const CANDIDATES = [
  { name: 'linear (degree 1)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 1 }) },
  { name: 'quadratic (degree 2)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 2 }) },
  { name: 'cubic (degree 3)', params: 4, minFit: 4, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 3 }) },
  { name: 'quartic (degree 4)', params: 5, minFit: 5, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 4 }) },
  { name: 'quintic (degree 5)', params: 6, minFit: 6, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 5 }) },
  { name: 'S-curve (logistic)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.fitLogisticCurve(xs, ys) },
  { name: 'piecewise (constant+linear)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.fitPiecewiseConstantLinear(xs, ys) },
  { name: 'exponential', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitExponential(xs, ys) },
  { name: 'power', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitPower(xs, ys) },
  { name: 'logarithmic', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitLogarithmic(xs, ys) },
  { name: 'reciprocal', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitReciprocal(xs, ys) },
  { name: 'sinusoidal', params: 4, minFit: 4, fit: (xs, ys) => RegressionLib.fitSinusoidal(xs, ys) },
];

// Splits (xs, ys) into a FIT slice (the first `fitFrac` of the data) and a
// CHECK slice right after it (the next `checkFrac`), fits every candidate on
// FIT alone, and scores each one on CHECK — data just past what it was fit
// on. That score is itself the extrapolation penalty: a model that overfits
// FIT predicts CHECK badly and simply loses, no separate complexity formula
// needed. Returns the winning candidate plus its fit/check R² — NOT a
// ready-to-use model, since the caller decides what to re-fit it on
// (backtest_models.js re-fits on the fit+check data only, to leave a true
// final test set untouched; the /regression page re-fits on all the data,
// since there's no held-out set to protect there — just the best equation
// to hand back).
function selectByExtrapolation(xs, ys, { fitFrac = 0.5, checkFrac = 0.2, tolerance = 0.02 } = {}) {
  const n = xs.length;
  const fitLen = Math.floor(n * fitFrac);
  const checkEnd = Math.floor(n * (fitFrac + checkFrac));
  const checkLen = checkEnd - fitLen;

  if (fitLen < 2 || checkLen < 2) {
    throw new Error(`Need more data for a fit/check split — got ${n} rows.`);
  }

  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen, checkEnd), checkYs = ys.slice(fitLen, checkEnd);

  const results = [];
  for (const candidate of CANDIDATES) {
    if (fitLen < candidate.minFit) continue;
    try {
      const fit = candidate.fit(fitXs, fitYs);
      if (!Number.isFinite(fit.r2)) continue;
      const checkR2 = RegressionLib.computeR2(checkXs, checkYs, fit.predict);
      if (!Number.isFinite(checkR2)) continue;
      results.push({ candidate, fitR2: fit.r2, checkR2 });
    } catch (e) { /* this model type doesn't apply to this data — skip it */ }
  }

  if (!results.length) throw new Error('No candidate model could be fit to this data.');

  // Pick the model with the best extrapolation score — but if a more
  // complex model only edges out a simpler one by less than `tolerance`,
  // prefer the simpler one instead. A win that small is more likely noise
  // than a real feature the extra parameters are capturing (this is what
  // stopped a quartic curve — 0.9997 R² vs quadratic's 0.9995 on one
  // dataset — from beating out the obviously-correct parabola shape).
  const bestCheckR2 = Math.max(...results.map(r => r.checkR2));
  const contenders = results.filter(r => r.checkR2 >= bestCheckR2 - tolerance);
  const best = contenders.reduce((simplest, r) => (r.candidate.params < simplest.candidate.params ? r : simplest));

  return { name: best.candidate.name, candidate: best.candidate, fitR2: best.fitR2, checkR2: best.checkR2, fitLen, checkLen };
}

const ModelCandidates = { CANDIDATES, selectByExtrapolation };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ModelCandidates;
} else {
  window.ModelCandidates = ModelCandidates;
}
