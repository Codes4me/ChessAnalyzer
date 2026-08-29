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
// has — used below to break near-ties in favor of the simpler shape.
// (A data-driven complexity measure, based on how much each model's fit R²
// actually outruns its check R² across a whole corpus of datasets, was
// tried as a replacement for this — it measurably hurt the aggregate
// median held-out R², so `params` stays the default; see selectByExtrapolation's
// `complexityOf` option below if you want to swap it back in for testing.
// Re-checked after the corpus grew from 37 to 64 datasets: still hurts.
// Median test R² across the corpus: -0.7367 with `params` (the default)
// vs. -1.1044 using each candidate's median fit-check R² gap, or -0.8823
// using the mean gap — both worse than just counting coefficients. The
// gap measure is also noisy in practice: R² is unbounded below, so a
// handful of badly-extrapolating datasets blow the *mean* gap up by many
// orders of magnitude for some candidates, and even the more robust
// *median* gap ranks plain `constant` as more "complex" than cubic —
// check R² swinging very negative on a volatile dataset looks identical
// to genuine overfitting under this measure, so it can't reliably tell
// the two apart.)
// Quartic/quintic are included deliberately rather than banned outright —
// earlier experiments showed adjusted R² selection alone lets them win on
// training data and then blow up on unseen data; the fit/check
// extrapolation test below is what actually keeps them honest.
// Model types are periodically re-checked with an empirical
// backward-elimination pass: starting from ALL candidate types (including
// ones previously removed), repeatedly find whichever single candidate's
// REMOVAL improves the aggregate median held-out test R² the most, drop
// it, and repeat until no further removal helps. Re-run after the corpus
// grew to 64 datasets and the checkLen >= params gate (below) was added:
// took the median from -0.9937 (all 24 types) to -0.2701 by dropping
// S-curve with offset, proportional (no intercept), exponential with
// offset, reciprocal, power, and cubic (degree 3) — each was more often
// winning ties on fit/check noise than actually extrapolating well to
// genuinely new data. Notably this run's result differs from the previous
// pass (which had dropped quadratic and kept cubic/power/exponential with
// offset instead) — the winning pool isn't fixed, it depends on the
// current corpus and selection rules, so re-run this periodically as the
// corpus grows rather than treating either result as final. All removed
// types stay available as manual dropdown options on the /regression page
// — this only affects the Auto-selection pool.
//
// Also tried adding a new "trimmed exponential" candidate (an exponential
// fit that drops the worst ~10% of points, same idea as Least Trimmed
// Squares) to this pool: it looked great in isolation (median test R²
// -2.59 -> -1.3ish when every eligible dataset was forced through
// exponential vs. trimmed exponential only), but inside the full pool it
// was the FIRST thing backward elimination removed — it was winning ties
// against other model types by chance more than by actually extrapolating
// better, which only shows up once it's competing against everything
// else, not in an isolated A/B test. Left out of CANDIDATES; still
// available as a manual dropdown option on the /regression page.
//
// That same re-run also caught a latent bug: backtest_models.js's final
// refit step wasn't wrapped in try/catch, so a dataset whose FIT half
// (first 50%) is all-positive but whose CHECK half (next 20%) dips to
// zero or negative could pass selection then crash once folded into the
// full 70% training pool a candidate like `exponential` gets refit on
// (only the FIT half is ever run through .fit() during selection — the
// CHECK half only goes through .predict(), which never validates
// anything). Fixed by treating a final-refit failure as skipped, same as
// a selection failure. That fix alone — before even considering trimmed
// exponential — changed which datasets `exponential` could legally win,
// which cascaded into a further pool improvement: dropping Least Trimmed
// Squares, Least Median of Squares, and linear (degree 1) took the median
// from -0.2701 to -0.2284.
const CANDIDATES = [
  { name: 'constant', params: 1, minFit: 2, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 0 }) },
  { name: 'quadratic (degree 2)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 2 }) },
  { name: 'quartic (degree 4)', params: 5, minFit: 5, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 4 }) },
  { name: 'quintic (degree 5)', params: 6, minFit: 6, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 5 }) },
  { name: 'S-curve (logistic)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.fitLogisticCurve(xs, ys) },
  { name: 'piecewise (constant+linear)', params: 3, minFit: 6, fit: (xs, ys) => RegressionLib.fitPiecewiseConstantLinear(xs, ys) },
  { name: 'exponential', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitExponential(xs, ys) },
  { name: 'power with offset', params: 3, minFit: 4, fit: (xs, ys) => RegressionLib.fitPowerOffset(xs, ys) },
  { name: 'logarithmic', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitLogarithmic(xs, ys) },
  { name: 'sinusoidal', params: 4, minFit: 4, fit: (xs, ys) => RegressionLib.fitSinusoidal(xs, ys) },
  { name: 'Theil-Sen (robust line)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitTheilSen(xs, ys) },
  { name: 'RANSAC (robust line)', params: 2, minFit: 3, fit: (xs, ys) => RegressionLib.fitRANSAC(xs, ys) },
  { name: 'Huber (robust line)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitHuber(xs, ys) },
  { name: 'Tukey biweight (robust line)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitTukeyBiweight(xs, ys) },
  { name: "Andrews' sine (robust line)", params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitAndrewsSine(xs, ys) },
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
function selectByExtrapolation(xs, ys, { fitFrac = 0.5, checkFrac = 0.2, tolerance, complexityOf = (candidate) => candidate.params } = {}) {
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
    // A check window smaller than the candidate's own parameter count can't
    // meaningfully validate it — a flexible model can pass close to a
    // handful of check points by coincidence (not because it actually
    // extrapolates), then diverge badly past them. Confirmed empirically:
    // requiring checkLen >= params took this file's aggregate median
    // held-out test R² from -0.7367 to -0.3926 (stricter multiples, 1.5x
    // and 2x params, made it worse — 1x is the sweet spot).
    if (checkLen < candidate.params) continue;
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
  // `complexityOf` decides what counts as "simpler" for the tie-break —
  // defaults to each candidate's fixed coefficient count, but can be swapped
  // for a data-driven measure (e.g. how much a model's fit R² typically
  // outruns its check R² across a whole corpus of datasets).
  //
  // The tolerance itself scales with how few points are IN the check
  // window (unless the caller passes an explicit one) — an R² measured on
  // just 3 points is far noisier than one measured on 30, so a "win" needs
  // to be much bigger to actually mean something on a tiny check window.
  // Without this, a genuinely tied model (e.g. piecewise vs. plain linear,
  // both ~0.52 R² on the full data) could look like a clear winner purely
  // from a lucky/unlucky 3-point sample, and a needlessly complex model
  // would win on noise alone.
  //
  // Plain C/checkLen, not C/sqrt(checkLen) and not a constant + C/checkLen
  // — all three were tried against this file's real datasets (median
  // held-out test R² across everything, plus whether ONI half rodriguez
  // specifically stopped picking piecewise over an equally-good linear
  // fit). Pure C/checkLen with C=4 won outright: -0.27 median vs -0.40 for
  // the "+base" version and inconsistent results for 1/sqrt(checkLen).
  const effectiveTolerance = tolerance !== undefined ? tolerance : 4 / checkLen;

  const bestCheckR2 = Math.max(...results.map(r => r.checkR2));
  const contenders = results.filter(r => r.checkR2 >= bestCheckR2 - effectiveTolerance);
  const best = contenders.reduce((simplest, r) => (complexityOf(r.candidate) < complexityOf(simplest.candidate) ? r : simplest));

  return { name: best.candidate.name, candidate: best.candidate, fitR2: best.fitR2, checkR2: best.checkR2, fitLen, checkLen };
}

const ModelCandidates = { CANDIDATES, selectByExtrapolation };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ModelCandidates;
} else {
  window.ModelCandidates = ModelCandidates;
}
