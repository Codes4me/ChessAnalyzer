// Exploratory feasibility check for Bayesian 2SLS (two-stage least
// squares), NOT tied to any current dataset in data.txt -- none of this
// project's 7 multi-column datasets have a labeled column that plausibly
// serves as an instrument (see conversation: all just "x | y | list3").
// This is purely: does the method work, and is it worth having as a
// general capability for future datasets that DO have a real instrument?
//
// Setup being tested (textbook endogeneity): a confounder U affects both
// the "endogenous" regressor X and the outcome Y directly, so plain OLS of
// Y~X is biased (X carries some of U's effect, not just its own). Z is a
// valid INSTRUMENT: it affects X, but affects Y only THROUGH X, and is
// independent of U. 2SLS's whole point: regress X on Z first (stage 1),
// use the FITTED Xhat (purged of its correlation with U, since Z doesn't
// correlate with U) as the regressor for Y (stage 2) -- recovering the
// true causal effect of X on Y despite the confounding.
//
// "Bayesian" version here: stage 1 is ordinary OLS (X on Z); stage 2 is
// analytic Bayesian linear regression of Y on Xhat (conjugate Normal
// prior on the coefficient, Normal-Inverse-Gamma for noise variance),
// giving a posterior mean + credible interval for the causal effect
// instead of a plain point estimate + classical SE. This is the standard
// "second stage is Bayesian, first stage is plain OLS" approach -- not a
// full joint simultaneous-equations Bayesian IV model, which is a much
// heavier lift and arguably overkill until a real instrument-bearing
// dataset actually needs it.

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

function olsSimple(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = my - slope * mx;
  return { slope, intercept, predict: v => intercept + slope * v };
}

// Analytic Bayesian simple linear regression: y = a + b*xhat + noise.
// Prior: a, b ~ N(0, tau^2) independently (weakly informative, tau large);
// noise variance sigma^2 estimated via residual variance from the stage-1
// fit itself (empirical-Bayes style plug-in -- simpler than a full
// Normal-Inverse-Gamma conjugate update, and fine for a feasibility check;
// a production version would want the full NIG posterior for sigma too).
function bayesianLinearPosterior(xhat, y, { tau = 1000 } = {}) {
  const n = xhat.length;
  // Design matrix [1, xhat]. Prior precision = I / tau^2.
  const priorPrecision = 1 / (tau * tau);
  // First get an OLS point estimate to estimate residual noise variance.
  const ols = olsSimple(xhat, y);
  const resid = xhat.map((x, i) => y[i] - ols.predict(x));
  const sigma2 = resid.reduce((s, r) => s + r * r, 0) / Math.max(1, n - 2);

  // Posterior precision matrix = X'X/sigma2 + priorPrecision*I (2x2, since params = [a,b]).
  let Sxx = 0, Sx = 0, Sxy = 0, Sy = 0;
  for (let i = 0; i < n; i++) { Sx += 1; Sxx += xhat[i] * xhat[i]; Sxy += xhat[i] * y[i]; Sy += y[i]; }
  const A = [[n / sigma2 + priorPrecision, xhat.reduce((s, v) => s + v, 0) / sigma2],
             [xhat.reduce((s, v) => s + v, 0) / sigma2, Sxx / sigma2 + priorPrecision]];
  const b = [Sy / sigma2, Sxy / sigma2];
  // Solve 2x2 system for posterior mean [aPost, bPost].
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const aPost = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
  const bPost = (A[0][0] * b[1] - A[1][0] * b[0]) / det;
  // Posterior covariance = inverse of A.
  const covBB = A[0][0] / det; // [1][1] entry of inverse when solving [[A,B],[C,D]]^-1 -- see below
  // Proper 2x2 inverse: inv = (1/det) * [[D, -B], [-C, A]]
  const invBB = A[0][0] / det;
  const sdB = Math.sqrt(invBB);
  return { intercept: aPost, slope: bPost, slopeSD: sdB, sigma2 };
}

function twoStageBayesian(Z, X, Y) {
  const stage1 = olsSimple(Z, X);
  const Xhat = Z.map(z => stage1.predict(z));
  const stage2 = bayesianLinearPosterior(Xhat, Y);
  return { stage1, stage2, firstStageR2: (() => {
    const mx = X.reduce((a, b) => a + b, 0) / X.length;
    const sst = X.reduce((s, v) => s + (v - mx) ** 2, 0);
    const sse = X.reduce((s, v, i) => s + (v - Xhat[i]) ** 2, 0);
    return sst === 0 ? 0 : 1 - sse / sst;
  })() };
}

// --- Simulation: confounded data + a valid instrument ---
function simulate(seed, n, { trueBeta = 2, instrumentStrength = 1.0, confoundStrength = 1.5 } = {}) {
  const rand = mulberry32(seed);
  const Z = [], X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const z = randNormal(rand); // instrument: exogenous, independent of U by construction
    const u = randNormal(rand); // confounder: affects both X and Y, never observed/used by either method
    const x = instrumentStrength * z + confoundStrength * u + randNormal(rand) * 0.5;
    const y = trueBeta * x + confoundStrength * 1.8 * u + randNormal(rand) * 0.5; // U also leaks into Y directly -- the bias source
    Z.push(z); X.push(x); Y.push(y);
  }
  return { Z, X, Y };
}

console.log('=== Part 1: does Bayesian 2SLS recover the true causal effect where OLS is biased? ===\n');
console.log('True beta = 2.0 in every trial. Confounder U inflates OLS upward (U pushes X and Y the same direction).\n');
const TRIALS = 500;
for (const n of [50, 200]) {
  const olsEstimates = [], tslsEstimates = [], tslsCovered = [];
  for (let t = 0; t < TRIALS; t++) {
    const { Z, X, Y } = simulate(900000 + n * 31 + t, n);
    const naiveOls = olsSimple(X, Y);
    const tsls = twoStageBayesian(Z, X, Y);
    olsEstimates.push(naiveOls.slope);
    tslsEstimates.push(tsls.stage2.slope);
    const lo = tsls.stage2.slope - 1.96 * tsls.stage2.slopeSD, hi = tsls.stage2.slope + 1.96 * tsls.stage2.slopeSD;
    tslsCovered.push(lo <= 2 && 2 <= hi);
  }
  const meanOls = olsEstimates.reduce((a, b) => a + b, 0) / TRIALS;
  const meanTsls = tslsEstimates.reduce((a, b) => a + b, 0) / TRIALS;
  const coverage = tslsCovered.filter(Boolean).length / TRIALS;
  console.log(`--- n=${n} ---`);
  console.log(`  plain OLS:          mean estimate = ${meanOls.toFixed(3)}  (bias = ${(meanOls - 2).toFixed(3)})`);
  console.log(`  Bayesian 2SLS:      mean estimate = ${meanTsls.toFixed(3)}  (bias = ${(meanTsls - 2).toFixed(3)})`);
  console.log(`  95% credible interval coverage of true beta=2: ${(100 * coverage).toFixed(1)}%  (target: ~95%)\n`);
}

console.log('=== Part 2: weak-instrument sensitivity (the classic 2SLS failure mode) ===\n');
console.log('Same setup, but the instrument\'s effect on X is made progressively weaker.\n');
for (const strength of [1.0, 0.3, 0.1, 0.03]) {
  const n = 200;
  const olsEstimates = [], tslsEstimates = [], firstStageR2s = [];
  for (let t = 0; t < TRIALS; t++) {
    const { Z, X, Y } = simulate(700000 + t, n, { instrumentStrength: strength });
    const naiveOls = olsSimple(X, Y);
    const tsls = twoStageBayesian(Z, X, Y);
    olsEstimates.push(naiveOls.slope);
    tslsEstimates.push(tsls.stage2.slope);
    firstStageR2s.push(tsls.firstStageR2);
  }
  const meanOls = olsEstimates.reduce((a, b) => a + b, 0) / TRIALS;
  const meanTsls = tslsEstimates.reduce((a, b) => a + b, 0) / TRIALS;
  const meanR2 = firstStageR2s.reduce((a, b) => a + b, 0) / TRIALS;
  const sdTsls = Math.sqrt(tslsEstimates.reduce((s, v) => s + (v - meanTsls) ** 2, 0) / TRIALS);
  console.log(`  instrument strength=${strength}:  first-stage R²=${meanR2.toFixed(3)}  OLS bias=${(meanOls - 2).toFixed(3)}  2SLS bias=${(meanTsls - 2).toFixed(3)}  2SLS estimate SD=${sdTsls.toFixed(3)}`);
}
console.log('\n(as instrument strength -> 0, 2SLS should drift back toward the SAME bias as OLS and its variance should blow up -- the textbook "weak instrument" failure mode. If this check doesn\'t show that pattern, something\'s wrong with the implementation, not with 2SLS as a method.)');
