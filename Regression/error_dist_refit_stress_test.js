// Stress-test for the polynomial robust-refit gate (ROBUST_REFIT_DELTA_BIC_THRESHOLD=45
// in regression.js): every validation so far used Gaussian/Laplace/Student-t
// noise or additive outlier CONTAMINATION (rare, extreme points). This
// checks five genuinely different error shapes the gate has never been run
// against, specifically looking for a CONFIRMED NEGATIVE (net harmful)
// result among cases that clear the threshold — that would mean 45 is too
// loose for a shape the original sweep never saw.
//
//   1. Model misspecification: true curve has real (mild) quadratic
//      structure, but a LINE is fit anyway — residuals get a systematic,
//      non-random shape from lack of fit, not from noise. Does the
//      detector mistake structural misfit for a heavy-tailed error
//      distribution, and does "fixing" that with a robust refit help or
//      just mask the real problem?
//   2. Smooth heteroscedasticity, NO discrete outliers: variance grows
//      continuously with x (this is what the separate CI-method gate
//      already handles via White's test + wild bootstrap) — does the
//      error-distribution detector ALSO fire on this, and does refitting
//      help, hurt, or just duplicate what the other gate already does?
//   3. Very high contamination FREQUENCY, moderate severity (30% of points,
//      5x scale) — a different regime than the original sweep's fractions
//      (capped at 15%).
//   4. Extremely heavy pure tails: Student-t df=1 (Cauchy) and df=2 —
//      beyond the df=3/5 already tested, into "variance barely/doesn't
//      exist" territory.
//   5. Bimodal noise: 50/50 mixture of two well-separated clusters (not
//      rare contamination — HALF the points are "the other mode") — a
//      genuinely different shape than anything in the original sweep.
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
function randStudentT(r, df) { const z = randNormal(r); let c = 0; for (let i = 0; i < df; i++) { const n = randNormal(r); c += n * n; } return z / Math.sqrt(c / df); }
function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function bootstrapMedianCI(diffs, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  if (diffs.length < 5) return { unstable: true };
  const random = mulberry32(seed);
  const n = diffs.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) samples.push(median(Array.from({ length: n }, () => diffs[Math.floor(random() * n)])));
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

const TRUE_A = 3, TRUE_B = 2, N = 60;
const xs = Array.from({ length: N }, (_, i) => i + 1);
const THRESHOLD = 45; // regression.js's ROBUST_REFIT_DELTA_BIC_THRESHOLD

function weightedLine(xs, ys, w) {
  let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
  for (let i = 0; i < xs.length; i++) { sw += w[i]; swx += w[i] * xs[i]; swy += w[i] * ys[i]; swxx += w[i] * xs[i] * xs[i]; swxy += w[i] * xs[i] * ys[i]; }
  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-12) return null;
  const b = (sw * swxy - swx * swy) / denom;
  const a = (swy - b * swx) / sw;
  return { a, b, predict: x => a + b * x };
}
// fitRobustPolynomial (like fitPolynomial) returns {coeffs: [intercept, slope]},
// NOT {a, b} — only the log-linear fits (fitExponential/fitPower/etc.) use
// {a, b}. Mixing these up here silently produced NaN for every refit
// comparison (fit.a/fit.b undefined) until caught.
function paramError(fit) {
  const a = fit.a !== undefined ? fit.a : fit.coeffs[0];
  const b = fit.b !== undefined ? fit.b : fit.coeffs[1];
  return (a - TRUE_A) ** 2 + (b - TRUE_B) ** 2;
}

function runScenario(name, genYs, trials = 400, seedBase = 1000) {
  let flaggedCount = 0, triggeredCount = 0, degenerateCount = 0;
  const triggeredDiffs = [];
  const names = {};
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(rand);
    const olsFit = weightedLine(xs, ys, xs.map(() => 1));
    if (!olsFit) continue;
    const residuals = xs.map((x, i) => ys[i] - olsFit.predict(x));
    const errDist = Regression.chooseErrorDistribution(residuals);
    if (!errDist) continue;
    const nonGaussian = errDist.best.name !== 'Gaussian';
    if (nonGaussian) { flaggedCount++; names[errDist.best.name] = (names[errDist.best.name] || 0) + 1; }
    if (!nonGaussian || errDist.best.name === 'Gaussian') continue;
    if (errDist.deltaBICvsGaussian < THRESHOLD) continue;
    triggeredCount++;
    let refit;
    try { refit = Regression.fitRobustPolynomial(xs, ys, 1, errDist.best.name, errDist.best.params && errDist.best.params.nu); } catch (e) { continue; }
    const diff = paramError(refit) - paramError(olsFit);
    // Extreme tails (Cauchy in particular) can occasionally produce a
    // degenerate IRLS weight set (all near-zero, or a near-singular
    // weighted design matrix) that yields NaN/Infinity coefficients — this
    // IS a real finding about the method's robustness at that extreme, not
    // a bug to silently hide, so it's counted and reported separately
    // rather than just dropped.
    if (isFinite(diff)) triggeredDiffs.push(diff); else degenerateCount++;
  }
  const ci = bootstrapMedianCI(triggeredDiffs);
  const verdict = ci.unstable ? 'too few triggered cases to judge'
    : ci.upper < 0 ? 'refit HELPS at this threshold'
    : ci.lower > 0 ? '*** refit HURTS at this threshold — 45 may be TOO LOOSE here ***'
    : 'no reliable effect at this threshold';
  console.log(`\n=== ${name} ===`);
  console.log(`  flagged non-Gaussian: ${flaggedCount}/${trials} ${JSON.stringify(names)}`);
  console.log(`  triggered (ΔBIC>=${THRESHOLD}): ${triggeredCount}/${trials}` + (degenerateCount ? `  (${degenerateCount} produced a degenerate/non-finite refit — see note)` : ''));
  if (triggeredDiffs.length) {
    console.log(`  among triggered: median diff=${median(triggeredDiffs).toFixed(4)}  mean=${mean(triggeredDiffs).toFixed(4)}` +
      (ci.unstable ? '' : `  95% CI [${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`));
  }
  console.log(`  -> ${verdict}`);
}

// 1. Model misspecification: real (mild) quadratic structure, line fit anyway.
runScenario('Model misspecification (true mild quadratic, linear fit forced)',
  rand => xs.map(x => TRUE_A + TRUE_B * x + 0.02 * x * x + randNormal(rand) * 4));

// 2. Smooth heteroscedasticity, no discrete outliers.
runScenario('Smooth heteroscedasticity (sd = 0.15*x, no discrete outliers)',
  rand => xs.map(x => TRUE_A + TRUE_B * x + randNormal(rand) * (0.15 * x + 1)));

// 3. Very high contamination frequency (30%), moderate severity (5x).
runScenario('High-frequency moderate contamination (30% of points, 5x scale)',
  rand => xs.map(x => { const out = rand() < 0.3; return TRUE_A + TRUE_B * x + randNormal(rand) * (out ? 20 : 4); }));

// 4. Extremely heavy pure tails.
runScenario('Pure Student-t df=2 (extremely heavy, near-undefined variance)',
  rand => xs.map(x => TRUE_A + TRUE_B * x + randStudentT(rand, 2) * 2));
runScenario('Pure Cauchy (Student-t df=1, undefined variance)',
  rand => xs.map(x => TRUE_A + TRUE_B * x + randStudentT(rand, 1) * 1.5));

// 5. Bimodal noise: 50/50 two well-separated clusters.
runScenario('Bimodal noise (50/50 mixture, clusters at +/-10, sd=2 each)',
  rand => xs.map(x => TRUE_A + TRUE_B * x + (rand() < 0.5 ? 10 : -10) + randNormal(rand) * 2));
