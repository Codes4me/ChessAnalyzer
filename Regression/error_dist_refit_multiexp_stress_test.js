// Same stress-test methodology as error_dist_refit_stress_test.js and
// error_dist_refit_loglinear_stress_test.js (which found no looseness
// problem in either the polynomial or log-linear gates), applied to
// fitMultiExponential's own threshold (ΔBIC>=70): model misspecification,
// smooth heteroscedasticity with no discrete outliers, high-frequency
// moderate contamination, extreme pure heavy tails, and bimodal noise.
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

const THRESHOLD = 70; // regression.js's ROBUST_REFIT_MULTIEXP_DELTA_BIC_THRESHOLD
const N = 60;
const X2 = Array.from({ length: N }, (_, i) => [i + 1, Math.sin(i / 5) * 10 + 20]);
const TRUE_A = 5, TRUE_COEFFS = [0.02, -0.01];
const trueY = X2.map(row => TRUE_A * Math.exp(TRUE_COEFFS[0] * row[0] + TRUE_COEFFS[1] * row[1]));

function paramError(fit, trueA, trueCoeffs) {
  let e = (fit.a - trueA) ** 2;
  trueCoeffs.forEach((c, j) => { e += (fit.coeffs[j] - c) ** 2; });
  return e;
}

function runScenario(name, genY, trials = 400, seedBase = 1000) {
  let flaggedCount = 0, triggeredCount = 0, degenerateCount = 0;
  const triggeredDiffs = [];
  const names = {};
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const y = genY(rand);
    let olsFit;
    try { olsFit = Regression.fitMultiExponential(X2, y); } catch (e) { continue; }
    const result = Regression.maybeRobustRefitMultiExponential(olsFit, X2, y);
    const tResiduals = X2.map((row, i) => Math.log(y[i]) - Math.log(olsFit.predict(row)));
    const errDist = Regression.chooseErrorDistribution(tResiduals);
    if (!errDist) continue;
    const nonGaussian = errDist.best.name !== 'Gaussian';
    if (nonGaussian) { flaggedCount++; names[errDist.best.name] = (names[errDist.best.name] || 0) + 1; }
    if (!nonGaussian || errDist.deltaBICvsGaussian < THRESHOLD) continue;
    triggeredCount++;
    if (!result.switched) { degenerateCount++; continue; } // gate's own internal check disagreed — shouldn't happen, flag if it does
    const diff = paramError(result.fit, TRUE_A, TRUE_COEFFS) - paramError(olsFit, TRUE_A, TRUE_COEFFS);
    if (isFinite(diff)) triggeredDiffs.push(diff); else degenerateCount++;
  }
  const ci = bootstrapMedianCI(triggeredDiffs);
  const verdict = ci.unstable ? 'too few triggered cases to judge'
    : ci.upper < 0 ? 'refit HELPS at this threshold'
    : ci.lower > 0 ? `*** refit HURTS — ${THRESHOLD} may be TOO LOOSE here ***`
    : 'no reliable effect at this threshold';
  console.log(`\n=== ${name} (threshold=${THRESHOLD}) ===`);
  console.log(`  flagged non-Gaussian: ${flaggedCount}/${trials} ${JSON.stringify(names)}`);
  console.log(`  triggered (ΔBIC>=${THRESHOLD}): ${triggeredCount}/${trials}` + (degenerateCount ? `  (${degenerateCount} mismatched/degenerate)` : ''));
  if (triggeredDiffs.length) {
    console.log(`  among triggered: median diff=${median(triggeredDiffs).toFixed(4)}  mean=${mean(triggeredDiffs).toFixed(4)}` +
      (ci.unstable ? '' : `  95% CI [${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`));
  }
  console.log(`  -> ${verdict}`);
}

// 1. Model misspecification: a small extra wiggle the plain log-linear form can't capture.
runScenario('Model misspecification (small extra x1^2 term in the exponent)',
  rand => trueY.map((v, i) => { const x1 = X2[i][0]; const base = v * Math.exp(0.00005 * x1 * x1); return Math.max(1e-6, base + randNormal(rand) * (0.03 * base + 1)); }));

// 2. Smooth heteroscedasticity, no discrete outliers.
runScenario('Smooth heteroscedasticity (sd grows faster than base)',
  rand => trueY.map((v, i) => Math.max(1e-6, v + randNormal(rand) * (0.15 * X2[i][0] + 1))));

// 3. High-frequency moderate contamination (30%, moderate scale).
runScenario('High-frequency moderate contamination (30%, 5x)',
  rand => trueY.map(v => { const sd = 0.03 * v + 1; const out = rand() < 0.3; return Math.max(1e-6, v + randNormal(rand) * (out ? sd * 5 : sd)); }));

// 4. Extremely heavy pure tails.
runScenario('Pure Student-t df=2 (extreme tails)',
  rand => trueY.map(v => Math.max(1e-6, v + randStudentT(rand, 2) * (0.02 * v + 1))));
runScenario('Pure Cauchy (df=1, undefined variance)',
  rand => trueY.map(v => Math.max(1e-6, v + randStudentT(rand, 1) * (0.015 * v + 1))));

// 5. Bimodal noise: 50/50 two clusters.
runScenario('Bimodal noise (50/50, offset +/- 3sd)',
  rand => trueY.map(v => { const sd = 0.03 * v + 1; return Math.max(1e-6, v + (rand() < 0.5 ? 3 * sd : -3 * sd) + randNormal(rand) * sd); }));
