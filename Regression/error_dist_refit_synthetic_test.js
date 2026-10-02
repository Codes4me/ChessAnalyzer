// Synthetic ground-truth version of error_dist_refit_test.js's question:
// does refitting under the DETECTED error distribution (Laplace -> L1 IRLS,
// Student-t -> Student-t IRLS) actually recover the true line better than
// plain OLS? The real corpus (error_dist_refit_test.js) only had 6-13
// non-Gaussian-flagged datasets to work with — too few for a confident
// bootstrap verdict. Here the true (a, b) and true error distribution are
// both known, so trial count is unlimited and the comparison is exact:
// parameter recovery error (bias^2 + variance toward the KNOWN true a, b)
// instead of R^2 against noisy real data.
//
// This also tests the FULL pipeline end-to-end, not an idealized "if you
// already knew the true distribution" version: chooseErrorDistribution has
// to actually DETECT the right distribution from a single noisy sample
// before the refit even happens, so a scenario where detection is unreliable
// (small n, weak evidence) should show up as a weaker or null effect here
// too — exactly the pattern the real-corpus version already hinted at.
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
function randStudentT(r, df) {
  const z = randNormal(r);
  let chisq = 0; for (let i = 0; i < df; i++) { const n = randNormal(r); chisq += n * n; }
  return z / Math.sqrt(chisq / df);
}

const TRUE_A = 3, TRUE_B = 2, N = 60;
const xs = Array.from({ length: N }, (_, i) => i + 1);

function weightedLine(xs, ys, w) {
  let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += w[i]; swx += w[i] * xs[i]; swy += w[i] * ys[i];
    swxx += w[i] * xs[i] * xs[i]; swxy += w[i] * xs[i] * ys[i];
  }
  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-12) return null;
  const b = (sw * swxy - swx * swy) / denom;
  const a = (swy - b * swx) / sw;
  return { a, b, predict: x => a + b * x };
}
function fitL1IRLS(xs, ys, iterations = 20) {
  let fit = weightedLine(xs, ys, xs.map(() => 1));
  if (!fit) return null;
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / xs.length * 0.05);
    const w = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    const next = weightedLine(xs, ys, w);
    if (!next) break;
    fit = next;
  }
  return fit;
}
function fitStudentTIRLS(xs, ys, nu, iterations = 20) {
  let fit = weightedLine(xs, ys, xs.map(() => 1));
  if (!fit) return null;
  let sigma2 = (() => { const r = xs.map((x, i) => ys[i] - fit.predict(x)); return r.reduce((s, v) => s + v * v, 0) / xs.length; })();
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    const w = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    const next = weightedLine(xs, ys, w);
    if (!next) break;
    fit = next;
    const newResiduals = xs.map((x, i) => ys[i] - fit.predict(x));
    const wSum = w.reduce((s, v) => s + v, 0);
    sigma2 = newResiduals.reduce((s, r, i) => s + w[i] * r * r, 0) / wSum;
    if (sigma2 <= 0) break;
  }
  return fit;
}

function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function bootstrapMedianCI(diffs, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = diffs.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) {
    const resample = Array.from({ length: n }, () => diffs[Math.floor(random() * n)]);
    samples.push(median(resample));
  }
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

// Parameter-recovery error: squared distance from the true (a,b), the
// cleanest possible metric when the ground truth is actually known (no
// test-set noise to average away, unlike R^2 against fresh data).
function paramError(fit) { return (fit.a - TRUE_A) ** 2 + (fit.b - TRUE_B) ** 2; }

function runScenario(name, genYs, trials = 500, seedBase = 1000) {
  const diffs = [];
  const olsABias = [], refitABias = [], olsBBias = [], refitBBias = [];
  let detectedNonGaussian = 0;
  const detectedNames = {};
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(rand);
    const olsFit = weightedLine(xs, ys, xs.map(() => 1));
    if (!olsFit) continue;
    const residuals = xs.map((x, i) => ys[i] - olsFit.predict(x));
    const errDist = Regression.chooseErrorDistribution(residuals);
    if (!errDist) continue;
    const nonGaussian = errDist.best.name !== 'Gaussian';
    if (nonGaussian) { detectedNonGaussian++; detectedNames[errDist.best.name] = (detectedNames[errDist.best.name] || 0) + 1; }

    let refit = olsFit;
    if (nonGaussian && errDist.best.name === 'Laplace') refit = fitL1IRLS(xs, ys) || olsFit;
    else if (nonGaussian && errDist.best.name === 'Student-t') refit = fitStudentTIRLS(xs, ys, errDist.best.params.nu) || olsFit;

    const olsErr = paramError(olsFit);
    const refitErr = paramError(refit);
    diffs.push(refitErr - olsErr); // NEGATIVE = refit is closer to truth (good)
    olsABias.push(olsFit.a - TRUE_A); refitABias.push(refit.a - TRUE_A);
    olsBBias.push(olsFit.b - TRUE_B); refitBBias.push(refit.b - TRUE_B);
  }
  const ci = bootstrapMedianCI(diffs);
  console.log(`\n=== ${name} ===`);
  console.log(`  detected non-Gaussian: ${detectedNonGaussian}/${diffs.length} (${JSON.stringify(detectedNames)})`);
  console.log(`  median(refitErr - olsErr) = ${median(diffs).toFixed(4)}   mean = ${mean(diffs).toFixed(4)}`);
  console.log(`  bootstrap 95% CI (2000 resamples): ${ci.lower.toFixed(4)} to ${ci.upper.toFixed(4)}`);
  console.log(ci.upper < 0 ? '  -> CI excludes 0 (negative): refitting under the detected distribution IS genuinely more accurate here'
    : ci.lower > 0 ? '  -> CI excludes 0 (positive): refitting is genuinely LESS accurate here'
    : '  -> CI includes 0: no reliable effect');
  // Signed bias (not squared error): a skewed contamination can push OLS's
  // estimate consistently off-target in one direction, not just add noise.
  // Mean bias near 0 = no systematic skew effect; a clearly nonzero mean
  // (with the SAME sign for OLS and refit, or different magnitudes) shows
  // whether the refit actually corrects a systematic skew-driven shift.
  console.log(`  signed bias in a: OLS mean=${mean(olsABias).toFixed(4)}  refit mean=${mean(refitABias).toFixed(4)}`);
  console.log(`  signed bias in b: OLS mean=${mean(olsBBias).toFixed(4)}  refit mean=${mean(refitBBias).toFixed(4)}`);
}

runScenario('Gaussian errors (control, sigma=4)', rand => xs.map(x => TRUE_A + TRUE_B * x + randNormal(rand) * 4));
runScenario('Laplace errors (scale=3, similar variance to sigma~4.2)', rand => xs.map(x => TRUE_A + TRUE_B * x + randLaplace(rand, 3)));
runScenario('Student-t errors, df=3 (heavy tails, scaled ~sigma 4)', rand => xs.map(x => TRUE_A + TRUE_B * x + randStudentT(rand, 3) * 2.3));
runScenario('Student-t errors, df=5', rand => xs.map(x => TRUE_A + TRUE_B * x + randStudentT(rand, 5) * 2.8));
runScenario('Contaminated normal (90% N(0,2^2), 10% N(0,20^2) outliers)', rand => xs.map(x => {
  const outlier = rand() < 0.1;
  return TRUE_A + TRUE_B * x + randNormal(rand) * (outlier ? 20 : 2);
}));
runScenario('Contaminated normal, milder (95% N(0,3^2), 5% N(0,15^2))', rand => xs.map(x => {
  const outlier = rand() < 0.05;
  return TRUE_A + TRUE_B * x + randNormal(rand) * (outlier ? 15 : 3);
}));

// --- Skewed outlier contamination: unlike the symmetric mixtures above
// (outliers equally likely positive or negative, zero mean), these are
// one-sided — a systematic source of error (measurement floor/ceiling
// effects, logged/censored data, a sensor that only misfires high, etc.).
// Neither Laplace nor Student-t (both symmetric around 0) can represent
// this shape at all, so this tests something qualitatively different from
// "heavier tails": does a symmetric robust refit (L1/Student-t IRLS, which
// treats a +40 and -40 residual identically) still help when the real
// contamination has a direction, or does it just absorb the skew into a's
// estimate the same way OLS does?
runScenario('One-sided outliers: 90% N(0,2^2), 10% ALWAYS-POSITIVE (Uniform 10 to 30)', rand => xs.map(x => {
  const outlier = rand() < 0.1;
  return TRUE_A + TRUE_B * x + (outlier ? (10 + rand() * 20) : randNormal(rand) * 2);
}));
runScenario('One-sided outliers, milder: 95% N(0,2^2), 5% ALWAYS-POSITIVE (Uniform 8 to 20)', rand => xs.map(x => {
  const outlier = rand() < 0.05;
  return TRUE_A + TRUE_B * x + (outlier ? (8 + rand() * 12) : randNormal(rand) * 2);
}));
runScenario('Asymmetric mix: 85% N(0,2^2), 10% positive outliers (~N(+15,5^2)), 5% negative (~N(-5,2^2))', rand => xs.map(x => {
  const r = rand();
  if (r < 0.10) return TRUE_A + TRUE_B * x + 15 + randNormal(rand) * 5;
  if (r < 0.15) return TRUE_A + TRUE_B * x - 5 + randNormal(rand) * 2;
  return TRUE_A + TRUE_B * x + randNormal(rand) * 2;
}));
