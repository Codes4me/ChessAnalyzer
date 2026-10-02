// Threshold-optimization sweep for fitMultiExponential's robust refit, same
// methodology as the polynomial (ΔBIC=45) and log-linear (95/145/110)
// sweeps: a severity sweep of contamination fraction x scale x
// symmetric/skewed, bootstrapping the ΔBIC threshold that minimizes the
// CONDITIONAL median parameter-recovery error among triggered cases.
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
function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }

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
function fitRobustMultiExponential(X, y, distName, nu, iterations = 20) {
  const lineFit = fitRobustMultiLinear(X, y.map(Math.log), distName, nu, iterations);
  const a = Math.exp(lineFit.intercept);
  const coeffs = lineFit.coeffs;
  return { a, coeffs, predict: xRow => a * Math.exp(xRow.reduce((s, v, j) => s + v * coeffs[j], 0)) };
}
function paramError(fit, trueA, trueCoeffs) {
  let e = (fit.a - trueA) ** 2;
  trueCoeffs.forEach((c, j) => { e += (fit.coeffs[j] - c) ** 2; });
  return e;
}

const N = 60;
const X2 = Array.from({ length: N }, (_, i) => [i + 1, Math.sin(i / 5) * 10 + 20]);
const TRUE_A = 5, TRUE_COEFFS = [0.02, -0.01];
const trueY = X2.map(row => TRUE_A * Math.exp(TRUE_COEFFS[0] * row[0] + TRUE_COEFFS[1] * row[1]));

function genNoisy(rand, frac, scale, skew) {
  return trueY.map(v => {
    const sd = 0.03 * v + 1;
    const isOutlier = rand() < frac;
    if (!isOutlier) return Math.max(1e-6, v + randNormal(rand) * sd);
    const outMag = sd * scale;
    return Math.max(1e-6, v + (skew ? (outMag + rand() * outMag) : randNormal(rand) * outMag));
  });
}

function buildPool() {
  const pool = [];
  const fractions = [0, 0.02, 0.05, 0.08, 0.10, 0.15];
  const scales = [3, 5, 8, 12, 20];
  let seedCounter = 0;
  for (const frac of fractions) {
    for (const scale of scales) {
      for (const skew of [false, true]) {
        for (let t = 0; t < 15; t++) {
          seedCounter++;
          const rand = mulberry32(400000 + seedCounter * 7919);
          const y = genNoisy(rand, frac, scale, skew);
          let olsFit;
          try { olsFit = Regression.fitMultiExponential(X2, y); } catch (e) { continue; }
          const tResiduals = X2.map((row, i) => Math.log(y[i]) - Math.log(olsFit.predict(row)));
          const errDist = Regression.chooseErrorDistribution(tResiduals);
          if (!errDist) continue;
          const deltaBIC = errDist.best.name === 'Gaussian' ? 0 : errDist.deltaBICvsGaussian;
          if (deltaBIC <= 0) { pool.push({ deltaBIC: 0, diff: 0 }); continue; }
          const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
          let refit;
          try { refit = fitRobustMultiExponential(X2, y, errDist.best.name, nu); } catch (e) { continue; }
          const diff = paramError(refit, TRUE_A, TRUE_COEFFS) - paramError(olsFit, TRUE_A, TRUE_COEFFS);
          if (isFinite(diff)) pool.push({ deltaBIC, diff });
        }
      }
    }
  }
  return pool;
}

const MIN_TRIGGERED = 5;
function netBenefitAt(pool, T) {
  const triggered = pool.filter(p => p.deltaBIC >= T).map(p => p.diff);
  if (triggered.length < MIN_TRIGGERED) return Infinity;
  return median(triggered);
}
function meanNetBenefitAt(pool, T) {
  const triggered = pool.filter(p => p.deltaBIC >= T).map(p => p.diff);
  return triggered.length ? mean(triggered) : 0;
}
function findOptimalThreshold(pool, grid) {
  let bestT = grid[0], bestVal = Infinity;
  for (const T of grid) { const v = netBenefitAt(pool, T); if (v < bestVal) { bestVal = v; bestT = T; } }
  return bestT;
}
function bootstrapOptimalThreshold(pool, grid, { iterations = 1000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = pool.length;
  const optimalTs = [];
  for (let i = 0; i < iterations; i++) optimalTs.push(findOptimalThreshold(Array.from({ length: n }, () => pool[Math.floor(random() * n)]), grid));
  optimalTs.sort((a, b) => a - b);
  return { median: median(optimalTs), lower: optimalTs[Math.floor(0.025 * (optimalTs.length - 1))], upper: optimalTs[Math.ceil(0.975 * (optimalTs.length - 1))] };
}
function bootstrapNetBenefitCI(pool, T, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = pool.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) {
    const v = netBenefitAt(Array.from({ length: n }, () => pool[Math.floor(random() * n)]), T);
    if (isFinite(v)) samples.push(v);
  }
  if (samples.length < iterations * 0.5) return { unstable: true };
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

const GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 22, 26, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300];

const pool = buildPool();
console.log(`fitMultiExponential threshold sweep (${pool.length} cases)`);
console.log(`  non-Gaussian-flagged: ${pool.filter(p => p.deltaBIC > 0).length}/${pool.length}`);
const optBoot = bootstrapOptimalThreshold(pool, GRID);
console.log(`  bootstrapped optimal ΔBIC: median=${optBoot.median.toFixed(2)}  95% CI [${optBoot.lower.toFixed(2)}, ${optBoot.upper.toFixed(2)}]`);
const pointOptimalT = findOptimalThreshold(pool, GRID);
console.log(`  point-estimate optimal: ${pointOptimalT}`);
for (const [label, T] of [['optimal', pointOptimalT], ['10% tighter', pointOptimalT * 1.1], ['20% tighter', pointOptimalT * 1.2], ['30% tighter', pointOptimalT * 1.3]]) {
  const nb = netBenefitAt(pool, T);
  const meanNb = meanNetBenefitAt(pool, T);
  const ci = bootstrapNetBenefitCI(pool, T);
  const triggered = pool.filter(p => p.deltaBIC >= T).length;
  const nbStr = isFinite(nb) ? nb.toFixed(4) : '(n<5)';
  const ciStr = ci.unstable ? '[unstable]' : `[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`;
  const verdict = ci.unstable ? 'not enough data' : ci.upper < 0 ? 'confirmed net positive' : ci.lower > 0 ? 'confirmed NEGATIVE' : 'not reliable';
  console.log(`  T=${T.toFixed(2).padStart(7)} (${label.padEnd(12)}) triggers ${triggered}/${pool.length}  median=${nbStr}  mean=${meanNb.toFixed(4)}  95%CI ${ciStr}  -> ${verdict}`);
}
