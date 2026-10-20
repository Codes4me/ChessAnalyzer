// Tests the orphaned diagnostic battery (moretests.js: Box-Pierce, Runs
// test, Glejser, Spearman-rank heteroscedasticity — implemented but never
// validated or wired into the live tool) for sample-size BIAS, using a
// rank-based (Spearman) correlation between n and each test's p-value.
//
// This is deliberately done on PURE NULL SYNTHETIC data (true IID Gaussian
// residuals, no real autocorrelation or heteroscedasticity), not real
// corpus data — that distinction matters. Checking on real data conflates
// two different things: BIAS (a miscalibrated test whose false-positive
// rate itself drifts with n) and POWER (a correctly-calibrated test that,
// with more data, gets better at catching a REAL effect that's actually
// there). That's exactly the distinction this project already drew for
// Shapiro-Wilk: real-corpus data showed rho=0.57 between significance and
// log(n), which looked alarming until direct simulation under the true
// null showed SW's false-positive rate stays flat at ~5% from n=15 to
// n=1334 — the real-data correlation was POWER, not bias. Simulating under
// a KNOWN true null isolates bias specifically, which is what "the effect
// of dataset size on the bias" is actually asking about.
const { boxPierce, runsTest, glejserTest, spearmanHetero } = require('./moretests.js');
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
function rankArray(arr) {
  const idx = arr.map((_, i) => i).sort((a, b) => arr[a] - arr[b]);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && arr[idx[j + 1]] === arr[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = avg;
    i = j + 1;
  }
  return ranks;
}
function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxx === 0 || syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
}
function spearman(a, b) { return pearson(rankArray(a), rankArray(b)); }
// Student-t CDF for the Spearman correlation's own significance (same
// approximation used throughout this project's other sensitivity scripts).
function logGamma(x) { const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015; for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; } return -tmp + Math.log(2.5066282746310005 * ser / x); }
function betaCF(x, a, b) { const MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300; const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d; for (let m = 1; m <= MAXIT; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break; } return h; }
function regIncBeta(x, a, b) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betaCF(x, a, b) / a : 1 - bt * betaCF(1 - x, b, a) / b; }
function studentTCDF(t, df) { const x = df / (df + t * t); const ib = regIncBeta(x, df / 2, 0.5); return t > 0 ? 1 - 0.5 * ib : 0.5 * ib; }
function spearmanPValue(rho, n) { const df = n - 2; if (df < 1) return 1; const t = rho * Math.sqrt(df / (1 - rho * rho)); return 2 * (1 - studentTCDF(Math.abs(t), df)); }

const SIZES = [10, 15, 20, 30, 50, 75, 100, 150, 250, 400, 700, 1000];
const TRIALS_PER_SIZE = 400;
const TRUE_A = 3, TRUE_B = 2;

function runNullTrials() {
  // Pooled (n, pValue) pairs per test, across every sample size.
  const pooled = { boxPierce: [], runsTest: [], glejser: [], spearmanHet: [] };
  const fprBySize = { boxPierce: {}, runsTest: {}, glejser: {}, spearmanHet: {} };

  for (const n of SIZES) {
    const xs = Array.from({ length: n }, (_, i) => i + 1);
    const h = Math.max(1, Math.min(10, Math.floor(n / 5)));
    const counts = { boxPierce: 0, runsTest: 0, glejser: 0, spearmanHet: 0 };
    let valid = 0;
    for (let t = 0; t < TRIALS_PER_SIZE; t++) {
      const rand = mulberry32(900000 + n * 7919 + t);
      const ys = xs.map(x => TRUE_A + TRUE_B * x + randNormal(rand) * 4); // pure IID Gaussian — true null for ALL four tests
      let fit; try { fit = Regression.runRegression(xs, ys, { degree: 1 }); } catch (e) { continue; }
      const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
      valid++;

      const bp = boxPierce(residuals, h);
      const rt = runsTest(residuals);
      const gl = glejserTest(xs, residuals);
      const sh = spearmanHetero(xs, residuals);

      pooled.boxPierce.push([n, bp.pValue]);
      pooled.runsTest.push([n, rt.pValue]);
      pooled.glejser.push([n, gl.pValue]);
      pooled.spearmanHet.push([n, sh.pValue]);

      if (bp.pValue < 0.05) counts.boxPierce++;
      if (rt.pValue < 0.05) counts.runsTest++;
      if (gl.pValue < 0.05) counts.glejser++;
      if (sh.pValue < 0.05) counts.spearmanHet++;
    }
    for (const key of Object.keys(counts)) fprBySize[key][n] = 100 * counts[key] / valid;
  }

  return { pooled, fprBySize };
}

const { pooled, fprBySize } = runNullTrials();

console.log(`Sample-size BIAS check (pure-null synthetic data, true IID Gaussian residuals), n = ${SIZES.join(', ')}\n`);

console.log('Empirical false-positive rate by n (should stay near 5% everywhere if unbiased):\n');
const testLabels = { boxPierce: 'Box-Pierce', runsTest: 'Runs test', glejser: 'Glejser', spearmanHet: 'Spearman-rank hetero' };
for (const key of Object.keys(testLabels)) {
  const row = SIZES.map(n => `n=${n}:${fprBySize[key][n].toFixed(1)}%`).join('  ');
  console.log(`  ${testLabels[key].padEnd(22)} ${row}`);
}

console.log('\nRank-based (Spearman) correlation between n and p-value, pooled across all sizes/trials:');
console.log('(A well-calibrated test has rho~0 here — p-values are Uniform(0,1) regardless of n under the true null.');
console.log(' A nonzero rho means the false-positive rate itself drifts with sample size — genuine BIAS, not power.)\n');
for (const key of Object.keys(testLabels)) {
  const ns = pooled[key].map(p => p[0]);
  const ps = pooled[key].map(p => p[1]);
  const rho = spearman(ns, ps);
  const p = spearmanPValue(rho, ns.length);
  const verdict = p < 0.001 ? (Math.abs(rho) > 0.05 ? '*** LIKELY BIASED (significant AND non-negligible) ***' : 'statistically significant but tiny — probably not practically meaningful')
    : p < 0.05 ? 'weak evidence of bias'
    : 'no evidence of bias';
  console.log(`  ${testLabels[key].padEnd(22)} rho=${rho.toFixed(4).padStart(8)}  p=${p < 0.0001 ? '<0.0001' : p.toFixed(4)}  (n pairs=${ns.length})  -> ${verdict}`);
}
