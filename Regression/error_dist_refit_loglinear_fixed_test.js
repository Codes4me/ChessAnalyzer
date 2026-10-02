// Re-validates the log-linear robust refit (regression.js: maybeRobustRefitLogLinear)
// after fixing the space mismatch found in error_dist_refit_other_types_test.js
// (that version detected/reweighted on ORIGINAL-scale residuals, which
// confirmed-HURT the power fit even under clean Gaussian noise). This
// version detects and reweights in the TRANSFORMED space each fit actually
// runs in (ln(y) for exponential, ln(x)/ln(y) for power, ln(x) for
// logarithmic) — same synthetic ground-truth methodology as before,
// parameter-recovery error against the known true (a, b), bootstrap CI.
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
function bootstrapMedianCI(diffs, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = diffs.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) samples.push(median(Array.from({ length: n }, () => diffs[Math.floor(random() * n)])));
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}
function paramError(fit, trueA, trueB) { return (fit.a - trueA) ** 2 + (fit.b - trueB) ** 2; }

const N = 60;
const xsPositive = Array.from({ length: N }, (_, i) => i + 1);

function genNoisy(rand, trueFn, opts) {
  return xsPositive.map(x => {
    const base = trueFn(x);
    if (opts.mode === 'gaussian') return base + randNormal(rand) * (0.03 * Math.abs(base) + 1);
    if (opts.mode === 'laplace') return base + randLaplace(rand, 0.02 * Math.abs(base) + 1);
    if (opts.mode === 't3') return base + randStudentT(rand, 3) * (0.02 * Math.abs(base) + 1);
    const sd = 0.03 * Math.abs(base) + 1;
    const isOutlier = rand() < opts.frac;
    if (!isOutlier) return base + randNormal(rand) * sd;
    const outMag = sd * 15;
    return base + (opts.skew ? (outMag + rand() * outMag) : randNormal(rand) * outMag);
  }).map(y => Math.max(1e-6, y)); // keep y positive for exponential/power's domain requirement
}

function runModelFamily(familyName, kind, fitFn, genYs, trueA, trueB, trials = 400, seedBase = 1000) {
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
    let flaggedCount = 0, n = 0;
    for (let t = 0; t < trials; t++) {
      const rand = mulberry32(seedBase + t * 7919);
      const ys = gen(rand);
      let olsFit;
      try { olsFit = fitFn(xsPositive, ys); } catch (e) { continue; }
      const result = Regression.maybeRobustRefitLogLinear(olsFit, xsPositive, ys, kind);
      if (result.switched) flaggedCount++;
      diffs.push(paramError(result.fit, trueA, trueB) - paramError(olsFit, trueA, trueB));
      n++;
    }
    if (!diffs.length) { console.log(`  ${label.padEnd(36)} (no valid trials)`); continue; }
    const ci = bootstrapMedianCI(diffs);
    const verdict = ci.upper < 0 ? 'refit HELPS' : ci.lower > 0 ? 'refit HURTS' : 'no reliable effect';
    console.log(`  ${label.padEnd(36)} switched ${flaggedCount}/${n}  median diff=${median(diffs).toFixed(4)}  CI[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]  -> ${verdict}`);
  }
}

runModelFamily('Exponential (y = a*e^(bx))', 'exponential', Regression.fitExponential,
  (rand, opts) => genNoisy(rand, x => 5 * Math.exp(0.03 * x), opts), 5, 0.03);

runModelFamily('Power (y = a*x^b)', 'power', Regression.fitPower,
  (rand, opts) => genNoisy(rand, x => 2 * Math.pow(x, 1.4), opts), 2, 1.4);

runModelFamily('Logarithmic (y = a*ln(x) + b)', 'logarithmic', Regression.fitLogarithmic,
  (rand, opts) => genNoisy(rand, x => 4 * Math.log(x) + 10, opts), 4, 10);
