// Finds the ΔBIC threshold that should gate the robust refit (only refit
// when chooseErrorDistribution's evidence against Gaussian is at least
// this strong), instead of refitting on ANY non-Gaussian flag — the
// earlier synthetic test showed refitting helps hugely under outlier
// contamination (especially skewed) but does nothing (or loses a little)
// for smoothly heavy-tailed noise, so a blanket "refit whenever flagged"
// rule wastes effort and occasionally hurts.
//
// Method: build a pool of (ΔBIC, diff) pairs — diff = refitErr - olsErr,
// negative means refitting helped — from many trials/datasets spanning a
// wide range of ΔBIC. For a candidate threshold T, "applying the gate"
// means: refit only when ΔBIC>=T, so the realized outcome per case is
// (ΔBIC>=T ? diff : 0). The optimal T minimizes the mean of that realized
// outcome (most negative = best net benefit). Bootstrap (resample the pool,
// re-find the optimal T each time) gives a distribution of "optimal T", not
// just one point estimate off a single noisy sample — then the net benefit
// of that optimal T, and of T tightened 10%/20%/30% (requiring MORE
// evidence, i.e. T*1.1/1.2/1.3), is itself re-evaluated with its own
// bootstrap CI, on both synthetic (known ground truth) and real data.
const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');

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

const TRUE_A = 3, TRUE_B = 2, N = 60;
const SYNTH_XS = Array.from({ length: N }, (_, i) => i + 1);

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
function refitFor(xs, ys, errDist, olsFit) {
  if (errDist.best.name === 'Laplace') return fitL1IRLS(xs, ys) || olsFit;
  if (errDist.best.name === 'Student-t') return fitStudentTIRLS(xs, ys, errDist.best.params.nu) || olsFit;
  return olsFit;
}
function paramError(fit) { return (fit.a - TRUE_A) ** 2 + (fit.b - TRUE_B) ** 2; }

function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }

// --- Build the SYNTHETIC pool: sweep contamination fraction x scale x
// symmetric/skewed, so ΔBIC and diff span a wide, continuous-ish range
// instead of a handful of fixed scenarios.
function buildSyntheticPool() {
  const pool = [];
  const fractions = [0, 0.02, 0.05, 0.08, 0.10, 0.15];
  const scales = [3, 5, 8, 12, 20];
  let seedCounter = 0;
  for (const frac of fractions) {
    for (const scale of scales) {
      for (const skewed of [false, true]) {
        for (let t = 0; t < 25; t++) {
          seedCounter++;
          const rand = mulberry32(100000 + seedCounter * 7919);
          const ys = SYNTH_XS.map(x => {
            const isOutlier = rand() < frac;
            if (!isOutlier) return TRUE_A + TRUE_B * x + randNormal(rand) * 2;
            return TRUE_A + TRUE_B * x + (skewed ? (scale * 2 + rand() * scale * 2) : randNormal(rand) * scale * 2);
          });
          const olsFit = weightedLine(SYNTH_XS, ys, SYNTH_XS.map(() => 1));
          if (!olsFit) continue;
          const residuals = SYNTH_XS.map((x, i) => ys[i] - olsFit.predict(x));
          const errDist = Regression.chooseErrorDistribution(residuals);
          if (!errDist) continue;
          const deltaBIC = errDist.best.name === 'Gaussian' ? 0 : errDist.deltaBICvsGaussian;
          if (deltaBIC <= 0) { pool.push({ deltaBIC: 0, diff: 0 }); continue; }
          const refit = refitFor(SYNTH_XS, ys, errDist, olsFit);
          const diff = paramError(refit) - paramError(olsFit);
          pool.push({ deltaBIC, diff });
        }
      }
    }
  }
  return pool;
}

// --- Build the REAL-DATA pool (same methodology as error_dist_refit_test.js,
// but keeping every dataset's (ΔBIC, diff) regardless of flag strength).
function buildRealPool(inputPath) {
  const rawText = fs.readFileSync(inputPath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);
  const pool = [];
  for (const group of groups) {
    const xs = group.columns[0], ys = group.columns[1];
    const valid = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
    const vxs = xs.filter((_, i) => valid[i]), vys = ys.filter((_, i) => valid[i]);
    if (vxs.length < 10) continue;
    const trainLen = Math.floor(vxs.length * 0.7);
    const testLen = vxs.length - trainLen;
    if (trainLen < 6 || testLen < 3) continue;
    const trainXs = vxs.slice(0, trainLen), trainYs = vys.slice(0, trainLen);
    const testXs = vxs.slice(trainLen), testYs = vys.slice(trainLen);
    let olsFit;
    try { olsFit = Regression.runRegression(trainXs, trainYs, { degree: 1 }); } catch (e) { continue; }
    const trainResiduals = trainXs.map((x, i) => trainYs[i] - olsFit.predict(x));
    const errDist = Regression.chooseErrorDistribution(trainResiduals);
    if (!errDist) continue;
    const deltaBIC = errDist.best.name === 'Gaussian' ? 0 : errDist.deltaBICvsGaussian;
    if (deltaBIC <= 0) { pool.push({ deltaBIC: 0, diff: 0, title: group.title }); continue; }
    let refit = null;
    if (errDist.best.name === 'Laplace') refit = fitL1IRLS(trainXs, trainYs);
    else if (errDist.best.name === 'Student-t') refit = fitStudentTIRLS(trainXs, trainYs, errDist.best.params.nu);
    if (!refit) continue;
    const olsR2 = Regression.computeR2(testXs, testYs, x => olsFit.predict(x));
    const refitR2 = Regression.computeR2(testXs, testYs, refit.predict);
    // For real data there's no known true (a,b), so "diff" here is
    // held-out R^2 improvement instead of parameter error — NEGATIVE
    // paramError-style convention kept consistent by flipping sign
    // (refitR2 - olsR2 > 0 is good, so store as -(refitR2-olsR2) so that,
    // same as the synthetic pool, negative = refit helped).
    pool.push({ deltaBIC, diff: -(refitR2 - olsR2), title: group.title });
  }
  return pool;
}

const MIN_TRIGGERED = 5; // too few triggered cases makes the conditional median meaningless/unstable

// CONDITIONAL median — among only the cases that WOULD trigger at this
// threshold, what's the typical (median) effect? Two simpler alternatives
// were tried and rejected first: a plain mean over the whole pool is
// dominated by a couple of extreme wins (e.g. Movie Gross's -207 on the
// real pool), making T=0 look trivially "optimal" regardless of how many
// small-harm weak-evidence cases exist. A plain median over the WHOLE pool
// (treating untriggered cases as 0) gets stuck at exactly 0 whenever fewer
// than half the pool ever triggers at all (true for the real pool: only
// 13/44 flagged) — mathematically correct but useless, since it can't
// distinguish any threshold from any other once majority of entries are 0.
// Conditioning on "cases that actually trigger" fixes both: it answers the
// real question ("when this DOES fire, does it typically help") instead of
// being swamped by either outliers or the non-firing majority.
function netBenefitAt(pool, T) {
  const triggered = pool.filter(p => p.deltaBIC >= T).map(p => p.diff);
  if (triggered.length < MIN_TRIGGERED) return Infinity; // disqualify: not enough data to trust
  return median(triggered);
}
function meanNetBenefitAt(pool, T) {
  const triggered = pool.filter(p => p.deltaBIC >= T).map(p => p.diff);
  return triggered.length ? mean(triggered) : 0;
}

function findOptimalThreshold(pool, grid) {
  let bestT = grid[0], bestVal = Infinity;
  for (const T of grid) {
    const v = netBenefitAt(pool, T);
    if (v < bestVal) { bestVal = v; bestT = T; }
  }
  return bestT;
}

function bootstrapOptimalThreshold(pool, grid, { iterations = 1000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = pool.length;
  const optimalTs = [];
  for (let i = 0; i < iterations; i++) {
    const resample = Array.from({ length: n }, () => pool[Math.floor(random() * n)]);
    optimalTs.push(findOptimalThreshold(resample, grid));
  }
  optimalTs.sort((a, b) => a - b);
  return {
    median: median(optimalTs),
    lower: optimalTs[Math.floor(0.025 * (optimalTs.length - 1))],
    upper: optimalTs[Math.ceil(0.975 * (optimalTs.length - 1))],
  };
}

function bootstrapNetBenefitCI(pool, T, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = pool.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) {
    const resample = Array.from({ length: n }, () => pool[Math.floor(random() * n)]);
    const v = netBenefitAt(resample, T);
    if (isFinite(v)) samples.push(v); // drop resamples with too few triggered cases to measure
  }
  if (samples.length < iterations * 0.5) return { lower: NaN, upper: NaN, unstable: true };
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

const GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 22, 26, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300];

function analyzePool(name, pool) {
  console.log(`\n########## ${name} (${pool.length} cases) ##########`);
  const nonTrivial = pool.filter(p => p.deltaBIC > 0).length;
  console.log(`  non-Gaussian-flagged cases: ${nonTrivial}/${pool.length}`);

  const optBoot = bootstrapOptimalThreshold(pool, GRID);
  console.log(`  bootstrapped optimal ΔBIC threshold: median=${optBoot.median.toFixed(2)}  95% CI [${optBoot.lower.toFixed(2)}, ${optBoot.upper.toFixed(2)}]`);

  const pointOptimalT = findOptimalThreshold(pool, GRID);
  console.log(`  point-estimate optimal threshold on full pool: ${pointOptimalT}`);

  const variants = [
    ['optimal', pointOptimalT],
    ['10% tighter', pointOptimalT * 1.1],
    ['20% tighter', pointOptimalT * 1.2],
    ['30% tighter', pointOptimalT * 1.3],
  ];
  for (const [label, T] of variants) {
    const nb = netBenefitAt(pool, T);
    const meanNb = meanNetBenefitAt(pool, T);
    const ci = bootstrapNetBenefitCI(pool, T);
    const triggered = pool.filter(p => p.deltaBIC >= T).length;
    const nbStr = isFinite(nb) ? nb.toFixed(4).padStart(9) : '  (n<5)'.padStart(9);
    const ciStr = ci.unstable ? '[unstable: too few triggered cases in most resamples]'
      : `[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`;
    const verdict = ci.unstable ? '  -> not enough data to judge'
      : ci.upper < 0 ? '  -> confirmed net positive'
      : ci.lower > 0 ? '  -> confirmed net NEGATIVE'
      : '  -> not reliably different from 0';
    console.log(`  T=${T.toFixed(2).padStart(7)} (${label.padEnd(12)}) triggers on ${String(triggered).padStart(3)}/${pool.length}  median=${nbStr}  mean=${meanNb.toFixed(4).padStart(9)}  95% CI(median) ${ciStr}${verdict}`);
  }
}

const inputPath = path.resolve(process.argv[2] || 'data.txt');
analyzePool('SYNTHETIC (known ground truth, parameter error)', buildSyntheticPool());
analyzePool('REAL DATA (held-out R^2, sign-flipped to match convention)', buildRealPool(inputPath));
