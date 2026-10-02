// Same methodology as error_dist_refit_threshold_test.js (which found
// ΔBIC=45's polynomial gate by bootstrapping the optimal threshold instead
// of guessing), but for the log-linear family (exponential, power,
// logarithmic) using the FIXED, space-correct detection/refit
// (maybeRobustRefitLogLinear — detects and reweights in each model's own
// transformed space, not the original scale; see the previous commit for
// why that fix mattered). Pools synthetic trials across all three families
// (a severity sweep of contamination fraction x scale x symmetric/skewed)
// plus real-corpus datasets, then bootstraps the ΔBIC threshold that
// minimizes the CONDITIONAL median effect (among cases that would actually
// trigger) — the same statistic choice error_dist_refit_threshold_test.js
// settled on after a plain mean (outlier-dominated) and a plain whole-pool
// median (swamped by the untriggered majority) both proved useless.
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
function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function paramError(fit, trueA, trueB) { return (fit.a - trueA) ** 2 + (fit.b - trueB) ** 2; }

const N = 60;
const xsPositive = Array.from({ length: N }, (_, i) => i + 1);

// Each family's transform pair and how to rebuild {a,b,predict} from a
// transformed-space line fit's coeffs — same mapping regression.js's
// LOG_LINEAR_TRANSFORMS uses internally, duplicated here only because this
// test needs to call the UNGATED refit (always refit when non-Gaussian is
// detected, at ANY ΔBIC) to sweep thresholds below 45 — maybeRobustRefitLogLinear
// itself already enforces the 45 bar internally, which would make every
// pool entry below T=45 trivially show diff=0 and defeat the whole sweep.
const FAMILIES = [
  {
    kind: 'exponential', fitFn: Regression.fitExponential, trueA: 5, trueB: 0.03, trueFn: x => 5 * Math.exp(0.03 * x),
    toX: x => x, toY: y => Math.log(y),
    fromLineFit: lineFit => { const a = Math.exp(lineFit.coeffs[0]), b = lineFit.coeffs[1]; return { a, b, predict: x => a * Math.exp(b * x) }; },
  },
  {
    kind: 'power', fitFn: Regression.fitPower, trueA: 2, trueB: 1.4, trueFn: x => 2 * Math.pow(x, 1.4),
    toX: x => Math.log(x), toY: y => Math.log(y),
    fromLineFit: lineFit => { const a = Math.exp(lineFit.coeffs[0]), b = lineFit.coeffs[1]; return { a, b, predict: x => a * Math.pow(x, b) }; },
  },
  {
    kind: 'logarithmic', fitFn: Regression.fitLogarithmic, trueA: 4, trueB: 10, trueFn: x => 4 * Math.log(x) + 10,
    toX: x => Math.log(x), toY: y => y,
    fromLineFit: lineFit => { const b = lineFit.coeffs[0], a = lineFit.coeffs[1]; return { a, b, predict: x => a * Math.log(x) + b }; },
  },
];

// Ungated refit — mirrors maybeRobustRefitLogLinear's internals exactly but
// without the ΔBIC>=45 check, so the resulting (deltaBIC, diff) pair is
// meaningful at every threshold in the sweep, not just at/above 45.
function ungatedRefit(family, xs, ys, errDist) {
  const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
  const txs = xs.map(family.toX), tys = ys.map(family.toY);
  const lineFit = Regression.fitRobustPolynomial(txs, tys, 1, errDist.best.name, nu);
  return family.fromLineFit(lineFit);
}

function genNoisy(rand, trueFn, frac, scale, skew) {
  return xsPositive.map(x => {
    const base = trueFn(x);
    const sd = 0.03 * Math.abs(base) + 1;
    const isOutlier = rand() < frac;
    if (!isOutlier) return Math.max(1e-6, base + randNormal(rand) * sd);
    const outMag = sd * scale;
    return Math.max(1e-6, base + (skew ? (outMag + rand() * outMag) : randNormal(rand) * outMag));
  });
}

// --- Build the synthetic pool, pooling all three families together (same
// shared ROBUST_REFIT_DELTA_BIC_THRESHOLD constant governs all of them in
// regression.js, so one shared threshold search is what's actually usable).
function buildSyntheticPool() {
  const pool = [];
  const fractions = [0, 0.02, 0.05, 0.08, 0.10, 0.15];
  const scales = [3, 5, 8, 12, 20];
  let seedCounter = 0;
  for (const family of FAMILIES) {
    for (const frac of fractions) {
      for (const scale of scales) {
        for (const skew of [false, true]) {
          for (let t = 0; t < 15; t++) {
            seedCounter++;
            const rand = mulberry32(300000 + seedCounter * 7919);
            const ys = genNoisy(rand, family.trueFn, frac, scale, skew);
            let olsFit;
            try { olsFit = family.fitFn(xsPositive, ys); } catch (e) { continue; }
            const tResiduals = xsPositive.map((x, i) => family.toY(ys[i]) - family.toY(olsFit.predict(x)));
            const errDist = Regression.chooseErrorDistribution(tResiduals);
            if (!errDist) continue;
            const deltaBIC = errDist.best.name === 'Gaussian' ? 0 : errDist.deltaBICvsGaussian;
            if (deltaBIC <= 0) { pool.push({ deltaBIC: 0, diff: 0 }); continue; }
            const refit = ungatedRefit(family, xsPositive, ys, errDist);
            const diff = paramError(refit, family.trueA, family.trueB) - paramError(olsFit, family.trueA, family.trueB);
            pool.push({ deltaBIC, diff, kind: family.kind });
          }
        }
      }
    }
  }
  return pool;
}

// --- Real-data pool: try fitting each family where the domain allows (x>0
// for power/logarithmic, y>0 for exponential/power), using the actual
// gate's own held-out R^2 as the outcome (same convention as the polynomial
// threshold test: sign-flipped so negative = helped).
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

    for (const family of FAMILIES) {
      if (family.kind !== 'exponential' && trainXs.some(x => x <= 0)) continue;
      if (trainYs.some(y => y <= 0)) continue;
      let olsFit;
      try { olsFit = family.fitFn(trainXs, trainYs); } catch (e) { continue; }
      const tResiduals = trainXs.map((x, i) => family.toY(trainYs[i]) - family.toY(olsFit.predict(x)));
      const errDist = Regression.chooseErrorDistribution(tResiduals);
      if (!errDist) continue;
      const deltaBIC = errDist.best.name === 'Gaussian' ? 0 : errDist.deltaBICvsGaussian;
      if (deltaBIC <= 0) { pool.push({ deltaBIC: 0, diff: 0, kind: family.kind, title: group.title }); continue; }
      if (testXs.some(x => x <= 0) || testYs.some(y => y <= 0)) continue; // can't score R^2 on this family's scale otherwise
      let refit;
      try { refit = ungatedRefit(family, trainXs, trainYs, errDist); } catch (e) { continue; }
      const olsR2 = Regression.computeR2(testXs, testYs, x => olsFit.predict(x));
      const refitR2 = Regression.computeR2(testXs, testYs, refit.predict);
      pool.push({ deltaBIC, diff: -(refitR2 - olsR2), kind: family.kind, title: group.title });
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

function analyzePool(name, pool) {
  console.log(`\n########## ${name} (${pool.length} cases) ##########`);
  console.log(`  non-Gaussian-flagged cases: ${pool.filter(p => p.deltaBIC > 0).length}/${pool.length}`);
  const optBoot = bootstrapOptimalThreshold(pool, GRID);
  console.log(`  bootstrapped optimal ΔBIC threshold: median=${optBoot.median.toFixed(2)}  95% CI [${optBoot.lower.toFixed(2)}, ${optBoot.upper.toFixed(2)}]`);
  const pointOptimalT = findOptimalThreshold(pool, GRID);
  console.log(`  point-estimate optimal threshold on full pool: ${pointOptimalT}`);
  const variants = [['optimal', pointOptimalT], ['10% tighter', pointOptimalT * 1.1], ['20% tighter', pointOptimalT * 1.2], ['30% tighter', pointOptimalT * 1.3]];
  for (const [label, T] of variants) {
    const nb = netBenefitAt(pool, T);
    const meanNb = meanNetBenefitAt(pool, T);
    const ci = bootstrapNetBenefitCI(pool, T);
    const triggered = pool.filter(p => p.deltaBIC >= T).length;
    const nbStr = isFinite(nb) ? nb.toFixed(4).padStart(9) : '  (n<5)'.padStart(9);
    const ciStr = ci.unstable ? '[unstable]' : `[${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`;
    const verdict = ci.unstable ? '  -> not enough data' : ci.upper < 0 ? '  -> confirmed net positive' : ci.lower > 0 ? '  -> confirmed net NEGATIVE' : '  -> not reliably different from 0';
    console.log(`  T=${T.toFixed(2).padStart(7)} (${label.padEnd(12)}) triggers ${String(triggered).padStart(4)}/${pool.length}  median=${nbStr}  mean=${meanNb.toFixed(4).padStart(9)}  95%CI ${ciStr}${verdict}`);
  }
}

const inputPath = path.resolve(process.argv[2] || 'data.txt');
analyzePool('LOG-LINEAR SYNTHETIC (exponential+power+logarithmic pooled)', buildSyntheticPool());
analyzePool('LOG-LINEAR REAL DATA (pooled, sign-flipped R^2)', buildRealPool(inputPath));
