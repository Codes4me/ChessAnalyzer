// Same stress-test methodology as error_dist_refit_stress_test.js (which
// found no looseness problem in the polynomial gate's ΔBIC>=45), applied to
// the log-linear family's own per-family thresholds (exponential=95,
// power=145, logarithmic=110) against 5 distribution shapes none of the
// original sweeps tested: model misspecification, smooth heteroscedasticity
// with no discrete outliers, high-frequency moderate contamination, extreme
// pure heavy tails, and bimodal noise.
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

const N = 60;
const xsPositive = Array.from({ length: N }, (_, i) => i + 1);

const FAMILIES = {
  exponential: {
    threshold: 95, fitFn: Regression.fitExponential, trueA: 5, trueB: 0.03, trueFn: x => 5 * Math.exp(0.03 * x),
    toX: x => x, toY: y => Math.log(y),
    fromLineFit: lf => { const a = Math.exp(lf.coeffs[0]), b = lf.coeffs[1]; return { a, b, predict: x => a * Math.exp(b * x) }; },
  },
  power: {
    threshold: 145, fitFn: Regression.fitPower, trueA: 2, trueB: 1.4, trueFn: x => 2 * Math.pow(x, 1.4),
    toX: x => Math.log(x), toY: y => Math.log(y),
    fromLineFit: lf => { const a = Math.exp(lf.coeffs[0]), b = lf.coeffs[1]; return { a, b, predict: x => a * Math.pow(x, b) }; },
  },
  logarithmic: {
    threshold: 110, fitFn: Regression.fitLogarithmic, trueA: 4, trueB: 10, trueFn: x => 4 * Math.log(x) + 10,
    toX: x => Math.log(x), toY: y => y,
    fromLineFit: lf => { const b = lf.coeffs[0], a = lf.coeffs[1]; return { a, b, predict: x => a * Math.log(x) + b }; },
  },
};

function paramError(fit, trueA, trueB) { return (fit.a - trueA) ** 2 + (fit.b - trueB) ** 2; }

function runScenario(family, name, genYs, trials = 400, seedBase = 1000) {
  let flaggedCount = 0, triggeredCount = 0, degenerateCount = 0;
  const triggeredDiffs = [];
  const names = {};
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(rand);
    let olsFit;
    try { olsFit = family.fitFn(xsPositive, ys); } catch (e) { continue; }
    const tResiduals = xsPositive.map((x, i) => family.toY(ys[i]) - family.toY(olsFit.predict(x)));
    const errDist = Regression.chooseErrorDistribution(tResiduals);
    if (!errDist) continue;
    const nonGaussian = errDist.best.name !== 'Gaussian';
    if (nonGaussian) { flaggedCount++; names[errDist.best.name] = (names[errDist.best.name] || 0) + 1; }
    if (!nonGaussian || errDist.deltaBICvsGaussian < family.threshold) continue;
    triggeredCount++;
    let refit;
    try {
      const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
      const txs = xsPositive.map(family.toX), tys = ys.map(family.toY);
      const lineFit = Regression.fitRobustPolynomial(txs, tys, 1, errDist.best.name, nu);
      refit = family.fromLineFit(lineFit);
    } catch (e) { continue; }
    const diff = paramError(refit, family.trueA, family.trueB) - paramError(olsFit, family.trueA, family.trueB);
    if (isFinite(diff)) triggeredDiffs.push(diff); else degenerateCount++;
  }
  const ci = bootstrapMedianCI(triggeredDiffs);
  const verdict = ci.unstable ? 'too few triggered cases to judge'
    : ci.upper < 0 ? 'refit HELPS at this threshold'
    : ci.lower > 0 ? `*** refit HURTS — ${family.threshold} may be TOO LOOSE here ***`
    : 'no reliable effect at this threshold';
  console.log(`\n=== ${name} (threshold=${family.threshold}) ===`);
  console.log(`  flagged non-Gaussian: ${flaggedCount}/${trials} ${JSON.stringify(names)}`);
  console.log(`  triggered (ΔBIC>=${family.threshold}): ${triggeredCount}/${trials}` + (degenerateCount ? `  (${degenerateCount} degenerate refits)` : ''));
  if (triggeredDiffs.length) {
    console.log(`  among triggered: median diff=${median(triggeredDiffs).toFixed(4)}  mean=${mean(triggeredDiffs).toFixed(4)}` +
      (ci.unstable ? '' : `  95% CI [${ci.lower.toFixed(4)}, ${ci.upper.toFixed(4)}]`));
  }
  console.log(`  -> ${verdict}`);
}

function genWithNoise(family, noiseFn) {
  return rand => xsPositive.map(x => Math.max(1e-6, noiseFn(rand, x, family.trueFn(x))));
}

for (const [kind, family] of Object.entries(FAMILIES)) {
  console.log(`\n########## ${kind.toUpperCase()} ##########`);

  // 1. Model misspecification: a small extra wiggle the plain form can't capture.
  runScenario(family, `${kind}: model misspecification (small extra x^2 wiggle)`,
    genWithNoise(family, (rand, x, base) => base + 0.001 * base * x + randNormal(rand) * (0.03 * Math.abs(base) + 1)));

  // 2. Smooth heteroscedasticity, no discrete outliers (noise scales up faster than the base model already implies).
  runScenario(family, `${kind}: smooth heteroscedasticity (sd grows faster than base)`,
    genWithNoise(family, (rand, x, base) => base + randNormal(rand) * (0.15 * x + 1)));

  // 3. High-frequency moderate contamination (30%, moderate scale).
  runScenario(family, `${kind}: high-frequency moderate contamination (30%, 5x)`,
    genWithNoise(family, (rand, x, base) => {
      const sd = 0.03 * Math.abs(base) + 1;
      const out = rand() < 0.3;
      return base + randNormal(rand) * (out ? sd * 5 : sd);
    }));

  // 4. Extremely heavy pure tails.
  runScenario(family, `${kind}: pure Student-t df=2 (extreme tails)`,
    genWithNoise(family, (rand, x, base) => base + randStudentT(rand, 2) * (0.02 * Math.abs(base) + 1)));
  runScenario(family, `${kind}: pure Cauchy (df=1, undefined variance)`,
    genWithNoise(family, (rand, x, base) => base + randStudentT(rand, 1) * (0.015 * Math.abs(base) + 1)));

  // 5. Bimodal noise: 50/50 two clusters.
  runScenario(family, `${kind}: bimodal noise (50/50, offset +/- 3sd)`,
    genWithNoise(family, (rand, x, base) => {
      const sd = 0.03 * Math.abs(base) + 1;
      return base + (rand() < 0.5 ? 3 * sd : -3 * sd) + randNormal(rand) * sd;
    }));
}
