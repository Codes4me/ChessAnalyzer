// Synthetic-ground-truth version of the outlier-resistance check. The
// real-corpus version (moretests_outlier_resistance_test.js) can only ask
// "does this flag survive trimming" — it can't tell whether an evaporated
// flag was a FALSE POSITIVE correctly fixed, or REAL signal destroyed by
// over-aggressive trimming, because real data's ground truth is unknown.
// Synthetic data separates these cleanly into two scenarios:
//
//   (A) TRUE NULL + outliers: no real autocorrelation/heteroscedasticity
//       at all, just IID Gaussian noise plus a few contaminating points.
//       Any significant flag here is BY DEFINITION a false positive caused
//       only by the outliers — so the IDEAL evaporation rate is 100%.
//       Anything less reveals genuine outlier-vulnerability beyond what
//       the fence removes.
//   (B) GENUINE effect + outliers: real moderate autocorrelation (for the
//       3 autocorrelation tests) or real heteroscedasticity (for the 3
//       heteroscedasticity tests), with the same outlier contamination on
//       top. Here the IDEAL evaporation rate is LOW — the real structure
//       should still be detectable after trimming. A high evaporation
//       rate here means trimming is destroying genuine signal, not just
//       fixing false positives — a different and arguably worse failure
//       mode than outlier-vulnerability itself.
const Regression = require('./regression.js');
const { boxPierce, runsTest, glejserTest, spearmanHetero } = require('./moretests.js');

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
function robustSigma(values) { const m = median(values); const mad = median(values.map(v => Math.abs(v - m))); return 1.4826 * mad; }
function dropOutliers(xs, residuals) {
  const m = median(residuals);
  const sigma = robustSigma(residuals);
  if (sigma === 0) return { xs, residuals, dropped: 0 };
  const fence = 1.5 * sigma;
  const keepXs = [], keepRes = [];
  let dropped = 0;
  for (let i = 0; i < residuals.length; i++) {
    if (Math.abs(residuals[i] - m) > fence) { dropped++; continue; }
    keepXs.push(xs[i]); keepRes.push(residuals[i]);
  }
  return { xs: keepXs, residuals: keepRes, dropped };
}

const N = 60;
const TRUE_A = 3, TRUE_B = 2;
const xs = Array.from({ length: N }, (_, i) => i + 1);

// Contamination: 10% of points get a large additive bump (symmetric, same
// convention as every other contamination scenario in this investigation).
function contaminate(rand, baseYs) {
  return baseYs.map(v => {
    const out = rand() < 0.1;
    return v + (out ? randNormal(rand) * 15 * 4 : 0); // 15x the base sigma=4
  });
}

function genNullPlusOutliers(rand) {
  const ys = xs.map(x => TRUE_A + TRUE_B * x + randNormal(rand) * 4);
  return contaminate(rand, ys);
}
function genAR1PlusOutliers(rand, phi) {
  const e = new Array(N); e[0] = randNormal(rand) * 4;
  const innovSd = 4 * Math.sqrt(Math.max(1e-9, 1 - phi * phi));
  for (let t = 1; t < N; t++) e[t] = phi * e[t - 1] + randNormal(rand) * innovSd;
  const ys = xs.map((x, i) => TRUE_A + TRUE_B * x + e[i]);
  return contaminate(rand, ys);
}
function genHeteroPlusOutliers(rand, strength) {
  const rawSd = xs.map(x => 1 + strength * (x / N));
  const meanRawVar = rawSd.reduce((s, v) => s + v * v, 0) / N;
  const scale = 4 / Math.sqrt(meanRawVar);
  const ys = xs.map((x, i) => TRUE_A + TRUE_B * x + randNormal(rand) * rawSd[i] * scale);
  return contaminate(rand, ys);
}

function fitLine(xs, ys) { return Regression.runRegression(xs, ys, { degree: 1 }); }

function runCheck(name, pValueFn, genYs, trials, seedBase) {
  let flagged = 0, evaporated = 0;
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(rand);
    let fit; try { fit = fitLine(xs, ys); } catch (e) { continue; }
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    const h = Math.max(1, Math.min(10, Math.floor(N / 5)));
    let pBefore;
    try { pBefore = pValueFn({ xs, residuals, h }); } catch (e) { continue; }
    if (!(pBefore < 0.05)) continue;
    flagged++;
    const { xs: txs, residuals: tres } = dropOutliers(xs, residuals);
    if (tres.length < 8) continue;
    const trimmedH = Math.max(1, Math.min(10, Math.floor(tres.length / 5)));
    let pAfter;
    try { pAfter = pValueFn({ xs: txs, residuals: tres, h: trimmedH }); } catch (e) { continue; }
    if (!(pAfter < 0.05)) evaporated++;
  }
  const pct = flagged ? (100 * evaporated / flagged).toFixed(1) : 'n/a';
  console.log(`  ${name.padEnd(24)} flagged ${flagged}/${trials}  evaporated ${evaporated}/${flagged} (${pct}%)`);
}

const AC_TESTS = {
  'Durbin-Watson': d => Regression.durbinWatsonTest(d.residuals).pValue,
  'Box-Pierce': d => boxPierce(d.residuals, d.h).pValue,
  'Runs test': d => runsTest(d.residuals).pValue,
};
const HETERO_TESTS = {
  "White's test": d => Regression.whiteTest(d.xs, d.residuals).pValue,
  'Glejser': d => glejserTest(d.xs, d.residuals).pValue,
  'Spearman-rank hetero': d => spearmanHetero(d.xs, d.residuals).pValue,
};

const TRIALS = 500;

console.log('=== SCENARIO A: TRUE NULL + outliers (every flag here is a false positive; ideal evaporation = 100%) ===\n');
for (const [name, fn] of Object.entries(AC_TESTS)) runCheck(name, fn, genNullPlusOutliers, TRIALS, 1000);
for (const [name, fn] of Object.entries(HETERO_TESTS)) runCheck(name, fn, genNullPlusOutliers, TRIALS, 2000);

console.log('\n=== SCENARIO B: GENUINE autocorrelation (phi=0.6) + outliers (real signal; ideal evaporation = LOW) ===\n');
for (const [name, fn] of Object.entries(AC_TESTS)) runCheck(name, fn, rand => genAR1PlusOutliers(rand, 0.6), TRIALS, 3000);

console.log('\n=== SCENARIO C: GENUINE heteroscedasticity (strength=6) + outliers (real signal; ideal evaporation = LOW) ===\n');
for (const [name, fn] of Object.entries(HETERO_TESTS)) runCheck(name, fn, rand => genHeteroPlusOutliers(rand, 6), TRIALS, 4000);
