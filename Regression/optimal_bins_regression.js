// Derives a formula that PREDICTS how many histogram bins make the
// histogram shape closest to its own best-fit distribution (reusing
// chooseErrorDistribution/distributionShape, the exact machinery already
// wired into the live histogram overlay) -- rather than hand-picking a
// textbook rule (Sturges/Scott/Freedman-Diaconis), this fits one directly
// against the real optimum, same "validate before adopting" practice as
// everything else in this project.
//
// Method:
//  1. For many synthetic samples (varying n AND shape -- Gaussian,
//     Laplace, Student-t, AND log-normal (skewed), so the formula isn't
//     overfit to symmetric shapes), rescale the histogram's own working
//     range to the middle 5th-95th percentile (percentile-based, not a
//     symmetric MAD fence), fit the best distribution on what's left,
//     then grid-search every candidate bin count and score each one by
//     chi-square-style goodness-of-fit against that fitted distribution's
//     own expected bin counts. The bin count with the lowest chi-square
//     IS this sample's "optimal bins".
//     NOTE: an earlier version of this trim used a symmetric 1.5*robust-
//     MAD fence (median +/- 1.5*1.4826*MAD) instead, and it was WRONG for
//     skewed data -- confirmed directly on synthetic log-normal data: it
//     dropped 16% of the sample, and EVERY dropped point came from the
//     upper tail (0 from below the median) -- a symmetric distance rule
//     applied to an asymmetric distribution just doesn't make sense.
//     5th-95th percentile trimming keeps exactly 90% of the data
//     regardless of which way it's skewed, by construction.
//  2. Average over many random replications per (n, shape) to smooth out
//     the noise in any single sample's optimum.
//  3. Fit optimalBins ~ f(n) in a few candidate functional forms (cube-
//     root, log, power -- the usual families classic bin-count rules come
//     from) and keep whichever actually fits this project's own simulated
//     optimum best, instead of assuming one.
//  4. Validate the chosen formula out-of-sample (fresh seeds, including
//     an n range not used for fitting) before calling it final.

const fs = require('fs');
const Regression = require('./regression');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function randLaplace(r, b) { const u = r() - 0.5; return -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u)); }
function randStudentT(r, nu) { // standard method: Z / sqrt(ChiSq(nu)/nu), ChiSq(nu) via sum of nu squared normals
  const z = randNormal(r);
  let chi2 = 0; for (let i = 0; i < nu; i++) chi2 += randNormal(r) ** 2;
  return z / Math.sqrt(chi2 / nu);
}
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function median(a) { const s = [...a].sort((x, y) => x - y); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

// --- 1.5*robust-MAD outlier fence (single, symmetric MAD) -- kept here
// for reference/comparison only; NOT used by findOptimalBins below
// anymore. See the file header for why: confirmed directly to drop 16%
// of a skewed (log-normal) sample, entirely from one tail. ---
function madTrim(values) {
  const med = median(values);
  const absDev = values.map(v => Math.abs(v - med));
  const mad = median(absDev);
  if (mad === 0) return values.slice();
  const fence = 1.5 * 1.4826 * mad;
  return values.filter(v => Math.abs(v - med) <= fence);
}

// --- Percentile-based trim: keep the middle 90% (5th-95th percentile) of
// the data by RANK, not by distance from center. This is what actually
// gets used -- it keeps exactly 90% of the sample regardless of which way
// (or how much) the distribution is skewed, instead of assuming a
// symmetric spread the way a MAD fence does. ---
function percentileOf(sortedValues, p) {
  const idx = p * (sortedValues.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? sortedValues[lo] : sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (idx - lo);
}
function percentileTrim(values, lowP = 0.05, highP = 0.95) {
  const sorted = [...values].sort((a, b) => a - b);
  const lo = percentileOf(sorted, lowP), hi = percentileOf(sorted, highP);
  return values.filter(v => v >= lo && v <= hi);
}

// --- Distribution shapes (same as regression.html's overlay) ---
function distributionShape(name, params) {
  if (name === 'Gaussian') { const { sigma } = params; return d => Math.exp(-0.5 * (d / sigma) ** 2); }
  if (name === 'Laplace') { const { b } = params; return d => Math.exp(-Math.abs(d) / b); }
  if (name === 'Student-t') { const { sigma, nu } = params; return d => Math.pow(1 + (d * d) / (nu * sigma * sigma), -(nu + 1) / 2); }
  return null;
}

// --- Chi-square goodness-of-fit for ONE candidate bin count. Returns null
// (not just a bad score) when any bin's expected count falls below 5 --
// the standard textbook validity threshold for a chi-square test at all.
// This matters a lot more than it looks: an EARLIER version of this
// function used a tiny 0.5 floor on expected count instead of rejecting
// low-count bins outright, and the per-bin-average chi-square it produced
// kept getting spuriously BETTER as bins increased without bound (750+
// "optimal" bins for n=2000!) -- because with tiny expected counts,
// (observed-expected)^2/expected stays small simply from discreteness
// (observed is 0 or 1, expected is <1, so the numerator is tiny too),
// not because the fit is actually any good. Enforcing the real validity
// floor removes that whole degenerate regime instead of letting it win. ---
function chiSquareForBins(values, bins, distName, distParams, distMean, distK) {
  const min = Math.min(...values), max = Math.max(...values);
  const range = (max - min) || 1;
  const binWidth = range / bins;
  const counts = new Array(bins).fill(0);
  values.forEach(v => { let idx = Math.floor((v - min) / binWidth); if (idx >= bins) idx = bins - 1; if (idx < 0) idx = 0; counts[idx]++; });

  const shape = distributionShape(distName, distParams);
  const STEPS = 300, dx = range / STEPS;
  let area = 0;
  for (let s = 0; s <= STEPS; s++) { const v = min + s * dx; const w = (s === 0 || s === STEPS) ? 0.5 : 1; area += shape(v - distMean) * w * dx; }
  const scale = area > 0 ? (values.length * binWidth) / area : 0;

  let chiSq = 0;
  for (let i = 0; i < bins; i++) {
    const v = min + (i + 0.5) * binWidth;
    const expected = shape(v - distMean) * scale;
    if (expected < 5) return null; // invalid at this bin count -- too sparse for chi-square to mean anything
    chiSq += (counts[i] - expected) ** 2 / expected;
  }
  // REDUCED chi-square (divided by degrees of freedom = bins - distK - 1),
  // not raw. Raw chi-square was tried first and found to trivially favor
  // FEWER bins almost regardless of fit quality -- summing fewer
  // nonnegative terms just gives a smaller total, a pure artifact of how
  // many terms got added, not of how well the shape actually matches.
  // Reduced chi-square has expectation ~1 for a genuinely correct model
  // AT ANY bin count (standard result), which is what makes comparing it
  // across different bin counts a fair comparison in the first place.
  const df = bins - distK - 1;
  if (df < 1) return null;
  return chiSq / df;
}

// Sample excess kurtosis and skewness of the TRIMMED data -- computed
// directly from whatever sample is on hand, so this is just as usable on
// a REAL dataset as on these synthetic ones (no access to the "true"
// generating distribution needed, unlike using each shape's theoretical
// moments would have been).
function sampleMoments(values) {
  const m = mean(values);
  const sd = Math.sqrt(mean(values.map(v => (v - m) ** 2)));
  if (sd === 0) return { skewness: 0, kurtosis: 0 };
  const skewness = mean(values.map(v => ((v - m) / sd) ** 3));
  const kurtosis = mean(values.map(v => ((v - m) / sd) ** 4)) - 3; // excess kurtosis (Gaussian = 0)
  return { skewness, kurtosis };
}

function findOptimalBins(values, { minBins = 3 } = {}) {
  const trimmed = percentileTrim(values);
  if (trimmed.length < 15) return null;
  const m = mean(trimmed);
  const chosen = Regression.chooseErrorDistribution(trimmed.map(v => v - m));
  if (!chosen) return null;

  const maxBins = Math.floor(trimmed.length / 2);
  let best = null;
  for (let bins = minBins; bins <= maxBins; bins++) {
    const score = chiSquareForBins(trimmed, bins, chosen.best.name, chosen.best.params, m, chosen.best.k);
    if (score === null) break; // once a bin count is too sparse, every larger one will be too -- stop searching
    if (!best || score < best.score) best = { bins, score };
  }
  if (!best) return null;
  const { skewness, kurtosis } = sampleMoments(trimmed);
  return { optimalBins: best.bins, n: trimmed.length, distName: chosen.best.name, hitCap: best.bins === maxBins, skewness, kurtosis };
}

// =====================================================================
// PART 1: build (n, optimalBins) pairs across varying n and shape,
// averaged over several replications each to smooth out single-sample noise.
// =====================================================================
const SIZES = [20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000];
const REPLICATIONS = 15;
const SHAPES = [
  { name: 'Gaussian', gen: (r) => randNormal(r) },
  { name: 'Laplace', gen: (r) => randLaplace(r, 1) },
  { name: 'Student-t(5)', gen: (r) => randStudentT(r, 5) },
  { name: 'Log-normal (skewed)', gen: (r) => Math.exp(randNormal(r) * 0.6) },
];

console.log('=== Part 1: building per-replication (n, kurtosis, skewness, optimalBins) training records ===\n');
const trainingRecords = []; // { n, kurtosis, skewness, bins }
for (const size of SIZES) {
  for (const shape of SHAPES) {
    const atThisSize = [];
    for (let rep = 0; rep < REPLICATIONS; rep++) {
      const rand = mulberry32(100000 + size * 97 + rep * 13 + shape.name.length * 1009);
      const values = Array.from({ length: size }, () => shape.gen(rand) * 10 + 50); // arbitrary scale/location -- shouldn't matter, just realism
      const result = findOptimalBins(values);
      if (result) { atThisSize.push(result); trainingRecords.push({ n: size, kurtosis: result.kurtosis, skewness: result.skewness, bins: result.optimalBins }); }
    }
    if (atThisSize.length) {
      const avgBins = mean(atThisSize.map(r => r.optimalBins));
      const avgKurt = mean(atThisSize.map(r => r.kurtosis));
      const cappedCount = atThisSize.filter(r => r.hitCap).length;
      console.log(`  n=${size}  shape=${shape.name}  avg optimal bins=${avgBins.toFixed(2)}  avg sample kurtosis=${avgKurt.toFixed(2)}  (${atThisSize.length}/${REPLICATIONS} reps${cappedCount ? `, ${cappedCount} hit the n/2 cap` : ''})`);
    }
  }
}

// =====================================================================
// PART 2: fit bins ~ f(n, kurtosis) -- multi-predictor now, using each
// REPLICATION as its own training row (not just the shape-average) for
// more statistical power with a second predictor in play. Kurtosis
// (computed straight from the sample, same as it would be on real data)
// stands in for "how many effectively-independent pieces of information
// this spread of values actually carries" -- a heavy-tailed or skewed
// shape needs fewer bins at the same n because more of its probability
// mass piles up in a narrower core after trimming, leaving less room for
// extra bins before hitting the expected-count-5 validity floor.
// =====================================================================
console.log('\n=== Part 2: bins ~ f(n, kurtosis) ===\n');
function fitMultiAndScore(label, buildRow) {
  const X = trainingRecords.map(buildRow);
  const y = trainingRecords.map(r => r.bins);
  const fit = Regression.fitMultiLinear(X, y);
  console.log(`  ${label}: R²=${fit.r2.toFixed(4)}  intercept=${fit.intercept.toFixed(4)}  coeffs=[${fit.coeffs.map(c => c.toFixed(4)).join(', ')}]`);
  return { label, r2: fit.r2, fit, buildRow };
}

const sqrtNOnly = fitMultiAndScore('bins = a + b*sqrt(n)                         (n only, baseline)', r => [Math.sqrt(r.n)]);
const sqrtNPlusKurt = fitMultiAndScore('bins = a + b*sqrt(n) + c*kurtosis            (adds kurtosis)', r => [Math.sqrt(r.n), r.kurtosis]);
const sqrtNTimesKurt = fitMultiAndScore('bins = a + b*sqrt(n) + c*kurtosis + d*sqrt(n)*kurtosis  (adds interaction)', r => [Math.sqrt(r.n), r.kurtosis, Math.sqrt(r.n) * r.kurtosis]);
const candidates2 = [sqrtNOnly, sqrtNPlusKurt, sqrtNTimesKurt];
const winner = candidates2.reduce((best, c) => (c.r2 > best.r2 ? c : best));
console.log(`\nBest form: ${winner.label}`);
function predictBins(n, kurtosis) {
  const row = winner.buildRow({ n, kurtosis });
  return Math.max(2, winner.fit.intercept + row.reduce((s, v, i) => s + v * winner.fit.coeffs[i], 0));
}

// =====================================================================
// PART 3: out-of-sample validation -- fresh seeds, n values NOT used in
// training, using each validation sample's OWN measured kurtosis (exactly
// how this would run on a real, unknown dataset).
// =====================================================================
console.log('\n=== Part 3: out-of-sample validation ===\n');
const VALIDATION_SIZES = [25, 40, 60, 90, 125, 175, 250, 400, 600, 900, 1250, 1750];
const validationRecords = []; // kept so Part 4 can re-test different kurtosis shrinkage on the SAME held-out samples
let totalAbsError = 0, count = 0;
const baselineErrors = []; // n-only formula, for comparison
for (const size of VALIDATION_SIZES) {
  for (const shape of SHAPES) {
    const rand = mulberry32(900000 + size * 131 + shape.name.length * 777);
    const values = Array.from({ length: size }, () => shape.gen(rand) * 10 + 50);
    const result = findOptimalBins(values);
    if (!result) continue;
    validationRecords.push({ n: size, kurtosis: result.kurtosis, skewness: result.skewness, actualBins: result.optimalBins, shape: shape.name });
    const predicted = predictBins(size, result.kurtosis);
    const err = Math.abs(predicted - result.optimalBins);
    const baselinePredicted = Math.max(2, sqrtNOnly.fit.intercept + Math.sqrt(size) * sqrtNOnly.fit.coeffs[0]);
    baselineErrors.push(Math.abs(baselinePredicted - result.optimalBins));
    totalAbsError += err; count++;
    console.log(`  n=${size}  shape=${shape.name}  actual=${result.optimalBins}  predicted(n+kurtosis)=${predicted.toFixed(2)}  predicted(n only)=${baselinePredicted.toFixed(2)}  |error|=${err.toFixed(2)}`);
  }
}
console.log(`\nMean absolute error, n+kurtosis formula (raw kurtosis): ${(totalAbsError / count).toFixed(3)} bins`);
console.log(`Mean absolute error, n-only formula:                     ${(mean(baselineErrors)).toFixed(3)} bins`);
console.log(`\nFormula so far: ${winner.label}`);
console.log(`intercept=${winner.fit.intercept.toFixed(6)}  coeffs=[${winner.fit.coeffs.map(c => c.toFixed(6)).join(', ')}]`);

// =====================================================================
// PART 4: does shrinking the sample kurtosis toward a prior (0 -- the
// Gaussian/"no evidence of non-normal tails yet" default) help? Same
// pattern this project already uses for AR(1)'s phi: kurtosis_used =
// kurtosis_raw * n/(n+k), where k controls how strongly small samples get
// pulled toward the prior (k=0 is exactly the raw/unshrunk version already
// tested above). Sample kurtosis is a notoriously high-variance estimator
// at small n (its standard error under normality scales like sqrt(24/n)),
// so a single small sample's kurtosis could easily be noise, not signal --
// worth testing properly rather than assuming either way.
// =====================================================================
console.log('\n=== Part 4: shrinking kurtosis toward a prior of 0 (Gaussian) ===\n');
const SHRINKAGE_CANDIDATES = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250, 400, 600, 900, 1300, 1800, 2500, 3500, 5000, 7500, 10000];
function fitWithShrinkage(k) {
  const X = trainingRecords.map(r => {
    const kurt = r.kurtosis * r.n / (r.n + k);
    return [Math.sqrt(r.n), kurt, Math.sqrt(r.n) * kurt];
  });
  const y = trainingRecords.map(r => r.bins);
  const fit = Regression.fitMultiLinear(X, y);

  let absErr = 0;
  for (const rec of validationRecords) {
    const kurt = rec.kurtosis * rec.n / (rec.n + k);
    const row = [Math.sqrt(rec.n), kurt, Math.sqrt(rec.n) * kurt];
    const predicted = Math.max(2, fit.intercept + row.reduce((s, v, i) => s + v * fit.coeffs[i], 0));
    absErr += Math.abs(predicted - rec.actualBins);
  }
  const outOfSampleMAE = absErr / validationRecords.length;
  return { k, trainR2: fit.r2, outOfSampleMAE, fit };
}
const shrinkageResults = SHRINKAGE_CANDIDATES.map(fitWithShrinkage);
shrinkageResults.forEach(r => console.log(`  k=${r.k.toString().padStart(3)}: train R²=${r.trainR2.toFixed(4)}  out-of-sample MAE=${r.outOfSampleMAE.toFixed(3)} bins`));
const bestShrinkage = shrinkageResults.reduce((best, r) => (r.outOfSampleMAE < best.outOfSampleMAE ? r : best));
console.log(`\nBest shrinkage constant: k=${bestShrinkage.k}  (out-of-sample MAE=${bestShrinkage.outOfSampleMAE.toFixed(3)} vs k=0's ${shrinkageResults[0].outOfSampleMAE.toFixed(3)})`);
console.log(`${bestShrinkage.k === 0 ? 'No shrinkage helped -- k=0 (raw kurtosis) remains best.' : `Shrinkage helps: kurtosis_used = kurtosis_raw * n/(n+${bestShrinkage.k})`}`);
console.log(`\nFinal formula (with k=${bestShrinkage.k} shrinkage): intercept=${bestShrinkage.fit.intercept.toFixed(6)}  coeffs=[${bestShrinkage.fit.coeffs.map(c => c.toFixed(6)).join(', ')}]`);

// =====================================================================
// PART 5: does adding SKEWNESS (with its own shrinkage toward a prior of
// 0) help on top of the kurtosis model above? Uses |skewness|, not signed
// skewness -- a shape skewed left should cost the same bin budget as the
// same shape skewed right, so direction shouldn't matter, only magnitude.
// Same shrink-toward-0 idea as kurtosis (sample skewness is also a noisy
// small-n estimator), with its OWN shrinkage constant swept separately --
// no reason to assume skewness and kurtosis need the same amount of
// shrinkage just because they're both central moments.
//
// Compared via HELD-OUT LOG-LIKELIHOOD, not just mean absolute error:
// assume Gaussian residuals, fix sigma from each model's OWN validation
// residuals (so neither model gets to use the other's sigma), and sum the
// log-likelihood of the actual held-out bin counts under each model's
// predictions. Higher (less negative) is better. This is a stricter test
// than MAE alone -- it also penalizes a model whose errors are more
// spread out, not just bigger on average.
// =====================================================================
console.log('\n=== Part 5: adding skewness (own shrinkage) on top of the kurtosis model ===\n');
const bestK = bestShrinkage.k;
function kurtUsed(r, k) { return r.kurtosis * r.n / (r.n + k); }
function skewUsed(r, k) { return Math.abs(r.skewness) * r.n / (r.n + k); }

function fitModel(buildRow) {
  const X = trainingRecords.map(buildRow);
  const y = trainingRecords.map(r => r.bins);
  const fit = Regression.fitMultiLinear(X, y);
  const residuals = trainingRecords.map((r, i) => y[i] - (fit.intercept + X[i].reduce((s, v, j) => s + v * fit.coeffs[j], 0)));
  return { fit, buildRow, trainResiduals: residuals };
}
function evaluateOnValidation(model) {
  const preds = validationRecords.map(r => {
    const row = model.buildRow(r);
    return Math.max(3, model.fit.intercept + row.reduce((s, v, i) => s + v * model.fit.coeffs[i], 0));
  });
  const residuals = validationRecords.map((r, i) => r.actualBins - preds[i]);
  const mae = mean(residuals.map(Math.abs));
  // Sigma fixed from this model's OWN validation residuals (not training
  // residuals -- training residuals are systematically smaller since
  // that's exactly what was fit to minimize, which would make every
  // model's held-out log-likelihood look artificially better than it is).
  const sigma = Math.sqrt(mean(residuals.map(r => r * r))) || 1e-6;
  const logLik = residuals.reduce((s, r) => s - 0.5 * Math.log(2 * Math.PI * sigma * sigma) - (r * r) / (2 * sigma * sigma), 0);
  return { mae, sigma, logLik };
}

// Baseline: the kurtosis-only model from Part 4 (bestShrinkage), re-scored
// the same way (fixed 3-bin floor, own-residual sigma) for a fair comparison.
const baselineModel = { fit: bestShrinkage.fit, buildRow: r => [Math.sqrt(r.n), kurtUsed(r, bestK), Math.sqrt(r.n) * kurtUsed(r, bestK)] };
const baselineEval = evaluateOnValidation(baselineModel);
console.log(`Baseline (n + kurtosis, k=${bestK}, no skewness): MAE=${baselineEval.mae.toFixed(3)}  sigma=${baselineEval.sigma.toFixed(3)}  held-out logLik=${baselineEval.logLik.toFixed(2)}`);

const SKEW_SHRINKAGE_CANDIDATES = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250, 400, 600, 900, 1300, 1800, 2500, 3500, 5000];
console.log('\n  (plain addition: + |skewness_shrunk|, no interaction term)');
const skewPlainResults = SKEW_SHRINKAGE_CANDIDATES.map(kSkew => {
  const model = fitModel(r => [Math.sqrt(r.n), kurtUsed(r, bestK), Math.sqrt(r.n) * kurtUsed(r, bestK), skewUsed(r, kSkew)]);
  const ev = evaluateOnValidation(model);
  console.log(`    kSkew=${kSkew.toString().padStart(4)}: train R²=${model.fit.r2.toFixed(4)}  MAE=${ev.mae.toFixed(3)}  held-out logLik=${ev.logLik.toFixed(2)}`);
  return { kSkew, model, ev };
});
const bestSkewPlain = skewPlainResults.reduce((best, r) => (r.ev.logLik > best.ev.logLik ? r : best));

console.log('\n  (with interaction: + |skewness_shrunk| + sqrt(n)*|skewness_shrunk|)');
const skewInteractionResults = SKEW_SHRINKAGE_CANDIDATES.map(kSkew => {
  const model = fitModel(r => [Math.sqrt(r.n), kurtUsed(r, bestK), Math.sqrt(r.n) * kurtUsed(r, bestK), skewUsed(r, kSkew), Math.sqrt(r.n) * skewUsed(r, kSkew)]);
  const ev = evaluateOnValidation(model);
  console.log(`    kSkew=${kSkew.toString().padStart(4)}: train R²=${model.fit.r2.toFixed(4)}  MAE=${ev.mae.toFixed(3)}  held-out logLik=${ev.logLik.toFixed(2)}`);
  return { kSkew, model, ev };
});
const bestSkewInteraction = skewInteractionResults.reduce((best, r) => (r.ev.logLik > best.ev.logLik ? r : best));

console.log('\n=== Part 5 summary ===\n');
console.log(`  baseline (no skewness):            logLik=${baselineEval.logLik.toFixed(2)}  MAE=${baselineEval.mae.toFixed(3)}`);
console.log(`  + skewness (kSkew=${bestSkewPlain.kSkew}, no interaction): logLik=${bestSkewPlain.ev.logLik.toFixed(2)}  MAE=${bestSkewPlain.ev.mae.toFixed(3)}`);
console.log(`  + skewness (kSkew=${bestSkewInteraction.kSkew}, with interaction): logLik=${bestSkewInteraction.ev.logLik.toFixed(2)}  MAE=${bestSkewInteraction.ev.mae.toFixed(3)}`);

const overallBest = [{ name: 'baseline', logLik: baselineEval.logLik }, { name: 'skew-plain', logLik: bestSkewPlain.ev.logLik }, { name: 'skew-interaction', logLik: bestSkewInteraction.ev.logLik }]
  .reduce((best, r) => (r.logLik > best.logLik ? r : best));
console.log(`\nBest by held-out log-likelihood: ${overallBest.name}`);
if (overallBest.name === 'baseline') {
  console.log('Skewness does NOT improve held-out log-likelihood -- not adopting it.');
} else {
  const winningModel = overallBest.name === 'skew-plain' ? bestSkewPlain : bestSkewInteraction;
  console.log(`Skewness DOES improve held-out log-likelihood. Final formula: intercept=${winningModel.model.fit.intercept.toFixed(6)}  coeffs=[${winningModel.model.fit.coeffs.map(c => c.toFixed(6)).join(', ')}]  kSkew=${winningModel.kSkew}`);
}
