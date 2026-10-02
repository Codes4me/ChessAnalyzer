// Does spike-and-slab's chosen model beat the DISCRETE polynomial models it
// overlaps with, on real held-out (extrapolation) R^2 — not just in-sample
// fit? "Overlaps with" matters here: the live Auto-selection pool
// (model_candidates.js) no longer includes cubic/quartic/quintic or even
// plain linear (degree 1) — only 'constant' and 'quadratic (degree 2)'
// remain from the polynomial family (see that file's comments: cubic/
// quartic/quintic were removed as "bootstrap-confirmed dead weight", plain
// linear removed 2026-09-30). Spike-and-slab's basis {x, x^2, x^3, x^4}
// strictly contains both of those as 2 of its 16 subsets, plus 14 the live
// tool doesn't consider at all. So the real question is whether letting
// spike-and-slab pick from that bigger menu — using ONLY training data to
// choose, same as any real selection procedure must — generalizes better
// than the live tool's 2-way {constant, quadratic} choice, not just fits
// the training data better (a bigger menu trivially can't do worse
// in-sample; the earlier cubic/quartic removal was ALSO an in-sample
// winner that lost on held-out data, which is exactly the failure mode
// this test is built to catch if it's happening again here).
const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');

const BASIS = ['x', 'x^2', 'x^3', 'x^4'];
const EXCLUDED_TITLES = ['Movie Gross']; // sort-order artifact, see spike_and_slab_test.js

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

function allSubsets(items) {
  const subsets = [[]];
  for (const item of items) {
    const n = subsets.length;
    for (let i = 0; i < n; i++) subsets.push([...subsets[i], item]);
  }
  return subsets;
}

function rowFor(subset, x) {
  return subset.map(term => {
    if (term === 'x') return x;
    if (term === 'x^2') return x * x;
    if (term === 'x^3') return x * x * x;
    return x * x * x * x;
  });
}

// Fits a subset on TRAIN data only and returns a predict function — used
// both to fit the final chosen model and, for spike-and-slab, to score
// every candidate subset before picking (so the pick itself never sees test
// data, same as any real selection procedure must).
function fitSubset(trainXs, trainYs, subset) {
  if (subset.length === 0) {
    const mean = trainYs.reduce((s, y) => s + y, 0) / trainYs.length;
    return { predict: () => mean, r2: 0 };
  }
  const X = trainXs.map(x => rowFor(subset, x));
  return Regression.fitMultiLinear(X, trainYs);
}

// Spike-and-slab pick: exact FLS g-prior enumeration over all 16 subsets,
// TRAIN DATA ONLY (same formula as spike_and_slab_test.js's original pass).
function spikeSlabPick(trainXs, trainYs) {
  const n = trainXs.length;
  const g = n;
  const subsets = allSubsets(BASIS);
  let best = null, bestLogBF = -Infinity;
  for (const subset of subsets) {
    const fit = fitSubset(trainXs, trainYs, subset);
    const r2 = Math.max(0, fit.r2 || 0);
    const p = subset.length;
    const logBF = p === 0 ? 0 : ((n - 1 - p) / 2) * Math.log(1 + g) - ((n - 1) / 2) * Math.log(1 + g * (1 - r2));
    if (logBF > bestLogBF) { bestLogBF = logBF; best = subset; }
  }
  return best;
}

// Discrete pick: the live tool's actual overlapping candidates — constant
// and quadratic (degree 2) ONLY, chosen by held-out extrapolation R^2 on
// a fit/check split within the training data itself (mimicking
// selectByExtrapolation's own methodology), exactly like the real pipeline
// would pick between them.
function discretePick(trainXs, trainYs) {
  const fitLen = Math.floor(trainXs.length * 0.7);
  if (fitLen < 3 || trainXs.length - fitLen < 2) {
    // too little training data to sub-split for selection — fall back to
    // whichever has better in-sample R^2, same degenerate case the real
    // tool would hit.
    const candidates = [[], ['x'], ['x', 'x^2']].map(s => ({ subset: s, fit: fitSubset(trainXs, trainYs, s) }));
    candidates.sort((a, b) => (b.fit.r2 || 0) - (a.fit.r2 || 0));
    return candidates[0].subset;
  }
  const fXs = trainXs.slice(0, fitLen), fYs = trainYs.slice(0, fitLen);
  const cXs = trainXs.slice(fitLen), cYs = trainYs.slice(fitLen);
  const candidates = [[], ['x', 'x^2']]; // constant, quadratic — the live pool's actual overlap
  let best = null, bestCheckR2 = -Infinity;
  for (const subset of candidates) {
    const fit = fitSubset(fXs, fYs, subset);
    const checkR2 = Regression.computeR2(cXs, cYs, x => fit.predict(subset.length ? rowFor(subset, x) : x));
    if (checkR2 > bestCheckR2) { bestCheckR2 = checkR2; best = subset; }
  }
  return best;
}

function heldOutR2(subset, trainXs, trainYs, testXs, testYs) {
  const fit = fitSubset(trainXs, trainYs, subset);
  return Regression.computeR2(testXs, testYs, x => fit.predict(subset.length ? rowFor(subset, x) : x));
}

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
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

function main() {
  const inputPath = path.resolve(process.argv[2] || 'data.txt');
  const rawText = fs.readFileSync(inputPath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);

  const rows = [];
  for (const group of groups) {
    const title = group.title || '(untitled)';
    if (EXCLUDED_TITLES.some(ex => title.includes(ex))) continue;
    const xs = group.columns[0], ys = group.columns[1];
    const valid = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
    const vxs = xs.filter((_, i) => valid[i]), vys = ys.filter((_, i) => valid[i]);
    if (vxs.length < 15 || new Set(vxs).size < 6) continue;

    const trainLen = Math.floor(vxs.length * 0.7);
    const testLen = vxs.length - trainLen;
    if (trainLen < 6 || testLen < 4) continue; // need enough for a quartic+intercept on train, and a meaningful test

    const trainXs = vxs.slice(0, trainLen), trainYs = vys.slice(0, trainLen);
    const testXs = vxs.slice(trainLen), testYs = vys.slice(trainLen);

    try {
      const ssSubset = spikeSlabPick(trainXs, trainYs);
      const discSubset = discretePick(trainXs, trainYs);
      const ssR2 = heldOutR2(ssSubset, trainXs, trainYs, testXs, testYs);
      const discR2 = heldOutR2(discSubset, trainXs, trainYs, testXs, testYs);
      rows.push({ title, n: vxs.length, ssSubset, discSubset, ssR2, discR2, diff: ssR2 - discR2 });
    } catch (e) { /* skip datasets a subset fit chokes on (singular design, etc.) */ }
  }

  console.log(`Spike-and-slab vs. the live tool's overlapping discrete models (constant, quadratic), ${rows.length} datasets, held-out R^2 on a 70/30 split\n`);
  rows.forEach(r => {
    const ssLabel = r.ssSubset.length ? r.ssSubset.join('+') : '(intercept only)';
    const discLabel = r.discSubset.length ? r.discSubset.join('+') : '(intercept only)';
    console.log(`  ${r.title.slice(0, 38).padEnd(38)} n=${String(r.n).padStart(4)}  spike-slab: ${ssLabel.padEnd(14)} R2=${r.ssR2.toFixed(3).padStart(7)}   discrete: ${discLabel.padEnd(10)} R2=${r.discR2.toFixed(3).padStart(7)}   diff=${r.diff.toFixed(3)}`);
  });

  const diffs = rows.map(r => r.diff);
  const ci = bootstrapMedianCI(diffs);
  console.log(`\nMedian held-out R^2 difference (spike-slab minus discrete): ${median(diffs).toFixed(4)}`);
  console.log(`Mean held-out R^2 difference (outlier-sensitive, reported for reference): ${mean(diffs).toFixed(4)}`);
  console.log(`Dataset-level bootstrap 95% CI on the median difference (2000 resamples): ${ci.lower.toFixed(4)} to ${ci.upper.toFixed(4)}`);
  console.log(ci.lower > 0
    ? '-> CI excludes 0 (positive): spike-and-slab\'s pick genuinely generalizes better on this corpus.'
    : ci.upper < 0
    ? '-> CI excludes 0 (negative): the discrete {constant, quadratic} pick genuinely generalizes BETTER — spike-and-slab\'s bigger menu is overfitting on held-out data.'
    : '-> CI includes 0: no reliable evidence either beats the other on held-out R^2 for this corpus.');

  const ssWon = rows.filter(r => r.diff > 0).length;
  const discWon = rows.filter(r => r.diff < 0).length;
  const tied = rows.length - ssWon - discWon;
  console.log(`\nHead-to-head: spike-slab better on ${ssWon}/${rows.length}, discrete better on ${discWon}/${rows.length}, tied on ${tied}/${rows.length}`);
}

main();
