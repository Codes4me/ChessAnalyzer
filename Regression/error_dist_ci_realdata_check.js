// Does the error-distribution flag (chooseErrorDistribution in regression.js)
// actually predict a meaningful gap between the exact OLS CI and a
// bootstrap CI on this project's REAL data? The synthetic test already
// showed non-Gaussian errors alone don't break either CI's coverage in a
// controlled simulation — this checks the same question the other way,
// directly on the 51-dataset corpus, using the bootstrap CI as the
// non-parametric reference point (it never assumed Gaussian errors to begin
// with) instead of a known true value we don't have for real data.
//
// Per dataset: fit a line, flag Gaussian vs. not (same chooseErrorDistribution
// used on the live page), then compare the exact CI's width for the slope
// against a bootstrap CI's width for the same slope. If the error-distribution
// flag mattered, the two should diverge (bootstrap noticeably wider) more
// on the flagged datasets than the unflagged ones. A dataset-level bootstrap
// (resampling WHICH DATASETS, not which points — same technique as
// spike_and_slab_test.js's aggregate check) puts a confidence interval on
// that group difference instead of reporting a single noisy number.
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

function fitLine(xs, ys) { return Regression.runRegression(xs, ys, { degree: 1 }); }

function evaluate(xs, ys) {
  const fit = fitLine(xs, ys);
  if (!fit.ci || !fit.ci.coeffs[1]) return null;
  const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
  const errDist = Regression.chooseErrorDistribution(residuals);
  if (!errDist) return null;

  const exactSlope = fit.ci.coeffs[1];
  const exactWidth = exactSlope.upper - exactSlope.lower;

  const bootCi = Regression.bootstrapParameterCI(fitLine, xs, ys, { iterations: 300, seed: 0x5eed1234 });
  if (!bootCi || !bootCi.coeffs[1]) return null;
  const bootWidth = bootCi.coeffs[1].upper - bootCi.coeffs[1].lower;

  const flagged = errDist.best.name !== 'Gaussian' && errDist.deltaBICvsGaussian >= 2;
  return { flagged, bestName: errDist.best.name, deltaBIC: errDist.deltaBICvsGaussian, exactWidth, bootWidth, widthRatio: bootWidth / exactWidth };
}

function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// A plain mean of these ratios is exactly the outlier-sensitive statistic
// this project's own bootstrap methodology (see memory: trimmed-average R²
// instead of plain average) warns against — a near-degenerate fit (tiny
// residual variance, a handful of near-identical x values) can send a
// single dataset's ratio into the hundreds and swamp everything else.
// Median is the robust alternative used throughout; report both so the
// difference is visible.
function bootstrapGroupDiff(flaggedRatios, unflaggedRatios, stat, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const diffs = [];
  for (let i = 0; i < iterations; i++) {
    const fSample = Array.from({ length: flaggedRatios.length }, () => flaggedRatios[Math.floor(random() * flaggedRatios.length)]);
    const uSample = Array.from({ length: unflaggedRatios.length }, () => unflaggedRatios[Math.floor(random() * unflaggedRatios.length)]);
    diffs.push(stat(fSample) - stat(uSample));
  }
  diffs.sort((a, b) => a - b);
  return { lower: diffs[Math.floor(0.025 * (diffs.length - 1))], upper: diffs[Math.ceil(0.975 * (diffs.length - 1))] };
}

function main() {
  const inputPath = path.resolve(process.argv[2] || 'data.txt');
  const rawText = fs.readFileSync(inputPath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);

  const results = [];
  for (const group of groups) {
    const xs = group.columns[0], ys = group.columns[1];
    const valid = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
    const vxs = xs.filter((_, i) => valid[i]), vys = ys.filter((_, i) => valid[i]);
    if (vxs.length < 10) continue;
    try {
      const r = evaluate(vxs, vys);
      if (r) results.push({ title: group.title || '(untitled)', n: vxs.length, ...r });
    } catch (e) { /* skip fits that fail (degenerate X, etc.) */ }
  }

  console.log(`Exact-CI vs. bootstrap-CI slope width, ${results.length} real datasets\n`);
  results.forEach(r => {
    console.log(`  ${r.title.slice(0, 40).padEnd(40)} n=${String(r.n).padStart(4)}  ${r.flagged ? 'FLAGGED (' + r.bestName + ', dBIC=' + r.deltaBIC.toFixed(1) + ')' : 'gaussian-ok'.padEnd(24)}  exact=${r.exactWidth.toFixed(4)}  boot=${r.bootWidth.toFixed(4)}  ratio(boot/exact)=${r.widthRatio.toFixed(2)}`);
  });

  const flagged = results.filter(r => r.flagged);
  const unflagged = results.filter(r => !r.flagged);
  const flaggedRatios = flagged.map(r => r.widthRatio);
  const unflaggedRatios = unflagged.map(r => r.widthRatio);

  console.log(`\nFlagged non-Gaussian: ${flagged.length}/${results.length} datasets`);
  console.log(`  mean boot/exact width ratio,   flagged: ${mean(flaggedRatios).toFixed(3)}   unflagged: ${mean(unflaggedRatios).toFixed(3)}  (outlier-sensitive — see note below)`);
  console.log(`  median boot/exact width ratio, flagged: ${median(flaggedRatios).toFixed(3)}   unflagged: ${median(unflaggedRatios).toFixed(3)}`);
  console.log(`  (ratio > 1 means the bootstrap CI is WIDER than the exact CI for that dataset — i.e. the exact CI may be too narrow)`);

  if (flagged.length >= 3 && unflagged.length >= 3) {
    for (const [label, stat] of [['mean', mean], ['median', median]]) {
      const diffCI = bootstrapGroupDiff(flaggedRatios, unflaggedRatios, stat);
      const pointDiff = stat(flaggedRatios) - stat(unflaggedRatios);
      console.log(`\nDataset-level bootstrap (2000 resamples) on the ${label} group difference (flagged - unflagged):`);
      console.log(`  difference = ${pointDiff.toFixed(3)}  (95% CI ${diffCI.lower.toFixed(3)} to ${diffCI.upper.toFixed(3)})`);
      console.log(diffCI.lower > 0
        ? `  -> CI excludes 0: the error-distribution flag DOES predict a real, measurable exact-vs-bootstrap gap on this corpus.`
        : diffCI.upper < 0
        ? `  -> CI excludes 0 (negative direction, unexpected): flagged datasets actually show a SMALLER gap.`
        : `  -> CI includes 0: no reliable evidence the flag predicts a meaningfully different exact-vs-bootstrap gap on this corpus.`);
    }
  } else {
    console.log(`\nNot enough datasets in one group (need >=3 each) for a meaningful group-difference bootstrap.`);
  }
}

main();
