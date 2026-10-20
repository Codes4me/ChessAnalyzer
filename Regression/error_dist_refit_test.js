// The one angle not yet checked on the error-distribution detector
// (chooseErrorDistribution in regression.js): does actually REFITTING a
// model's coefficients under its detected error distribution's loss —
// instead of always using OLS's squared-error loss — give a better FIT?
// Already checked and found no effect: CI width (error_dist_ci_realdata_check.js)
// and BIC-based model choice (the now-removed spike-and-slab robustModelBIC
// experiment). This is the third leg: does the loss function used to find
// the COEFFICIENTS THEMSELVES matter, on real held-out data.
//
// Method: for each dataset, fit OLS on a 70% train split, detect the best
// error distribution on train residuals. If it's Laplace, refit via IRLS
// toward an L1 (least-absolute-deviations) loss; if Student-t, refit via
// IRLS toward the Student-t MLE (down-weighting points far from the fit,
// same idea as the Student-t IRLS already used inside studentTErrorFit).
// Compare the REFIT's held-out R^2 against plain OLS's, on the same 30%
// test split. A dataset-level bootstrap on the median difference separates
// "flagged non-Gaussian" datasets (where a real effect should show up, if
// it exists at all) from "Gaussian-flagged" ones (expected to show nothing,
// a built-in control).
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

// Weighted least squares for y = a + b*x, closed form (2x2 normal
// equations) — the inner step of every IRLS iteration below.
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

// IRLS toward L1 (least absolute deviations) — the loss that matches a
// Laplace error distribution's MLE. Weight w_i = 1/max(|r_i|, floor);
// floor avoids blowing up when a point lands exactly on the line.
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

// IRLS toward the Student-t MLE — weight w_i = (nu+1)/(nu*sigma^2 + r_i^2),
// same form used inside regression.js's studentTErrorFit, alternating
// refitting the line and re-estimating sigma^2/weights.
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

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }

function bootstrapMedianCI(diffs, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  if (diffs.length < 3) return null;
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
    // Relaxed from the original >=2 ("positive evidence") bar to ANY
    // preference at all (deltaBIC>0) — the original bar left only 5
    // datasets with enough data for this test, too few to draw a
    // conclusion from either way.
    const nonGaussian = errDist.best.name !== 'Gaussian' && errDist.deltaBICvsGaussian > 0;

    let refit = null;
    if (nonGaussian && errDist.best.name === 'Laplace') refit = fitL1IRLS(trainXs, trainYs);
    else if (nonGaussian && errDist.best.name === 'Student-t') refit = fitStudentTIRLS(trainXs, trainYs, errDist.best.params.nu);
    else refit = olsFit; // Gaussian control: "refit" is just OLS again, diff should be ~0

    if (!refit) continue;

    const olsR2 = Regression.computeR2(testXs, testYs, x => olsFit.predict(x));
    const refitR2 = Regression.computeR2(testXs, testYs, refit.predict);
    const evidenceBand = !nonGaussian ? 'n/a'
      : errDist.deltaBICvsGaussian > 10 ? 'very strong'
      : errDist.deltaBICvsGaussian > 6 ? 'strong'
      : errDist.deltaBICvsGaussian > 2 ? 'positive'
      : 'weak';
    rows.push({ title: group.title || '(untitled)', n: vxs.length, nonGaussian, bestName: errDist.best.name, deltaBIC: errDist.deltaBICvsGaussian, evidenceBand, olsR2, refitR2, diff: refitR2 - olsR2 });
  }

  console.log(`Refit-under-detected-distribution vs. plain OLS, held-out R^2, ${rows.length} datasets\n`);
  rows.forEach(r => {
    const label = r.nonGaussian ? `${r.bestName}(${r.evidenceBand}, dBIC=${r.deltaBIC.toFixed(1)})` : 'gaussian';
    console.log(`  ${r.title.slice(0, 38).padEnd(38)} n=${String(r.n).padStart(4)}  ${label.padEnd(30)}  OLS=${r.olsR2.toFixed(3).padStart(7)}  refit=${r.refitR2.toFixed(3).padStart(7)}  diff=${r.diff.toFixed(3)}`);
  });

  for (const [label, subset] of [
    ['ALL non-Gaussian (any evidence)', rows.filter(r => r.nonGaussian)],
    ['  weak evidence only (0<dBIC<=2)', rows.filter(r => r.evidenceBand === 'weak')],
    ['  positive or better (dBIC>2)', rows.filter(r => r.nonGaussian && r.evidenceBand !== 'weak')],
    ['Gaussian (control)', rows.filter(r => !r.nonGaussian)],
  ]) {
    const diffs = subset.map(r => r.diff);
    console.log(`\n${label}: ${subset.length} datasets`);
    if (!diffs.length) { console.log('  (none)'); continue; }
    console.log(`  median diff = ${median(diffs).toFixed(4)}   mean diff (outlier-sensitive) = ${mean(diffs).toFixed(4)}`);
    const ci = bootstrapMedianCI(diffs);
    if (ci) {
      console.log(`  bootstrap 95% CI on median diff (2000 resamples): ${ci.lower.toFixed(4)} to ${ci.upper.toFixed(4)}`);
      console.log(ci.lower > 0 ? '  -> refitting under the detected distribution genuinely helps here'
        : ci.upper < 0 ? '  -> refitting under the detected distribution genuinely HURTS here'
        : '  -> no reliable effect either way');
    } else {
      console.log('  (too few datasets for a bootstrap CI)');
    }
  }
}

main();
