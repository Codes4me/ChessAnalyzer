// Extends laplace_pool_selection_bootstrap.js: that script could only
// compute real Laplace log-evidence for 6 of the 10 CANDIDATES (anything
// with a clean Gaussian likelihood) -- ARIMA and the two robust-line
// methods (Theil-Sen, RANSAC) were excluded from the Laplace rule
// entirely, which handicapped it unfairly on any dataset where one of
// those would otherwise have won.
//
// Fix: within each dataset, collect every PAIR of "anchor" candidates
// (the 6 with real evidence) and record (deltaCheckR2, deltaLogEvidence)
// between them -- same dataset, same n, so the n-dependent baseline terms
// mostly cancel. Fit a plain regression through these pairs: deltaEvidence
// ~ deltaCheckR2. That gives a conversion factor from a checkR2 GAP to an
// evidence GAP. For ARIMA/Theil-Sen/RANSAC (no real evidence), anchor each
// one to whichever real-evidence candidates are present on the SAME
// dataset, convert its checkR2 gap to those anchors into a predicted
// evidence gap, and average the resulting proxy evidence estimates --
// giving every candidate an evidence-or-proxy score on the same footing,
// so picking the single best across ALL of them is a fair comparison for
// the first time.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { CANDIDATES } = require('./model_candidates');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function median(nums) { const s = [...nums].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function doubleMADFence(nums, med, k = 1.5) {
  const below = nums.filter(v => v <= med).map(v => med - v);
  const above = nums.filter(v => v > med).map(v => v - med);
  const leftMAD = below.length >= 2 ? median(below) : 0;
  const rightMAD = above.length >= 2 ? median(above) : 0;
  const lower = leftMAD > 0 ? med - k * 1.4826 * leftMAD : -Infinity;
  const upper = rightMAD > 0 ? med + k * 1.4826 * rightMAD : Infinity;
  return { lower, upper };
}
function trimmedAverage(nums) {
  if (!nums.length) return null;
  const med = median(nums);
  const { lower, upper } = doubleMADFence(nums, med);
  const kept = nums.filter(v => v >= lower && v <= upper);
  return kept.length ? mean(kept) : null;
}

// --- Laplace log-evidence machinery (same as before) ---
function numericHessian(f, theta) {
  const k = theta.length;
  const h = theta.map(v => Math.max(1e-6, Math.abs(v) * 1e-4));
  const H = Array.from({ length: k }, () => new Array(k).fill(0));
  const f0 = f(theta);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      if (i === j) {
        const tp = theta.slice(); tp[i] += h[i];
        const tm = theta.slice(); tm[i] -= h[i];
        H[i][i] = (f(tp) - 2 * f0 + f(tm)) / (h[i] * h[i]);
      } else {
        const tpp = theta.slice(); tpp[i] += h[i]; tpp[j] += h[j];
        const tpm = theta.slice(); tpm[i] += h[i]; tpm[j] -= h[j];
        const tmp = theta.slice(); tmp[i] -= h[i]; tmp[j] += h[j];
        const tmm = theta.slice(); tmm[i] -= h[i]; tmm[j] -= h[j];
        const v = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4 * h[i] * h[j]);
        H[i][j] = v; H[j][i] = v;
      }
    }
  }
  return H;
}
function determinant(M) {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  let det = 0;
  for (let col = 0; col < n; col++) {
    const minor = M.slice(1).map(row => row.filter((_, j) => j !== col));
    det += ((col % 2 === 0) ? 1 : -1) * M[0][col] * determinant(minor);
  }
  return det;
}
function laplaceLogEvidence(loglikFn, theta, xs, ys) {
  const k = theta.length;
  const negH = numericHessian((t) => -loglikFn(t, xs, ys), theta);
  const detH = determinant(negH);
  const ll = loglikFn(theta, xs, ys);
  if (!(detH > 0) || !Number.isFinite(detH) || !Number.isFinite(ll)) return -Infinity;
  return ll - 0.5 * Math.log(detH) + (k / 2) * Math.log(2 * Math.PI);
}
function gaussianLoglik(residuals, sigma) {
  if (sigma <= 0) return -Infinity;
  let ll = 0;
  for (const r of residuals) ll += -0.5 * Math.log(2 * Math.PI * sigma * sigma) - (r * r) / (2 * sigma * sigma);
  return ll;
}
function buildLoglikAndTheta(name, fit, xs, ys) {
  const residualsOf = predict => xs.map((x, i) => ys[i] - predict(x));
  const sigmaFromResiduals = resid => Math.sqrt(mean(resid.map(r => r * r)));
  if (name === 'quadratic (degree 2)') {
    const theta = [...fit.coeffs, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [c0, c1, c2, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - (c0 + c1 * x + c2 * x * x)), s); }, theta };
  }
  if (name === 'exponential') {
    const theta = [fit.a, fit.b, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [a, b, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - a * Math.exp(b * x)), s); }, theta };
  }
  if (name === 'exponential with offset') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [a, b, c, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - (a * Math.exp(b * x) + c)), s); }, theta };
  }
  if (name === 'power') {
    const theta = [fit.a, fit.b, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [a, b, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - a * Math.pow(x, b)), s); }, theta };
  }
  if (name === 'power with offset') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [a, b, c, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - (a * Math.pow(x, b) + c)), s); }, theta };
  }
  if (name === 'S-curve (logistic)') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t, xs2, ys2) => { const [a, b, c, s] = t; return gaussianLoglik(xs2.map((x, i) => ys2[i] - a / (1 + Math.exp(-(b * x + c)))), s); }, theta };
  }
  return null; // Theil-Sen/RANSAC/AR(1)/ARIMA: no "anchor" likelihood here -- these are exactly the ones getting a PROXY below instead
}
const ANCHOR_NAMES = new Set(['quadratic (degree 2)', 'exponential', 'exponential with offset', 'power', 'power with offset', 'S-curve (logistic)']);

function evaluateAllCandidates(xs, ys) {
  const n = xs.length;
  const fitLen = Math.floor(n * 0.5);
  const checkEnd = Math.floor(n * 0.7);
  const checkLen = checkEnd - fitLen;
  if (fitLen < 4 || checkLen < 2) return null;
  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen, checkEnd), checkYs = ys.slice(fitLen, checkEnd);

  const rows = [];
  for (const cand of CANDIDATES) {
    if (fitXs.length < cand.minFit) continue;
    let fit; try { fit = cand.fit(fitXs, fitYs); } catch (e) { continue; }
    const checkR2 = Regression.computeR2Sequential(fitYs[fitYs.length - 1], checkXs, checkYs, fit.predict);
    if (!Number.isFinite(checkR2)) continue;
    let evidence = null;
    if (ANCHOR_NAMES.has(cand.name)) {
      const built = buildLoglikAndTheta(cand.name, fit, fitXs, fitYs);
      if (built) { try { evidence = laplaceLogEvidence(built.loglikFn, built.theta, fitXs, fitYs); } catch (e) { evidence = null; } }
      if (!Number.isFinite(evidence)) evidence = null;
    }
    // Adjusted R² (the standard 1 - (1-R²)*(n-1)/(n-k-1) correction):
    // penalizes a model for how many parameters it spent to get that R²,
    // same spirit as log-evidence's own Occam factor -- raw checkR2 has no
    // such penalty at all, which may be why it barely predicted
    // deltaLogEvidence (R²=0.0003 in the first attempt at this proxy
    // regression). n here is checkLen (the window checkR2 was itself
    // measured on), not fitLen -- adjusting R² for the sample size IT was
    // computed on, which is what the standard formula adjusts for.
    const dof = checkLen - cand.params - 1;
    const adjustedCheckR2 = dof > 0 ? 1 - (1 - checkR2) * (checkLen - 1) / dof : checkR2; // falls back to raw checkR2 when there isn't enough room to adjust
    rows.push({ candidate: cand, checkR2, adjustedCheckR2, evidence, isAnchor: ANCHOR_NAMES.has(cand.name) });
  }
  if (!rows.length) return null;

  const toleranceConstant = 3, maxTolerance = 0.48;
  const effectiveTolerance = Math.min(toleranceConstant / checkLen, maxTolerance);
  const bestCheckR2 = Math.max(...rows.map(r => r.checkR2));
  const contenders = rows.filter(r => r.checkR2 >= bestCheckR2 - effectiveTolerance);
  const currentWinner = contenders.reduce((simplest, r) => (r.candidate.params < simplest.candidate.params ? r : simplest));

  return { rows, contenders, currentWinner, fitLen, checkLen };
}

// =====================================================================
// PASS 1: collect every real deltaCheckR2/deltaEvidence PAIR between two
// anchor candidates on the SAME dataset -- the regression's training data.
// =====================================================================
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));
const deltaCheckR2List = [], deltaEvidenceList = [];

const datasetSelections = []; // cache so PASS 2 doesn't redo the fitting work
groups.forEach(group => {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  const trainLen = Math.floor(n * 0.7);
  if (trainLen < 10 || n - trainLen < 2) return;
  const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);
  const sel = evaluateAllCandidates(trainXs, trainYs);
  if (!sel) return;
  datasetSelections.push({ group, trainXs, trainYs, testXs: xs.slice(trainLen), testYs: ys.slice(trainLen), sel });

  const anchors = sel.rows.filter(r => r.evidence !== null);
  for (let i = 0; i < anchors.length; i++) {
    for (let j = i + 1; j < anchors.length; j++) {
      deltaCheckR2List.push(anchors[i].adjustedCheckR2 - anchors[j].adjustedCheckR2);
      deltaEvidenceList.push(anchors[i].evidence - anchors[j].evidence);
    }
  }
});

console.log(`=== Comma-separated lists (${deltaCheckR2List.length} anchor-vs-anchor pairs, same dataset each time, using ADJUSTED checkR2) ===\n`);
console.log('deltaAdjustedCheckR2:');
console.log(deltaCheckR2List.map(v => v.toFixed(4)).join(','));
console.log('\ndeltaLogEvidence:');
console.log(deltaEvidenceList.map(v => v.toFixed(4)).join(','));

// =====================================================================
// Regression: deltaEvidence ~ deltaCheckR2 (plain OLS, with intercept --
// let the data say whether it actually passes through the origin rather
// than forcing it to).
// =====================================================================
function olsSimple(x, y) {
  const n = x.length;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = my - slope * mx;
  const predict = v => intercept + slope * v;
  const ssRes = x.reduce((s, v, i) => s + (y[i] - predict(v)) ** 2, 0);
  const r2 = syy === 0 ? 1 : 1 - ssRes / syy;
  return { slope, intercept, r2, predict };
}
const reg = olsSimple(deltaCheckR2List, deltaEvidenceList);
console.log(`\n=== Regression: deltaLogEvidence ≈ ${reg.slope.toFixed(2)} * deltaAdjustedCheckR2 ${reg.intercept >= 0 ? '+' : '-'} ${Math.abs(reg.intercept).toFixed(3)} ===`);
console.log(`R² of this proxy regression: ${reg.r2.toFixed(4)}  (n=${deltaCheckR2List.length} pairs)`);
console.log(`(intercept ${Math.abs(reg.intercept) < 0.2 ? 'is close to 0, as expected -- two identical candidates (deltaCheckR2=0) should have deltaEvidence=0' : 'is NOT close to 0 -- a real offset, not just noise around the expected origin'})`);

// =====================================================================
// PASS 2: give every non-anchor candidate (ARIMA, Theil-Sen, RANSAC,
// AR(1) -- anything without a real evidence value) a PROXY evidence:
// average, over every anchor present on the SAME dataset, of
// anchorEvidence + regression.predict(targetCheckR2 - anchorCheckR2).
// Then pick the single best-scoring candidate (real evidence OR proxy)
// across ALL of them -- the "universal" rule -- and compare it to the
// CURRENT rule the same way as before.
// =====================================================================
function evaluateUniversal(sel) {
  const anchors = sel.rows.filter(r => r.evidence !== null);
  const scored = sel.rows.map(r => {
    if (r.evidence !== null) return { row: r, score: r.evidence, isProxy: false };
    if (!anchors.length) return { row: r, score: null, isProxy: true };
    const estimates = anchors.map(a => a.evidence + reg.predict(r.adjustedCheckR2 - a.adjustedCheckR2));
    return { row: r, score: mean(estimates), isProxy: true };
  }).filter(s => s.score !== null && Number.isFinite(s.score));
  if (!scored.length) return null;
  return scored.reduce((best, s) => (s.score > best.score ? s : best));
}

const results = [];
datasetSelections.forEach(({ group, trainXs, trainYs, testXs, testYs, sel }) => {
  const universal = evaluateUniversal(sel);
  if (!universal) return;
  function scoreOnTest(candidate) {
    let fit; try { fit = candidate.fit(trainXs, trainYs); } catch (e) { return null; }
    return Regression.computeR2FullVarianceSequential([...trainYs, ...testYs], trainYs[trainYs.length - 1], testXs, testYs, fit.predict);
  }
  const currentR2 = scoreOnTest(sel.currentWinner.candidate);
  const universalR2 = scoreOnTest(universal.row.candidate);
  if (currentR2 === null || universalR2 === null) return;
  results.push({
    title: group.title, n: trainXs.length + testXs.length,
    currentName: sel.currentWinner.candidate.name, currentR2,
    universalName: universal.row.candidate.name, universalR2, universalIsProxy: universal.isProxy,
  });
});

console.log(`\n\n=== Per-dataset: current rule vs. UNIVERSAL rule (real evidence + proxy evidence for ARIMA/robust-line, all on one scale) ===\n`);
let agree = 0, disagree = 0;
results.forEach(r => {
  const same = r.currentName === r.universalName;
  if (same) agree++; else disagree++;
  if (!same) {
    console.log(`  ${r.title}  n=${r.n}`);
    console.log(`    current:   ${r.currentName}  (test R²=${r.currentR2.toFixed(4)})`);
    console.log(`    universal: ${r.universalName}${r.universalIsProxy ? ' [proxy]' : ' [real evidence]'}  (test R²=${r.universalR2.toFixed(4)})`);
  }
});
console.log(`\nRules agreed on ${agree}/${results.length}, disagreed on ${disagree}/${results.length}.\n`);

console.log('=== Bootstrap: does the UNIVERSAL rule beat the current rule on trimmed-average test R² (full-variance)? ===\n');
const BOOT = 3000;
const currentAverages = [], universalAverages = [];
let universalWins = 0, currentWins = 0, ties = 0;
for (let b = 0; b < BOOT; b++) {
  const rand = mulberry32(3033000 + b * 131);
  const resample = Array.from({ length: results.length }, () => results[Math.floor(rand() * results.length)]);
  const curAvg = trimmedAverage(resample.map(r => r.currentR2));
  const uniAvg = trimmedAverage(resample.map(r => r.universalR2));
  if (curAvg === null || uniAvg === null) continue;
  currentAverages.push(curAvg); universalAverages.push(uniAvg);
  if (uniAvg > curAvg + 1e-9) universalWins++;
  else if (curAvg > uniAvg + 1e-9) currentWins++;
  else ties++;
}
console.log(`Point estimate (no resampling): current trimmed-avg=${trimmedAverage(results.map(r => r.currentR2)).toFixed(4)}  universal trimmed-avg=${trimmedAverage(results.map(r => r.universalR2)).toFixed(4)}`);
console.log(`Bootstrap (${BOOT} resamples): universal rule higher in ${universalWins} (${(100 * universalWins / BOOT).toFixed(1)}%), current rule higher in ${currentWins} (${(100 * currentWins / BOOT).toFixed(1)}%), tied in ${ties}.`);
console.log(`Mean trimmed-average across resamples: current=${mean(currentAverages).toFixed(4)}  universal=${mean(universalAverages).toFixed(4)}`);
