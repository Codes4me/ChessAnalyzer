// Does replacing the CURRENT candidate-selection rule (best CHECK-window
// R², with a tolerance band + "prefer the simplest contender" tie-break --
// see selectByExtrapolation's own comment in model_candidates.js) with a
// Laplace approximation of log-evidence (computed on FIT alone, no
// separate CHECK window needed -- the Occam/complexity penalty comes from
// the Hessian itself) would improve TRIMMED-AVERAGE TEST R² on unseen
// datasets?
//
// Method:
//  1. For each real dataset, replicate the CORE of selectByExtrapolation
//     (FIT/CHECK split, per-candidate checkR2, tolerance band, simplest-
//     wins tie-break among "contenders" -- candidates within tolerance of
//     the best). This intentionally skips the z_check/derivative safety
//     gates (edge-case guards, not the primary selection mechanism) --
//     documented simplification, not a secret rewrite of the real logic.
//  2. For the same candidates (minus ARIMA -- no clean Gaussian likelihood
//     for its differenced/integrated structure without much more custom
//     work; documented exclusion, not swept under the rug), compute a
//     Laplace log-evidence on FIT via the same numeric-Hessian machinery
//     already built and used in laplace_evidence_compare.js earlier this
//     project. Pick whichever candidate has the highest log-evidence.
//  3. Refit each rule's chosen candidate on FIT+CHECK, score on the real
//     TEST set via computeR2Sequential (walk-forward, same convention
//     backtest_models.js already uses for AR(1)/ARIMA).
//  4. Bootstrap over DATASETS (resample with replacement): for each
//     resample, compute the trimmed-average test R² (full-variance,
//     double-MAD fenced -- same metric backtest_models.js reports as its
//     headline number) under the CURRENT rule and under the LAPLACE rule.
//     Win-rate = fraction of resamples where Laplace's trimmed-average
//     beats Current's.
//  5. "A model's performance is judged on all datasets it was selected on
//     or close to being selected": for each candidate, report its outcome
//     (chosen vs. passed-over, and the resulting test R²) across every
//     dataset where it's either the winner OR a CONTENDER (within
//     tolerance of the best checkR2 -- the project's own existing
//     definition of "close"), under both rules.

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

// --- Laplace log-evidence machinery (same as laplace_evidence_compare.js) ---
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

// --- Per-candidate theta + log-likelihood builders. Direct-fit shapes
// (polynomial/exponential/power/logistic/robust-line) all reduce to
// "Gaussian residuals around predict(x,theta)" -- only AR(1)'s conditional
// (y_t | y_t-1) structure needs its own builder. ARIMA is deliberately
// NOT covered (see file header) -- laplace evidence for it returns null,
// excluding it from the Laplace-rule's contention on every dataset.
function buildLoglikAndTheta(name, fit, xs, ys) {
  const residualsOf = predict => xs.map((x, i) => ys[i] - predict(x));
  const sigmaFromResiduals = resid => Math.sqrt(mean(resid.map(r => r * r)));

  if (name === 'quadratic (degree 2)') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [...fit.coeffs, sigma]; // coeffs = [c0,c1,c2], predict(x) = c0+c1*x+c2*x^2
    const loglikFn = (t, xs2, ys2) => {
      const [c0, c1, c2, s] = t;
      const r = xs2.map((x, i) => ys2[i] - (c0 + c1 * x + c2 * x * x));
      return gaussianLoglik(r, s);
    };
    return { loglikFn, theta };
  }
  if (name === 'exponential') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.a, fit.b, sigma];
    const loglikFn = (t, xs2, ys2) => { const [a, b, s] = t; const r = xs2.map((x, i) => ys2[i] - a * Math.exp(b * x)); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'exponential with offset') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.a, fit.b, fit.c, sigma];
    const loglikFn = (t, xs2, ys2) => { const [a, b, c, s] = t; const r = xs2.map((x, i) => ys2[i] - (a * Math.exp(b * x) + c)); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'power') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.a, fit.b, sigma];
    const loglikFn = (t, xs2, ys2) => { const [a, b, s] = t; const r = xs2.map((x, i) => ys2[i] - a * Math.pow(x, b)); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'power with offset') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.a, fit.b, fit.c, sigma];
    const loglikFn = (t, xs2, ys2) => { const [a, b, c, s] = t; const r = xs2.map((x, i) => ys2[i] - (a * Math.pow(x, b) + c)); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'S-curve (logistic)') {
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.a, fit.b, fit.c, sigma];
    const loglikFn = (t, xs2, ys2) => { const [a, b, c, s] = t; const r = xs2.map((x, i) => ys2[i] - a / (1 + Math.exp(-(b * x + c)))); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'Theil-Sen (robust line)' || name === 'RANSAC (robust line)') {
    // CAVEAT: these weren't fit by maximizing Gaussian likelihood (that's
    // the whole point of a robust estimator), so their (slope,intercept)
    // isn't the true MLE the Laplace approximation assumes it's expanding
    // around. Treated here as "the evidence FOR this specific line, under
    // a Gaussian-noise assumption" -- a defensible but approximate reuse,
    // not a claim that these were MLE-fit.
    const resid = residualsOf(fit.predict);
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.slope, fit.intercept, sigma];
    const loglikFn = (t, xs2, ys2) => { const [sl, ic, s] = t; const r = xs2.map((x, i) => ys2[i] - (sl * x + ic)); return gaussianLoglik(r, s); };
    return { loglikFn, theta };
  }
  if (name === 'AR(1)') {
    // Conditional likelihood of y_t | y_t-1 (not a function of x at all) --
    // AR(1)'s own natural likelihood, not a reuse of the curve-fit shape.
    const yLag = ys.slice(0, -1), yNow = ys.slice(1);
    const resid = yNow.map((y, i) => y - (fit.c + fit.phi * yLag[i]));
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.c, fit.phi, sigma];
    const loglikFn = (t, xsIgnored, ysSeries) => {
      const [c, phi, s] = t;
      const lag = ysSeries.slice(0, -1), now = ysSeries.slice(1);
      const r = now.map((y, i) => y - (c + phi * lag[i]));
      return gaussianLoglik(r, s);
    };
    return { loglikFn, theta, usesFullSeriesNotXY: true };
  }
  return null; // ARIMA and anything else unlisted: excluded from the Laplace rule
}

// --- Replica of selectByExtrapolation's CORE logic (FIT/CHECK, tolerance
// band, simplest-wins tie-break) -- deliberately WITHOUT the z_check/
// derivative safety gates (edge-case guards, not what's being tested
// here). Returns per-candidate checkR2 (for "contenders"/"close to
// selected"), the current-rule winner, and the Laplace-rule winner. ---
function evaluateSelectionRules(xs, ys) {
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
    let laplaceEvidence = null;
    const built = buildLoglikAndTheta(cand.name, fit, fitXs, fitYs);
    if (built) {
      try { laplaceEvidence = laplaceLogEvidence(built.loglikFn, built.theta, fitXs, fitYs); } catch (e) { laplaceEvidence = null; }
      if (!Number.isFinite(laplaceEvidence)) laplaceEvidence = null;
    }
    rows.push({ candidate: cand, checkR2, laplaceEvidence });
  }
  if (!rows.length) return null;

  const toleranceConstant = 3, maxTolerance = 0.48;
  const effectiveTolerance = Math.min(toleranceConstant / checkLen, maxTolerance);
  const bestCheckR2 = Math.max(...rows.map(r => r.checkR2));
  const contenders = rows.filter(r => r.checkR2 >= bestCheckR2 - effectiveTolerance); // "close to selected", current rule's own definition
  const currentWinner = contenders.reduce((simplest, r) => (r.candidate.params < simplest.candidate.params ? r : simplest));

  const laplaceEligible = rows.filter(r => r.laplaceEvidence !== null);
  const laplaceWinner = laplaceEligible.length
    ? laplaceEligible.reduce((best, r) => (r.laplaceEvidence > best.laplaceEvidence ? r : best))
    : null;

  return { rows, contenders, currentWinner, laplaceWinner, fitLen, checkLen };
}

// =====================================================================
// Build per-dataset results: both rules' chosen candidate, refit on
// FIT+CHECK (train pool), scored on the real TEST set.
// =====================================================================
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));

function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  const trainLen = Math.floor(n * 0.7);
  const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
  if (testXs.length < 2 || trainLen < 10) return null;
  const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);

  const sel = evaluateSelectionRules(trainXs, trainYs);
  if (!sel || !sel.laplaceWinner) return null;

  function scoreOnTest(candidate) {
    let fit; try { fit = candidate.fit(trainXs, trainYs); } catch (e) { return null; }
    return Regression.computeR2FullVarianceSequential(ys, trainYs[trainYs.length - 1], testXs, testYs, fit.predict);
  }
  const currentR2 = scoreOnTest(sel.currentWinner.candidate);
  const laplaceR2 = scoreOnTest(sel.laplaceWinner.candidate);
  if (currentR2 === null || laplaceR2 === null) return null;

  return {
    title: group.title, n,
    currentName: sel.currentWinner.candidate.name, currentR2,
    laplaceName: sel.laplaceWinner.candidate.name, laplaceR2,
    contenders: sel.contenders.map(r => r.candidate.name),
  };
}

const results = groups.map(evaluateDataset).filter(Boolean);
console.log(`Evaluated ${results.length} of ${groups.length} datasets.\n`);

console.log('=== Per-dataset: current rule (CHECK R² + tolerance + simplicity) vs. Laplace rule (log-evidence on FIT) ===\n');
let agree = 0, disagree = 0;
results.forEach(r => {
  const same = r.currentName === r.laplaceName;
  if (same) agree++; else disagree++;
  if (!same) {
    console.log(`  ${r.title}  n=${r.n}`);
    console.log(`    current:  ${r.currentName}  (test R²=${r.currentR2.toFixed(4)})`);
    console.log(`    laplace:  ${r.laplaceName}  (test R²=${r.laplaceR2.toFixed(4)})`);
  }
});
console.log(`\nRules agreed on ${agree}/${results.length} datasets, disagreed on ${disagree}/${results.length}.\n`);

// =====================================================================
// Bootstrap over datasets: trimmed-average test R² under each rule.
// =====================================================================
console.log('=== Bootstrap: does the Laplace rule beat the current rule on trimmed-average test R² (full-variance)? ===\n');
const BOOT = 3000;
const currentAverages = [], laplaceAverages = [];
let laplaceWins = 0, currentWins = 0, ties = 0;
for (let b = 0; b < BOOT; b++) {
  const rand = mulberry32(2024000 + b * 131);
  const resample = Array.from({ length: results.length }, () => results[Math.floor(rand() * results.length)]);
  const currentAvg = trimmedAverage(resample.map(r => r.currentR2));
  const laplaceAvg = trimmedAverage(resample.map(r => r.laplaceR2));
  if (currentAvg === null || laplaceAvg === null) continue;
  currentAverages.push(currentAvg); laplaceAverages.push(laplaceAvg);
  if (laplaceAvg > currentAvg + 1e-9) laplaceWins++;
  else if (currentAvg > laplaceAvg + 1e-9) currentWins++;
  else ties++;
}
console.log(`Point estimate (full dataset, no resampling): current trimmed-avg=${trimmedAverage(results.map(r => r.currentR2)).toFixed(4)}  laplace trimmed-avg=${trimmedAverage(results.map(r => r.laplaceR2)).toFixed(4)}`);
console.log(`Bootstrap (${BOOT} resamples): Laplace rule's trimmed-average was higher in ${laplaceWins} (${(100 * laplaceWins / BOOT).toFixed(1)}%), current rule higher in ${currentWins} (${(100 * currentWins / BOOT).toFixed(1)}%), tied in ${ties}.`);
console.log(`Mean trimmed-average across resamples: current=${mean(currentAverages).toFixed(4)}  laplace=${mean(laplaceAverages).toFixed(4)}`);

// =====================================================================
// Per-candidate breakdown: outcome across every dataset it was selected
// on OR was a contender (close to selected) under the CURRENT rule.
// =====================================================================
console.log('\n=== Per-candidate: outcome across every dataset it was selected on or was a contender for (current rule\'s own "close" definition) ===\n');
const byCandidate = {};
results.forEach(r => {
  r.contenders.forEach(name => {
    byCandidate[name] = byCandidate[name] || { contenderCount: 0, wonCurrent: 0, wonLaplace: 0, r2WhenWonCurrent: [], r2WhenWonLaplace: [] };
    byCandidate[name].contenderCount++;
    if (r.currentName === name) { byCandidate[name].wonCurrent++; byCandidate[name].r2WhenWonCurrent.push(r.currentR2); }
    if (r.laplaceName === name) { byCandidate[name].wonLaplace++; byCandidate[name].r2WhenWonLaplace.push(r.laplaceR2); }
  });
});
Object.entries(byCandidate).sort((a, b) => b[1].contenderCount - a[1].contenderCount).forEach(([name, stats]) => {
  const avgCur = stats.r2WhenWonCurrent.length ? mean(stats.r2WhenWonCurrent).toFixed(3) : 'n/a';
  const avgLap = stats.r2WhenWonLaplace.length ? mean(stats.r2WhenWonLaplace).toFixed(3) : 'n/a';
  console.log(`  ${name}: contender on ${stats.contenderCount} datasets -- won under CURRENT rule ${stats.wonCurrent}x (avg test R²=${avgCur}), won under LAPLACE rule ${stats.wonLaplace}x (avg test R²=${avgLap})`);
});
