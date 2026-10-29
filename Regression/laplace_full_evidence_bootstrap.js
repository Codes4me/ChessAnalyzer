// Closes the gap the proxy-regression attempt (laplace_proxy_universal.js)
// couldn't: that approach tried to CONVERT checkR2 into an evidence
// estimate for ARIMA, and failed (R²=0.0001 -- checkR2 and log-evidence
// just don't have a linear relationship to exploit). This instead builds
// ARIMA's REAL conditional Gaussian likelihood directly, the same way
// AR(1)'s was already built in laplace_pool_selection_bootstrap.js --
// reusing the exact one-step residual machinery fitARIMA/fitARIMAAuto
// already compute internally (Hannan-Rissanen: a fixed long-AR proxy for
// e_t, then Gaussian residuals of the one-step forecast around it), just
// exposing it with an explicit theta=[c, phis..., thetas..., sigma] for
// the Laplace/Hessian treatment. With this, every one of the 10
// CANDIDATES (quadratic, exponential, exponential-offset, power,
// power-offset, S-curve, Theil-Sen, RANSAC, AR(1), ARIMA) has a REAL
// evidence value -- no proxy needed anywhere, and no candidate structurally
// excluded from contention.

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

// --- Laplace log-evidence machinery (unchanged from before) ---
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
function laplaceLogEvidence(loglikFn, theta) {
  const k = theta.length;
  const negH = numericHessian((t) => -loglikFn(t), theta);
  const detH = determinant(negH);
  const ll = loglikFn(theta);
  if (!(detH > 0) || !Number.isFinite(detH) || !Number.isFinite(ll)) return -Infinity;
  return ll - 0.5 * Math.log(detH) + (k / 2) * Math.log(2 * Math.PI);
}
function gaussianLoglik(residuals, sigma) {
  if (sigma <= 0) return -Infinity;
  let ll = 0;
  for (const r of residuals) ll += -0.5 * Math.log(2 * Math.PI * sigma * sigma) - (r * r) / (2 * sigma * sigma);
  return ll;
}
function sigmaFromResiduals(resid) { return Math.sqrt(mean(resid.map(r => r * r))); }
function differenceOnce(series) { const out = []; for (let t = 1; t < series.length; t++) out.push(series[t] - series[t - 1]); return out; }

// --- ARIMA's real conditional likelihood: rebuilds the exact Hannan-
// Rissanen structure fitARIMA uses internally (fixed long-AR e-proxy from
// a preliminary regression, then Gaussian one-step residuals around
// c + sum(phi*w_lag) + sum(theta*eProxy_lag)), exposed with
// theta=[c, phis..., thetas..., sigma] for the Hessian. ---
function buildArimaLoglik(fit, ys) {
  const { p, d, q } = fit;
  let series = ys.slice();
  for (let k = 0; k < d; k++) series = differenceOnce(series);
  const w = series;

  let eProxy = null, arOrderUsed = 0;
  if (q > 0) {
    arOrderUsed = Math.max(p + q + 2, Math.min(6, Math.floor(w.length / 3)));
    const yLagsLong = [], yTargetLong = [];
    for (let t = arOrderUsed; t < w.length; t++) { yTargetLong.push(w[t]); const row = []; for (let l = 1; l <= arOrderUsed; l++) row.push(w[t - l]); yLagsLong.push(row); }
    if (!yTargetLong.length) return null;
    const longAr = Regression.fitMultiLinear(yLagsLong, yTargetLong);
    eProxy = yTargetLong.map((y, i) => y - longAr.predict(yLagsLong[i]));
  }
  const start = q > 0 ? arOrderUsed + q : p;
  if (start >= w.length) return null;
  const target = w.slice(start);

  function oneStepAt(t, c, phis, thetas) {
    let v = c;
    for (let l = 1; l <= p; l++) v += phis[l - 1] * w[t - l];
    for (let l = 1; l <= q; l++) v += thetas[l - 1] * (eProxy ? eProxy[t - l - arOrderUsed] : 0);
    return v;
  }
  const residAtFit = target.map((_, i) => { const t = start + i; return w[t] - oneStepAt(t, fit.c, fit.phis, fit.thetas); });
  const sigma = sigmaFromResiduals(residAtFit);
  if (!(sigma > 0)) return null;
  const theta = [fit.c, ...fit.phis, ...fit.thetas, sigma];

  const loglikFn = (th) => {
    const c = th[0], phis = th.slice(1, 1 + p), thetas = th.slice(1 + p, 1 + p + q), s = th[th.length - 1];
    const resid = [];
    for (let i = 0; i < target.length; i++) { const t = start + i; resid.push(w[t] - oneStepAt(t, c, phis, thetas)); }
    return gaussianLoglik(resid, s);
  };
  return { loglikFn, theta };
}

// --- Per-candidate theta + log-likelihood builders -- identical to
// laplace_pool_selection_bootstrap.js, except ARIMA now has a real one
// (buildArimaLoglik above) instead of returning null. ---
function buildLoglikAndTheta(name, fit, xs, ys) {
  const residualsOf = predict => xs.map((x, i) => ys[i] - predict(x));

  if (name === 'quadratic (degree 2)') {
    const theta = [...fit.coeffs, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [c0, c1, c2, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - (c0 + c1 * x + c2 * x * x)), s); }, theta };
  }
  if (name === 'exponential') {
    const theta = [fit.a, fit.b, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [a, b, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - a * Math.exp(b * x)), s); }, theta };
  }
  if (name === 'exponential with offset') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [a, b, c, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - (a * Math.exp(b * x) + c)), s); }, theta };
  }
  if (name === 'power') {
    const theta = [fit.a, fit.b, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [a, b, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - a * Math.pow(x, b)), s); }, theta };
  }
  if (name === 'power with offset') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [a, b, c, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - (a * Math.pow(x, b) + c)), s); }, theta };
  }
  if (name === 'S-curve (logistic)') {
    const theta = [fit.a, fit.b, fit.c, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [a, b, c, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - a / (1 + Math.exp(-(b * x + c)))), s); }, theta };
  }
  if (name === 'Theil-Sen (robust line)' || name === 'RANSAC (robust line)') {
    // Same caveat as before: not MLE-fit originally, treated as "evidence
    // for this specific line" under a Gaussian-noise assumption.
    const theta = [fit.slope, fit.intercept, sigmaFromResiduals(residualsOf(fit.predict))];
    return { loglikFn: (t) => { const [sl, ic, s] = t; return gaussianLoglik(xs.map((x, i) => ys[i] - (sl * x + ic)), s); }, theta };
  }
  if (name === 'AR(1)') {
    const yLag = ys.slice(0, -1), yNow = ys.slice(1);
    const resid = yNow.map((y, i) => y - (fit.c + fit.phi * yLag[i]));
    const sigma = sigmaFromResiduals(resid);
    const theta = [fit.c, fit.phi, sigma];
    return { loglikFn: (t) => { const [c, phi, s] = t; const lag = ys.slice(0, -1), now = ys.slice(1); return gaussianLoglik(now.map((y, i) => y - (c + phi * lag[i])), s); }, theta };
  }
  if (name === 'ARIMA') return buildArimaLoglik(fit, ys);
  return null;
}

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
      try { laplaceEvidence = laplaceLogEvidence(built.loglikFn, built.theta); } catch (e) { laplaceEvidence = null; }
      if (!Number.isFinite(laplaceEvidence)) laplaceEvidence = null;
    }
    rows.push({ candidate: cand, checkR2, laplaceEvidence });
  }
  if (!rows.length) return null;

  const toleranceConstant = 3, maxTolerance = 0.48;
  const effectiveTolerance = Math.min(toleranceConstant / checkLen, maxTolerance);
  const bestCheckR2 = Math.max(...rows.map(r => r.checkR2));
  const contenders = rows.filter(r => r.checkR2 >= bestCheckR2 - effectiveTolerance);
  const currentWinner = contenders.reduce((simplest, r) => (r.candidate.params < simplest.candidate.params ? r : simplest));

  const laplaceEligible = rows.filter(r => r.laplaceEvidence !== null);
  const laplaceWinner = laplaceEligible.length ? laplaceEligible.reduce((best, r) => (r.laplaceEvidence > best.laplaceEvidence ? r : best)) : null;

  console.log(`  [coverage] ${rows.length} candidates fit, ${laplaceEligible.length} have real evidence${laplaceEligible.length < rows.length ? ` (missing: ${rows.filter(r => r.laplaceEvidence === null).map(r => r.candidate.name).join(', ')})` : ' (ALL of them)'}`);

  return { rows, contenders, currentWinner, laplaceWinner, fitLen, checkLen };
}

const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));

function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  const trainLen = Math.floor(n * 0.7);
  const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
  if (testXs.length < 2 || trainLen < 10) return null;
  const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);

  console.log(`${group.title}`);
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
console.log(`\nEvaluated ${results.length} of ${groups.length} datasets.\n`);

console.log('=== Per-dataset: current rule (CHECK R² + tolerance + simplicity) vs. FULL-EVIDENCE Laplace rule (every candidate has a real likelihood now) ===\n');
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

console.log('=== Bootstrap: does the full-evidence Laplace rule beat the current rule on trimmed-average test R² (full-variance)? ===\n');
const BOOT = 3000;
const currentAverages = [], laplaceAverages = [];
let laplaceWins = 0, currentWins = 0, ties = 0;
for (let b = 0; b < BOOT; b++) {
  const rand = mulberry32(4044000 + b * 131);
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

console.log('\n=== Per-candidate: outcome across every dataset it was a contender for ===\n');
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
