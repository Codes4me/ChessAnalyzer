// ARIMA(p,d,q) feasibility check. This project already has AR(1), AR(2),
// and ARMA(1,1) (regression.js), all fit the same way: Hannan-Rissanen
// two-step OLS (no iterative MLE/Kalman filter needed), shrinkage prior on
// the coefficients, one-step-ahead backtesting via knownPrevY. ARIMA just
// adds the missing "I" (integrated/differencing) term those don't have --
// needed for genuinely NON-STATIONARY series (a stochastic trend/random
// walk), where AR(1)/AR(2)/ARMA(1,1) are fitting a mean-reverting model to
// data that doesn't mean-revert.
//
// Implementation: difference the series d times, fit AR(p)+MA(q) on the
// DIFFERENCED series via the same Hannan-Rissanen approach as ARMA(1,1)
// (generalized to arbitrary small p/q), forecast forward in differenced
// space, then integrate back up (cumulative-sum) to the original Y scale.
// Same shrinkage-toward-0 prior on the AR/MA coefficients as the existing
// models, for the same reason (tiny FIT windows destabilize raw OLS).
//
// Two-part validation, same as every other new method this session:
// (1) synthetic data with a KNOWN true (p,d,q) -- does it recover
// reasonable coefficients, and does over-differencing a genuinely
// STATIONARY series visibly hurt (the classic ARIMA pitfall) rather than
// being silently fine; (2) real corpus backtest against the existing
// AR(1)/AR(2)/ARMA(1,1)/persistence-drift candidates already in
// model_candidates.js, on real held-out R².

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { selectByExtrapolation, CANDIDATES } = require('./model_candidates');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

function differenceOnce(series) {
  const out = [];
  for (let t = 1; t < series.length; t++) out.push(series[t] - series[t - 1]);
  return out;
}

// General ARIMA(p,d,q), same conventions as regression.js's AR(1)/AR(2)/
// ARMA(1,1): shrinkage toward 0, predict(x) steps forward from the last
// training row using the training data's average X spacing.
function fitARIMA(xs, ys, { p = 1, d = 1, q = 0, shrinkage = 1.5 } = {}) {
  const n = xs.length;
  const lastLevels = []; // lastLevels[k] = last value of the k-times-differenced series (k=0..d-1), needed to integrate forecasts back up
  let series = ys.slice();
  for (let k = 0; k < d; k++) {
    lastLevels.push(series[series.length - 1]);
    series = differenceOnce(series);
  }
  const w = series; // the series actually modeled (length n-d)
  const minNeeded = Math.max(p, q) + p + q + 3;
  if (w.length < minNeeded) throw new Error(`Need at least ${minNeeded + d} data points for ARIMA(${p},${d},${q}).`);

  let phis = new Array(p).fill(0), thetas = new Array(q).fill(0), c = 0, eProxy = null, arOrderUsed = 0;

  if (p === 0 && q === 0) {
    c = mean(w);
  } else if (q === 0) {
    // Pure AR(p) on w, same OLS-by-lag-matrix approach as fitAR2 generalized.
    const yLags = [], yTarget = [];
    for (let t = p; t < w.length; t++) {
      yTarget.push(w[t]);
      const row = []; for (let l = 1; l <= p; l++) row.push(w[t - l]);
      yLags.push(row);
    }
    const fit = Regression.fitMultiLinear(yLags, yTarget);
    const pairs = yTarget.length;
    const shrink = pairs / (pairs + shrinkage);
    phis = fit.coeffs.map(v => v * shrink);
    const means = Array.from({ length: p }, (_, l) => mean(yLags.map(row => row[l])));
    c = mean(yTarget) - phis.reduce((s, phi, l) => s + phi * means[l], 0);
  } else {
    // Hannan-Rissanen two-step, generalized from ARMA(1,1): long AR for
    // proxy residuals, then regress w_t on [w lags..p, eProxy lags..q].
    arOrderUsed = Math.max(p + q + 2, Math.min(6, Math.floor(w.length / 3)));
    const yLagsLong = [], yTargetLong = [];
    for (let t = arOrderUsed; t < w.length; t++) {
      yTargetLong.push(w[t]);
      const row = []; for (let l = 1; l <= arOrderUsed; l++) row.push(w[t - l]);
      yLagsLong.push(row);
    }
    const longAr = Regression.fitMultiLinear(yLagsLong, yTargetLong);
    eProxy = yTargetLong.map((y, i) => y - longAr.predict(yLagsLong[i])); // index i corresponds to t = arOrderUsed + i

    const X = [], yTarget2 = [];
    const start = arOrderUsed + q; // need both p w-lags and q e-lags available
    for (let t = start; t < w.length; t++) {
      const row = [];
      for (let l = 1; l <= p; l++) row.push(w[t - l]);
      for (let l = 1; l <= q; l++) row.push(eProxy[t - l - arOrderUsed]);
      X.push(row);
      yTarget2.push(w[t]);
    }
    if (X.length < p + q + 2) throw new Error(`Not enough data left after warm-up for ARIMA(${p},${d},${q}).`);
    const step2 = Regression.fitMultiLinear(X, yTarget2);
    const pairs = X.length;
    const shrink = pairs / (pairs + shrinkage);
    phis = step2.coeffs.slice(0, p).map(v => v * shrink);
    thetas = step2.coeffs.slice(p, p + q).map(v => v * shrink);
    const colMeans = Array.from({ length: p + q }, (_, j) => mean(X.map(row => row[j])));
    c = mean(yTarget2) - colMeans.reduce((s, m, j) => s + step2.coeffs[j] * shrink * m, 0);
  }

  // One-step-ahead fit quality on the MODELED (differenced) series.
  let oneStepPredict, target;
  if (p === 0 && q === 0) {
    target = w; oneStepPredict = w.map(() => c);
  } else if (q === 0) {
    target = w.slice(p);
    oneStepPredict = target.map((_, i) => c + phis.reduce((s, phi, l) => s + phi * w[p + i - 1 - l], 0));
  } else {
    const start = arOrderUsed + q;
    target = w.slice(start);
    oneStepPredict = target.map((_, i) => {
      const t = start + i;
      let v = c;
      for (let l = 1; l <= p; l++) v += phis[l - 1] * w[t - l];
      for (let l = 1; l <= q; l++) v += thetas[l - 1] * eProxy[t - l - arOrderUsed];
      return v;
    });
  }
  const targetMean = mean(target);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < target.length; i++) { ssRes += (target[i] - oneStepPredict[i]) ** 2; ssTot += (target[i] - targetMean) ** 2; }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  // Forecasting: generate future w-values recursively (e=0 beyond the
  // first available lag, same convention as ARMA(1,1)), then integrate
  // each forecasted w back up through the d levels of differencing.
  const lastX = xs[n - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;
  const wHistory = w.slice(); // mutable tail used to seed the AR lags during recursion
  const eHistory = eProxy ? eProxy.slice() : [];

  function forecastW(steps) {
    const wh = wHistory.slice(), eh = eHistory.slice();
    const forecasts = [];
    for (let s = 0; s < steps; s++) {
      let v = c;
      for (let l = 1; l <= p; l++) v += phis[l - 1] * wh[wh.length - l];
      for (let l = 1; l <= q; l++) v += thetas[l - 1] * (eh.length >= l ? eh[eh.length - l] : 0);
      wh.push(v); eh.push(0); // future shocks assumed 0 (their expectation)
      forecasts.push(v);
    }
    return forecasts;
  }

  // Walk-forward state (mirrors fitAR1's knownPrevY, generalized): every
  // time the caller reveals a REAL previous y (via computeR2Sequential),
  // append it to a running history of actual y's, rebuild the REAL
  // differenced series and REAL one-step residuals from that actual
  // history, and forecast exactly ONE step using genuine lagged data --
  // instead of recursing blind from the end of training the way the
  // multi-step fallback below has to. This is what fixes the compounding-
  // error blowups found on real corpus backtests (e.g. Movie Gross,
  // income by age): those happened because a 400+-step blind recursion
  // has nothing real to correct itself against as it goes; one real step
  // at a time does.
  let yHistory = ys.slice();
  function computeWLevels(yArr) { let s = yArr.slice(); for (let k = 0; k < d; k++) s = differenceOnce(s); return s; }
  function oneStepWForecast(wHist, eHist) {
    let v = c;
    for (let l = 1; l <= p; l++) v += phis[l - 1] * wHist[wHist.length - l];
    for (let l = 1; l <= q; l++) v += thetas[l - 1] * (eHist.length >= l ? eHist[eHist.length - l] : 0);
    return v;
  }
  function rebuildEHistory(wHist) {
    if (q === 0) return [];
    const eh = [];
    const minHist = Math.max(p, 1);
    for (let t = minHist; t < wHist.length; t++) {
      const forecast = oneStepWForecast(wHist.slice(0, t), eh);
      eh.push(wHist[t] - forecast);
    }
    return eh;
  }

  const predict = (x, knownPrevY) => {
    if (typeof knownPrevY === 'number' && Number.isFinite(knownPrevY)) {
      if (knownPrevY !== yHistory[yHistory.length - 1]) yHistory.push(knownPrevY); // skip the harness's first call, which just re-reveals the last training point already in yHistory
      const levels = [yHistory];
      for (let k = 0; k < d; k++) levels.push(differenceOnce(levels[levels.length - 1]));
      const wHist = levels[d];
      const minNeededHere = Math.max(p, 1) + (q > 0 ? q : 0);
      if (wHist.length <= minNeededHere) return yHistory[yHistory.length - 1]; // not enough real history yet to form a lagged forecast -- fall back to last known value
      const eHist = rebuildEHistory(wHist);
      const wForecast = oneStepWForecast(wHist, eHist);
      let v = wForecast;
      for (let k = d - 1; k >= 0; k--) v = levels[k][levels[k].length - 1] + v;
      return v;
    }
    // No real previous y available (e.g. drawing the extrapolated curve
    // past the end of all data) -- falls back to the original blind
    // multi-step recursion from the last training point.
    const steps = Math.max(0, Math.round((x - lastX) / avgStep));
    if (steps === 0) return ys[n - 1];
    const wForecasts = forecastW(steps);
    let levelForecasts = wForecasts;
    for (let k = d - 1; k >= 0; k--) {
      let running = lastLevels[k];
      const next = [];
      for (const v of levelForecasts) { running += v; next.push(running); }
      levelForecasts = next;
    }
    return levelForecasts[levelForecasts.length - 1];
  };

  return { type: `ARIMA(${p},${d},${q})`, p, d, q, phis, thetas, c, r2, predict };
}

// =====================================================================
// PART 1: synthetic validation
// =====================================================================
console.log('=== Part 1a: ARIMA(1,1,0) on a genuinely non-stationary series (random walk + autocorrelated drift) ===\n');
console.log('True: differenced series follows AR(1) with phi=0.5. Compare ARIMA(1,1,0) vs. plain AR(1) (no differencing) vs. persistence-drift.\n');
for (const n of [40, 150]) {
  const truePhi = 0.5;
  const arima110Estimates = [], ar1Estimates = [];
  const TRIALS = 300;
  for (let t = 0; t < TRIALS; t++) {
    const rand = mulberry32(300000 + n * 13 + t);
    const xs = Array.from({ length: n }, (_, i) => i + 1);
    const w = [randNormal(rand)];
    for (let i = 1; i < n; i++) w.push(truePhi * w[i - 1] + randNormal(rand));
    const ys = [100]; for (let i = 0; i < n - 1; i++) ys.push(ys[ys.length - 1] + w[i + 1]); // integrate: y is a random walk whose INCREMENTS are AR(1)
    let fitArima; try { fitArima = fitARIMA(xs, ys, { p: 1, d: 1, q: 0 }); } catch (e) { continue; }
    arima110Estimates.push(fitArima.phis[0]);
    let fitAr1; try { fitAr1 = Regression.fitAR1(xs, ys); } catch (e) { continue; }
    ar1Estimates.push(fitAr1.phi);
  }
  console.log(`--- n=${n} (true phi on the DIFFERENCED series = ${truePhi}) ---`);
  console.log(`  ARIMA(1,1,0) recovered phi: mean=${mean(arima110Estimates).toFixed(3)}  (${arima110Estimates.length}/${TRIALS} converged)`);
  console.log(`  plain AR(1) on raw levels "phi": mean=${mean(ar1Estimates).toFixed(3)}  (expected to land near 1.0 -- AR(1) on levels of a random walk just rediscovers the unit root, not the true dynamics)\n`);
}

console.log('=== Part 1b: over-differencing penalty -- fit ARIMA(1,1,0) on a STATIONARY AR(1) series that needs NO differencing ===\n');
console.log('If ARIMA correctly warns against blind differencing, ARIMA(1,1,0)\'s held-out forecast R² should be clearly worse than plain AR(1)\'s on truly stationary data.\n');
{
  const n = 80, truePhi = 0.6, TRIALS = 300;
  const edgeArima = [], edgeAr1 = [];
  for (let t = 0; t < TRIALS; t++) {
    const rand = mulberry32(400000 + t);
    const xs = Array.from({ length: n }, (_, i) => i + 1);
    const ys = [randNormal(rand) * 3];
    for (let i = 1; i < n; i++) ys.push(truePhi * ys[i - 1] + randNormal(rand) * 3);
    const trainLen = Math.floor(n * 0.7);
    const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);
    const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
    let fitArima, fitAr1;
    try { fitArima = fitARIMA(trainXs, trainYs, { p: 1, d: 1, q: 0 }); } catch (e) { continue; }
    try { fitAr1 = Regression.fitAR1(trainXs, trainYs); } catch (e) { continue; }
    edgeArima.push(Regression.computeR2(testXs, testYs, fitArima.predict));
    edgeAr1.push(Regression.computeR2(testXs, testYs, fitAr1.predict));
  }
  console.log(`  ARIMA(1,1,0) (wrongly differenced) median held-out R²: ${medianOf(edgeArima).toFixed(4)}`);
  console.log(`  plain AR(1) (correctly specified) median held-out R²:  ${medianOf(edgeAr1).toFixed(4)}`);
}
function medianOf(a) { const s = [...a].sort((x, y) => x - y); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

// =====================================================================
// PART 2: real corpus backtest -- does any ARIMA(p,d,q) beat the existing
// AR(1)/AR(2)/ARMA(1,1)/persistence-drift candidates on real held-out R²?
// =====================================================================
console.log('\n=== Part 2: real corpus -- does ARIMA beat the existing candidates already in model_candidates.js? ===\n');
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));
const ARIMA_CONFIGS = [{ p: 1, d: 1, q: 0 }, { p: 0, d: 1, q: 1 }, { p: 1, d: 1, q: 1 }, { p: 2, d: 1, q: 0 }];

function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  if (n < 20) return null;
  const trainLen = Math.floor(n * 0.7);
  const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);
  const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
  if (testXs.length < 3) return null;

  // Walk-forward scoring throughout (Regression.computeR2Sequential), same
  // convention backtest_models.js already uses for AR(1): every candidate
  // whose predict() ignores the extra "real previous y" argument scores
  // identically to plain computeR2 anyway, so this is a safe superset, not
  // a different rule for different candidates.
  let bestExisting = null;
  try {
    const sel = selectByExtrapolation(trainXs, trainYs, { fitFrac: 0.5, checkFrac: 0.2 });
    const fit = sel.candidate.fit(trainXs, trainYs);
    bestExisting = { name: sel.candidate.name, r2: Regression.computeR2Sequential(trainYs[trainYs.length - 1], testXs, testYs, fit.predict) };
  } catch (e) { return null; }

  // Pick the ARIMA config honestly: a FIT/CHECK split carved out of TRAIN
  // only (same 70/20 convention selectByExtrapolation uses internally) --
  // never touching TEST during selection, same rule every other candidate
  // already follows.
  const fitLen = Math.floor(trainXs.length * 0.7);
  const fitXs = trainXs.slice(0, fitLen), fitYs = trainYs.slice(0, fitLen);
  const checkXs = trainXs.slice(fitLen), checkYs = trainYs.slice(fitLen);
  if (checkXs.length < 3) return null;

  let chosenCfg = null, chosenCheckR2 = -Infinity;
  for (const cfg of ARIMA_CONFIGS) {
    let fit; try { fit = fitARIMA(fitXs, fitYs, cfg); } catch (e) { continue; }
    const r2 = Regression.computeR2Sequential(fitYs[fitYs.length - 1], checkXs, checkYs, fit.predict);
    if (r2 > chosenCheckR2) { chosenCheckR2 = r2; chosenCfg = cfg; }
  }
  if (!chosenCfg) return null;

  // Refit the CHOSEN config on the full TRAIN set, then score on TEST --
  // same two-stage discipline (select on fit/check, report on held-out
  // test) as the rest of this project's backtests.
  let finalFit; try { finalFit = fitARIMA(trainXs, trainYs, chosenCfg); } catch (e) { return null; }
  const arimaR2 = Regression.computeR2Sequential(trainYs[trainYs.length - 1], testXs, testYs, finalFit.predict);
  return { title: group.title, n, existingName: bestExisting.name, existingR2: bestExisting.r2, arimaName: finalFit.type, arimaR2 };
}

const results = groups.map(evaluateDataset).filter(Boolean);
console.log(`${results.length} of ${groups.length} datasets evaluated (rest too small or failed to fit).\n`);
let helped = 0, hurt = 0;
results.slice().sort((a, b) => (b.arimaR2 - b.existingR2) - (a.arimaR2 - a.existingR2)).forEach(r => {
  const delta = r.arimaR2 - r.existingR2;
  if (delta > 0.01) helped++; else if (delta < -0.01) hurt++;
  console.log(`  ${r.title}  n=${r.n}  existing best=${r.existingName} (R²=${r.existingR2.toFixed(3)})  best ARIMA=${r.arimaName} (R²=${r.arimaR2.toFixed(3)})  delta=${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`);
});
console.log(`\nBest-available ARIMA beat the existing best candidate (>0.01 R²) on ${helped}/${results.length}, lost on ${hurt}/${results.length}.`);
