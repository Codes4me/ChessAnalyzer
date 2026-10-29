// Fine-grained follow-up to the coarse block-length sweep that picked
// n/3 as the moving-block bootstrap's default (regression.js:
// movingBlockBootstrapParameterCI). That sweep only tested n/2, n/3, n/4,
// n/5, n/6, n/8, n/10, n/12, n/15, n/20 — big jumps. This sweeps every
// INTEGER block length in a window around n/3 (since L=round(n/a) means
// many nearby fractional `a` values collapse to the same integer anyway,
// a fine sweep over `a` really means a fine sweep over integer L), and
// bootstraps the COVERAGE ESTIMATE itself at each L — not just the point
// estimate — so differences between neighboring L values can be judged
// against their own uncertainty instead of eyeballing raw percentages.
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
function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

const TRUE_A = 3, TRUE_B = 2;
function fitLine(xs, ys) { return Regression.runRegression(xs, ys, { degree: 1 }); }
function covers(ci, idx, trueVal) { if (!ci) return null; const c = ci.coeffs && ci.coeffs[idx]; if (!c) return null; return trueVal >= c.lower && trueVal <= c.upper; }

function genAR1(n, xs, rand, phi, sigma) {
  const e = new Array(n); e[0] = randNormal(rand) * sigma;
  const innovSd = sigma * Math.sqrt(Math.max(1e-9, 1 - phi * phi));
  for (let t = 1; t < n; t++) e[t] = phi * e[t - 1] + randNormal(rand) * innovSd;
  return xs.map((x, i) => TRUE_A + TRUE_B * x + e[i]);
}
function genHetero(n, xs, rand, strength, sigma) {
  const rawSd = xs.map(x => 1 + strength * (x / n));
  const meanRawVar = rawSd.reduce((s, v) => s + v * v, 0) / n;
  const scale = sigma / Math.sqrt(meanRawVar);
  return xs.map((x, i) => TRUE_A + TRUE_B * x + randNormal(rand) * rawSd[i] * scale);
}

// Bootstrap CI on a COVERAGE RATE (resampling the hit/miss booleans
// themselves, not re-running the simulation) — standard bootstrap of a
// proportion.
function bootstrapCoverageCI(hits, { iterations = 1500, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = hits.length;
  const samples = [];
  for (let i = 0; i < iterations; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += hits[Math.floor(random() * n)];
    samples.push(s / n);
  }
  samples.sort((a, b) => a - b);
  return { lower: samples[Math.floor(0.025 * (samples.length - 1))], upper: samples[Math.ceil(0.975 * (samples.length - 1))] };
}

function coverageForL(n, genYs, L, seedBase, trials = 400) {
  const xs = Array.from({ length: n }, (_, i) => i + 1);
  const hitsA = [], hitsB = [];
  for (let t = 0; t < trials; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(xs, rand);
    let fit; try { fit = fitLine(xs, ys); } catch (e) { continue; }
    const ci = Regression.movingBlockBootstrapParameterCI(fitLine, xs, ys, { iterations: 150, seed: 42, blockLength: L });
    const hA = covers(ci, 0, TRUE_A), hB = covers(ci, 1, TRUE_B);
    if (hA !== null && hB !== null) { hitsA.push(hA ? 1 : 0); hitsB.push(hB ? 1 : 0); }
  }
  const meanA = hitsA.reduce((s, v) => s + v, 0) / hitsA.length;
  const meanB = hitsB.reduce((s, v) => s + v, 0) / hitsB.length;
  const ciA = bootstrapCoverageCI(hitsA);
  const ciB = bootstrapCoverageCI(hitsB);
  return { meanA, meanB, ciA, ciB, n: hitsA.length };
}

function runSweep(n, scenarios, Lrange) {
  console.log(`\n========== n=${n} ==========`);
  for (const [label, genYs, seedBase] of scenarios) {
    console.log(`\n-- ${label} --`);
    let bestL = null, bestAvg = -Infinity;
    const rows = [];
    for (const L of Lrange) {
      const r = coverageForL(n, genYs, L, seedBase, 400);
      const avg = (r.meanA + r.meanB) / 2;
      rows.push({ L, r, avg });
      if (avg > bestAvg) { bestAvg = avg; bestL = L; }
    }
    for (const { L, r, avg } of rows) {
      const a = (n / L).toFixed(2);
      const marker = L === bestL ? '  <== best' : '';
      console.log(`  L=${String(L).padStart(3)} (a=n/${a})  cov(a)=${(100 * r.meanA).toFixed(1)}% [${(100 * r.ciA.lower).toFixed(1)},${(100 * r.ciA.upper).toFixed(1)}]  cov(b)=${(100 * r.meanB).toFixed(1)}% [${(100 * r.ciB.lower).toFixed(1)},${(100 * r.ciB.upper).toFixed(1)}]  avg=${(100 * avg).toFixed(1)}%${marker}`);
    }
  }
}

// n=60 fine sweep: integer L from 14 to 26 (covers a = 60/14=4.29 down to
// 60/26=2.31 — a window straddling the coarse sweep's winner, a=n/3=L20).
runSweep(60, [
  ['AR1 phi=0.7 (strong)', (xs, rand) => genAR1(60, xs, rand, 0.7, 4), 3000],
  ['AR1 phi=0.9 (very strong — the hard case)', (xs, rand) => genAR1(60, xs, rand, 0.9, 4), 4000],
  ['Heteroscedastic, severe (strength=8)', (xs, rand) => genHetero(60, xs, rand, 8, 4), 6000],
], Array.from({ length: 13 }, (_, i) => 14 + i));

// n=120 spot-check: same relative window, scaled (a=120/28=4.29 down to
// a=120/52=2.31 — the same `a` range as the n=60 sweep above).
runSweep(120, [
  ['AR1 phi=0.9 (very strong)', (xs, rand) => genAR1(120, xs, rand, 0.9, 4), 4000],
], Array.from({ length: 13 }, (_, i) => 28 + i * 2));
