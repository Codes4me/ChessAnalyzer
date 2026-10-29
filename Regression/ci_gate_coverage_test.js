// Validates the CI-method gate added in regression.js (chooseParameterCI,
// movingBlockBootstrapParameterCI, wildBootstrapParameterCI): the gate was
// designed (block bootstrap under autocorrelation, wild bootstrap under
// heteroscedasticity, classical/plain otherwise) but never checked the same
// way every other test in this project was — by generating synthetic data
// with a KNOWN true parameter and known autocorrelation/heteroscedasticity,
// then counting how often the resulting 95% interval actually contains the
// true value. A well-calibrated 95% CI should cover the truth ~95% of the
// time; this is the same "hand-compute one case you know the answer for"
// check used for Shapiro-Wilk and Anderson-Darling elsewhere in this
// project.
//
// Two things get checked per error structure:
//   (a) DIRECT coverage: call the matching CI method directly (skip
//       detection) to validate the method itself is correctly calibrated
//       when its assumption actually holds.
//   (b) END-TO-END (gate) coverage: run the full chooseParameterCI gate,
//       which first runs Durbin-Watson/White's test and only switches
//       methods if THEY fire — this is what the live page actually does,
//       and its coverage depends on both the detector's power and the
//       method's own calibration.
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
function randNormal(rand) {
  // Box-Muller
  const u1 = Math.max(1e-12, rand()), u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const TRUE_A = 3, TRUE_B = 2; // y = a + b*x
const N = 60;
const TRIALS = 400;
const xs = Array.from({ length: N }, (_, i) => i + 1);

function fitLine(rxs, rys) {
  return Regression.runRegression(rxs, rys, { degree: 1 });
}

// --- Scenario generators -----------------------------------------------

// AR(1) errors: e_t = phi*e_{t-1} + eta_t, eta_t ~ N(0, sigma^2*(1-phi^2))
// so the marginal variance of e_t stays sigma^2 regardless of phi (keeps
// scenarios comparable to each other and to the IID baseline).
function genAR1(rand, phi, sigma) {
  const e = new Array(N);
  e[0] = randNormal(rand) * sigma;
  const innovSd = sigma * Math.sqrt(Math.max(1e-9, 1 - phi * phi));
  for (let t = 1; t < N; t++) e[t] = phi * e[t - 1] + randNormal(rand) * innovSd;
  return xs.map((x, i) => TRUE_A + TRUE_B * x + e[i]);
}

// Heteroscedastic errors: sd grows linearly with x, scaled so the AVERAGE
// variance across the sample matches `sigma` (comparable overall noise
// level to the other scenarios).
function genHetero(rand, strength, sigma) {
  const rawSd = xs.map(x => 1 + strength * (x / N));
  const meanRawVar = rawSd.reduce((s, v) => s + v * v, 0) / N;
  const scale = sigma / Math.sqrt(meanRawVar);
  return xs.map((x, i) => TRUE_A + TRUE_B * x + randNormal(rand) * rawSd[i] * scale);
}

function genIID(rand, sigma) {
  return xs.map(x => TRUE_A + TRUE_B * x + randNormal(rand) * sigma);
}

// --- Coverage harness -----------------------------------------------------

function covers(ci, idx, trueVal) {
  if (!ci) return null;
  const c = ci.coeffs && ci.coeffs[idx];
  if (!c) return null;
  return trueVal >= c.lower && trueVal <= c.upper;
}

function runScenario(name, genYs, { directMethod, seedBase }) {
  let directHitsA = 0, directHitsB = 0, directN = 0;
  let gateHitsA = 0, gateHitsB = 0, gateN = 0, gateFired = 0;
  let classicalHitsA = 0, classicalHitsB = 0, classicalN = 0; // what you'd get WITHOUT the gate, for comparison

  for (let t = 0; t < TRIALS; t++) {
    const rand = mulberry32(seedBase + t * 7919);
    const ys = genYs(rand);
    let fit;
    try { fit = fitLine(xs, ys); } catch (e) { continue; }

    // (a) direct: call the matching method regardless of detection
    const directCi = directMethod(fit, xs, ys);
    const hitA = covers(directCi, 0, TRUE_A), hitB = covers(directCi, 1, TRUE_B);
    if (hitA !== null && hitB !== null) {
      directN++; if (hitA) directHitsA++; if (hitB) directHitsB++;
    }

    // (b) end-to-end gate, as the live page actually calls it
    const classical = fit.ci;
    const gated = Regression.chooseParameterCI(fitLine, xs, ys, fit.predict, classical, { iterations: 150, seed: seedBase + t * 7919 + 1 });
    const gHitA = covers(gated, 0, TRUE_A), gHitB = covers(gated, 1, TRUE_B);
    if (gHitA !== null && gHitB !== null) {
      gateN++; if (gHitA) gateHitsA++; if (gHitB) gateHitsB++;
      if (gated.gateReason) gateFired++;
    }

    // classical-only baseline (what the page would have shown before this
    // gate existed) — useful to see HOW MUCH the gate actually helps.
    const cHitA = covers(classical, 0, TRUE_A), cHitB = covers(classical, 1, TRUE_B);
    if (cHitA !== null && cHitB !== null) {
      classicalN++; if (cHitA) classicalHitsA++; if (cHitB) classicalHitsB++;
    }
  }

  const pct = (h, n) => n ? (100 * h / n).toFixed(1) : 'n/a';
  console.log(`\n=== ${name} ===`);
  console.log(`  direct method coverage:    a=${pct(directHitsA, directN)}%  b=${pct(directHitsB, directN)}%  (n=${directN})`);
  console.log(`  gate (end-to-end) coverage: a=${pct(gateHitsA, gateN)}%  b=${pct(gateHitsB, gateN)}%  (n=${gateN}, gate fired ${gateFired}/${gateN} = ${pct(gateFired, gateN)}%)`);
  console.log(`  classical-only baseline:    a=${pct(classicalHitsA, classicalN)}%  b=${pct(classicalHitsB, classicalN)}%  (n=${classicalN})`);
}

// --- Scenarios, using phi/strength ranges this project's own corpus has
// actually shown (see the autocorrelation comment near regression.js:1495:
// "lag-1 autocorrelation 0.55-0.98" on flagged real datasets).

runScenario(
  'IID baseline (sanity check — gate should mostly stay classical)',
  rand => genIID(rand, 4),
  { directMethod: (fit) => fit.ci, seedBase: 1000 }
);

runScenario(
  'AR(1), phi=0.3 (mild autocorrelation)',
  rand => genAR1(rand, 0.3, 4),
  { directMethod: (fit, xs, ys) => Regression.movingBlockBootstrapParameterCI(fitLine, xs, ys, { iterations: 150, seed: 42 }), seedBase: 2000 }
);

runScenario(
  'AR(1), phi=0.7 (strong autocorrelation)',
  rand => genAR1(rand, 0.7, 4),
  { directMethod: (fit, xs, ys) => Regression.movingBlockBootstrapParameterCI(fitLine, xs, ys, { iterations: 150, seed: 42 }), seedBase: 3000 }
);

runScenario(
  'AR(1), phi=0.9 (very strong autocorrelation)',
  rand => genAR1(rand, 0.9, 4),
  { directMethod: (fit, xs, ys) => Regression.movingBlockBootstrapParameterCI(fitLine, xs, ys, { iterations: 150, seed: 42 }), seedBase: 4000 }
);

runScenario(
  'Heteroscedastic, strength=3 (moderate — sd roughly quadruples end to end)',
  rand => genHetero(rand, 3, 4),
  { directMethod: (fit, xs, ys) => Regression.wildBootstrapParameterCI(fitLine, xs, ys, fit.predict, { iterations: 150, seed: 42 }), seedBase: 5000 }
);

runScenario(
  'Heteroscedastic, strength=8 (severe)',
  rand => genHetero(rand, 8, 4),
  { directMethod: (fit, xs, ys) => Regression.wildBootstrapParameterCI(fitLine, xs, ys, fit.predict, { iterations: 150, seed: 42 }), seedBase: 6000 }
);

// --- Block-length sweep: regression.js's movingBlockBootstrapParameterCI
// default (round(n/3)) came from this sweep, not a textbook rule of thumb.
// Two textbook candidates (n^(1/3), log(n)) and several n/k fractions were
// tried directly against AR(1) and heteroscedastic synthetic data at
// n=30/60/120 — n/3 won EVERY scenario tested (n/4 a close second; both
// n^(1/3) and log(n) were clearly worse than even the prior sqrt(n)
// default). Kept here, runnable on demand (`node ci_gate_coverage_test.js
// --sweep`), so the choice can be re-checked if the fitLine model, corpus,
// or sample sizes this project cares about ever change.
function sweepBlockLengths() {
  const N_VALUES = [30, 60, 120];
  const CANDIDATES = [
    ['sqrt(n)', n => Math.round(Math.sqrt(n))],
    ['log(n)', n => Math.max(2, Math.round(Math.log(n)))],
    ['n/2', n => Math.round(n / 2)],
    ['n/3', n => Math.round(n / 3)],
    ['n/4', n => Math.round(n / 4)],
    ['n/6', n => Math.round(n / 6)],
    ['n/10', n => Math.round(n / 10)],
  ];
  function coverageAt(n, genYs, L, seedBase, trials = 250) {
    const txs = Array.from({ length: n }, (_, i) => i + 1);
    let hitsA = 0, hitsB = 0, trialsRun = 0;
    for (let t = 0; t < trials; t++) {
      const rand = mulberry32(seedBase + t * 7919);
      const ys = genYs(txs, rand);
      let fit; try { fit = fitLine(txs, ys); } catch (e) { continue; }
      const ci = Regression.movingBlockBootstrapParameterCI(fitLine, txs, ys, { iterations: 150, seed: 42, blockLength: L });
      const hA = covers(ci, 0, TRUE_A), hB = covers(ci, 1, TRUE_B);
      if (hA !== null && hB !== null) { trialsRun++; if (hA) hitsA++; if (hB) hitsB++; }
    }
    return { a: 100 * hitsA / trialsRun, b: 100 * hitsB / trialsRun };
  }
  for (const n of N_VALUES) {
    console.log(`\n=== n=${n} ===`);
    for (const [label, fn] of CANDIDATES) {
      const L = fn(n);
      const txs = Array.from({ length: n }, (_, i) => i + 1);
      const genAR1n = (phi, sigma) => (xs2, rand) => {
        const e = new Array(n); e[0] = randNormal(rand) * sigma;
        const innovSd = sigma * Math.sqrt(Math.max(1e-9, 1 - phi * phi));
        for (let t = 1; t < n; t++) e[t] = phi * e[t - 1] + randNormal(rand) * innovSd;
        return xs2.map((x, i) => TRUE_A + TRUE_B * x + e[i]);
      };
      const genHeteroN = (strength, sigma) => (xs2, rand) => {
        const rawSd = xs2.map(x => 1 + strength * (x / n));
        const meanRawVar = rawSd.reduce((s, v) => s + v * v, 0) / n;
        const scale = sigma / Math.sqrt(meanRawVar);
        return xs2.map((x, i) => TRUE_A + TRUE_B * x + randNormal(rand) * rawSd[i] * scale);
      };
      const r7 = coverageAt(n, genAR1n(0.7, 4), L, 3000);
      const r9 = coverageAt(n, genAR1n(0.9, 4), L, 4000);
      const rh = coverageAt(n, genHeteroN(8, 4), L, 6000);
      console.log(`  ${label.padEnd(8)} L=${String(L).padStart(3)}  AR1(0.7) a=${r7.a.toFixed(1)}% b=${r7.b.toFixed(1)}%   AR1(0.9) a=${r9.a.toFixed(1)}% b=${r9.b.toFixed(1)}%   Hetero(8) a=${rh.a.toFixed(1)}% b=${rh.b.toFixed(1)}%`);
    }
  }
}

if (process.argv.includes('--sweep')) sweepBlockLengths();
