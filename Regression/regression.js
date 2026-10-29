// Fits a polynomial regression (degree 1 = straight line, 2 = curve with one
// bend, 3 = curve with two bends, etc.) to X/Y data using least squares.

// Math.min(...arr)/Math.max(...arr) spread every element into a function
// call — past roughly 65k-130k elements (varies by JS engine), that blows
// the call stack with "Maximum call stack size exceeded". A plain loop has
// no such limit, so use these for any array that might be data-set-sized
// (a real CSV can easily have 100k+ rows) rather than reaching for
// Math.min/max directly.
function arrayMin(arr) {
  let m = Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
  return m;
}
function arrayMax(arr) {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

// Recommends a histogram bin count by actually fitting a regression for
// it (optimal_bins_regression.js), rather than reaching for a textbook
// rule (Sturges/Scott/Freedman-Diaconis): for many synthetic samples
// (varying n AND shape -- Gaussian, Laplace, Student-t, log-normal), the
// REAL optimal bin count was found by grid-searching every candidate bin
// count and keeping whichever minimized a (degrees-of-freedom-reduced)
// chi-square against that sample's own best-fit distribution (the same
// Gaussian/Laplace/Student-t machinery chooseErrorDistribution already
// uses) -- then bins~f(n, kurtosis) was fit against that real optimum.
//
// Outliers are excluded by rescaling to the middle 5th-95th percentile,
// NOT a symmetric 1.5*MAD fence -- that was tried first and found to be
// wrong for skewed data: confirmed directly on synthetic log-normal data,
// a symmetric MAD fence dropped 16% of the sample with EVERY dropped
// point coming from one tail (0% from the other) -- a symmetric distance
// rule just doesn't make sense applied to an asymmetric distribution.
// Percentile trimming keeps exactly the middle 90% regardless of which
// way (or how much) the data is skewed, by construction.
//
// A second predictor (sample kurtosis of the trimmed data, plus its
// interaction with sqrt(n)) was added after finding that shape matters
// MORE than sample size alone: at the same n=2000, the real optimal bin
// count ranged from 13 (log-normal) to 69 (Gaussian) -- over 5x apart --
// so an n-only formula was fundamentally underspecified (R²=0.63,
// out-of-sample mean error 5.08 bins). Adding kurtosis roughly halved
// that error (R²=0.90, out-of-sample mean error 2.88 bins).
//
// The raw sample kurtosis is then SHRUNK toward 0 (the Gaussian/"no
// evidence of non-normal tails yet" prior) before being used, same
// shrink-toward-a-neutral-prior idea this project already uses for
// AR(1)'s phi: kurtosis_used = kurtosis_raw * n/(n+900). Sample kurtosis
// is a notoriously high-variance estimator at small n, so a single small
// sample's raw value can easily be mostly noise. Swept k from 0 (no
// shrinkage) up to 10000 and validated out-of-sample each time (Part 4,
// optimal_bins_regression.js): k=900 is a genuine interior minimum (a
// real U-shape, not "more shrinkage is always better"), cutting
// out-of-sample mean error further from 2.88 to 2.00 bins. Notably,
// TRAINING R² kept climbing all the way to k=10000 even as out-of-sample
// error got worse past k=900 -- picking k by training fit alone would
// have picked the wrong answer, which is exactly why this was validated
// out-of-sample instead.
//
// |Sample skewness| (sign doesn't matter -- a shape skewed either way
// should cost the same bin budget) was added next, with its OWN
// shrinkage constant (no reason to assume skewness needs the same amount
// of shrinkage as kurtosis just because they're both central moments) --
// swept and found k=600. Accepted on a STRICTER test than out-of-sample
// MAE alone: held-out log-likelihood (Gaussian residuals, sigma fixed
// from each candidate model's own held-out residuals, so no model could
// borrow another's sigma to look better). Adding skewness improved
// held-out log-likelihood from -100.67 to -94.16 (Part 5, same script) --
// a large, clearly real improvement, not a marginal one. A version with
// an extra sqrt(n)*skewness interaction term was also tried and came out
// very slightly worse (-94.21) than this simpler, interaction-free form,
// so the simpler one is what's used.
function recommendedHistogramBins(values, { fallback = 20 } = {}) {
  if (values.length < 15) return fallback; // too few to estimate reliably
  const sorted = [...values].sort((a, b) => a - b);
  const percentileOf = p => {
    const idx = p * (sorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  };
  const lo = percentileOf(0.05), hi = percentileOf(0.95);
  const trimmed = values.filter(v => v >= lo && v <= hi);
  if (trimmed.length < 15) return fallback;

  const m = trimmed.reduce((a, v) => a + v, 0) / trimmed.length;
  const sd = Math.sqrt(trimmed.reduce((a, v) => a + (v - m) ** 2, 0) / trimmed.length);
  const n = trimmed.length;
  const rawKurtosis = sd > 0 ? trimmed.reduce((a, v) => a + ((v - m) / sd) ** 4, 0) / trimmed.length - 3 : 0;
  const rawSkewness = sd > 0 ? trimmed.reduce((a, v) => a + ((v - m) / sd) ** 3, 0) / trimmed.length : 0;
  const kurtosis = rawKurtosis * n / (n + 900); // shrink toward the Gaussian prior (0) -- see comment above
  const absSkewShrunk = Math.abs(rawSkewness) * n / (n + 600); // own shrinkage constant, separately validated
  const sqrtN = Math.sqrt(n);

  // Coefficients from optimal_bins_regression.js's Part 5 fit (kurtosis
  // k=900, skewness k=600), accepted on held-out log-likelihood.
  const a = -0.433706, b = 0.512906, c = 34.559888, d = -2.414417, e = -11.638164;
  const bins = a + b * sqrtN + c * kurtosis + d * sqrtN * kurtosis + e * absSkewShrunk;
  // Floor of 3, not 2: the training grid search itself never explored
  // fewer than 3 bins (minBins=3 in optimal_bins_regression.js), so this
  // formula was never validated to mean anything below that.
  return Math.max(3, Math.min(200, Math.round(bins)));
}

// Goes one step further than recommendedHistogramBins above: instead of
// PREDICTING a bin count from a formula trained on synthetic data, this
// directly runs the REAL search that formula was only ever trained to
// approximate (same method as findOptimalBins in
// optimal_bins_regression.js: trim to the 5th-95th percentile core, fit
// whichever of Gaussian/Laplace/Student-t chooseErrorDistribution picks
// for THAT core, then grid-search bin WIDTH by chi-square goodness-of-fit
// against it) -- an exact answer for the dataset on hand, not an
// approximation learned from other, synthetic data.
//
// Fits the distribution on the trimmed CORE, not the full data -- this
// matters, and isn't just mirroring the training script out of habit: a
// first version of this function reused whatever distribution fit the
// histogram's OVERLAY already had (fit on the full, untrimmed data, same
// as every other overlay fit in this project). That full-data fit's
// sigma/scale is measurably inflated by the same far tail points search
// should properly ignore (confirmed directly: 9.94 vs 7.82 on one 2000-
// point Gaussian test case -- a ~27% difference) -- a few genuinely
// far-out points widen the fitted curve just enough that almost every bin
// near those edges reads as "too sparse to trust" (expected < 5) long
// before the bins covering the dense middle do, which crashed the whole
// search down to single-digit bin counts even for clean, mostly-Gaussian
// data. Refitting on the core avoids that: the core's own fit is a
// tighter, truer description of its own shape, so the chi-square search
// run against it doesn't get spuriously poisoned by the very outliers it
// was never trying to bin finely in the first place.
//
// Searches bin WIDTH on that core, not bin COUNT on the full range, then
// applies that one winning width across the full range to get the bin
// count the chart actually renders (drawHistogram always spans full
// min-max) -- mirroring what classic rules (Scott/Freedman-Diaconis)
// already do by construction: pick a width from the reliable part of the
// data, then just tile the whole range with it.
//
// GATES on the winning width's own goodness-of-fit before trusting it,
// which is what makes this noise-robust -- found the hard way. Tested
// under 5% added noise (simulating real-world x/y measurement error)
// against the CLEAN data's own true distribution, a first version (no
// gate) lost to the plain formula above at n>=1000 with heavy tails
// (confirmed: n=1000 Laplace, formula's chi-square=0.98 vs this search's
// 1.87). The cause wasn't bin width specifically -- it traced back to
// chooseErrorDistribution itself sometimes picking a MEDIOCRE distribution
// for the noisy core (confirmed directly: for that Laplace case, it fit
// "Gaussian" by BIC, but every bin count from 3 up to the sparsity floor
// scored reduced chi-square between 3.4 and 17.5 -- NEVER close to the ~1
// a genuinely correct fit should hover around at any valid resolution).
// Once the underlying fit is that mediocre, tiling its winning width out
// across the full range just magnifies the mismatch into many more bins,
// which is exactly backwards. The fix: after finding the best-scoring
// candidate, check whether that score is even statistically consistent
// with a correct fit (chiSquareUpperCritical's Wilson-Hilferty-based
// critical value at alpha=0.10) -- if not, no candidate width here was
// trustworthy, so defer entirely to the formula rather than confidently
// tiling a bad fit onto the chart.
//
// Binary-searches a sorted copy of the core to count each candidate's
// bars in O(bins*log n), and computes the distribution's integral ONCE
// (it doesn't depend on bin count, only each candidate's bin width does)
// rather than redoing a 300-step numeric integral per candidate -- matters
// once a dataset reaches tens of thousands of rows, where the difference
// is instant versus a multi-second freeze on a single click.
//
// Returns null (caller should fall back to recommendedHistogramBins above)
// when there's too little data (<15, or <15 left in the trimmed core), no
// distribution fit was possible, every candidate core bin count is too
// sparse for chi-square to mean anything even at the floor of 3 bins, or
// the best candidate still fails the goodness-of-fit gate above.
function findOptimalHistogramBins(values, { minBins = 3, maxCoreBins = 200, maxTotalBins = 500 } = {}) {
  const n = values.length;
  if (n < 15) return null;

  const fullSorted = [...values].sort((a, b) => a - b);
  const fullMin = fullSorted[0], fullMax = fullSorted[n - 1];
  const fullRange = (fullMax - fullMin) || 1;

  // 5th-95th percentile core -- same percentile-by-rank trim used
  // elsewhere in this file (NOT a symmetric MAD fence, which was found
  // wrong for skewed data -- see recommendedHistogramBins's comment).
  const percentileOf = p => {
    const idx = p * (fullSorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? fullSorted[lo] : fullSorted[lo] + (fullSorted[hi] - fullSorted[lo]) * (idx - lo);
  };
  const loP = percentileOf(0.05), hiP = percentileOf(0.95);
  const core = fullSorted.filter(v => v >= loP && v <= hiP); // already sorted, since fullSorted was
  if (core.length < 15) return null;
  const coreMin = core[0], coreMax = core[core.length - 1];
  const coreRange = (coreMax - coreMin) || 1;

  const coreMean = core.reduce((a, v) => a + v, 0) / core.length;
  const chosen = chooseErrorDistribution(core.map(v => v - coreMean));
  if (!chosen) return null;
  const { name: distName, params: distParams, k: distK } = chosen.best;

  let shapeFn = null;
  if (distName === 'Gaussian') { const { sigma } = distParams; shapeFn = d => Math.exp(-0.5 * (d / sigma) ** 2); }
  else if (distName === 'Laplace') { const { b } = distParams; shapeFn = d => Math.exp(-Math.abs(d) / b); }
  else if (distName === 'Student-t') { const { sigma, nu } = distParams; shapeFn = d => Math.pow(1 + (d * d) / (nu * sigma * sigma), -(nu + 1) / 2); }
  if (!shapeFn) return null;

  const STEPS = 300, dx = coreRange / STEPS;
  let area = 0;
  for (let s = 0; s <= STEPS; s++) {
    const v = coreMin + s * dx;
    const w = (s === 0 || s === STEPS) ? 0.5 : 1; // trapezoidal rule
    area += shapeFn(v - coreMean) * w * dx;
  }
  if (area <= 0) return null;

  // Index of the first core value >= v -- avoids rescanning the whole
  // core per candidate bin count the way a fresh counts-array pass would.
  const countBelow = v => {
    let lo = 0, hi = core.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (core[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
  };

  // Picks the GLOBAL LOWEST reduced chi-square among candidates with
  // enough degrees of freedom for that score to mean anything (df >=
  // MIN_GATE_DF), then gates ONLY that single winner before trusting it --
  // not a step-by-step walk that requires every candidate along the way
  // to individually pass. A sequential walk (reject at the first
  // candidate whose score exceeds a textbook per-step critical value) was
  // tried first and rejected: requiring dozens of consecutive per-bin-
  // count tests to all independently pass even a lenient per-step test
  // compounds to a MUCH stricter effective standard (0.9^30 is under 5%)
  // -- confirmed directly, it made the search fall back to the formula on
  // 9-10 of 10 replications even on clean, noise-free synthetic data with
  // no model misspecification at all. One gate check on one chosen
  // candidate doesn't have that compounding problem.
  //
  // The df floor below exists for a different, narrower reason: without
  // it, a very coarse, low-power candidate (e.g. bins=3, df=1) can
  // trivially win the global minimum simply because so few bins barely
  // constrains the fit at all, regardless of whether the distribution fit
  // is actually any good (confirmed directly: one noisy n=1000 Laplace
  // case picked bins=3 over bins=16-22's genuinely better scores purely
  // for this reason, and tiled out, that pick scored 9.9 against the
  // clean reference -- nearly 8x worse than the formula's 1.25).
  // Excluding low-df candidates from winning at all closes that gap.
  const MIN_GATE_DF = 2;
  const cappedMax = Math.min(maxCoreBins, Math.floor(core.length / 2));
  let best = null;
  for (let bins = minBins; bins <= cappedMax; bins++) {
    const binWidth = coreRange / bins;
    const scale = (core.length * binWidth) / area;
    let chiSq = 0, invalid = false, prevCount = 0;
    for (let i = 0; i < bins; i++) {
      const edge = i === bins - 1 ? core.length : countBelow(coreMin + (i + 1) * binWidth);
      const observed = edge - prevCount;
      prevCount = edge;
      const v = coreMin + (i + 0.5) * binWidth;
      const expected = shapeFn(v - coreMean) * scale;
      if (expected < 5) { invalid = true; break; } // too sparse for chi-square to mean anything at this bin count
      chiSq += (observed - expected) ** 2 / expected;
    }
    if (invalid) break; // once too sparse, every larger bin count will be too -- stop searching
    const df = bins - distK - 1;
    if (df < MIN_GATE_DF) continue; // too little power at this resolution for its score to mean anything -- not eligible to win
    const reduced = chiSq / df; // reduced chi-square: comparable across different bin counts, unlike raw chi-square
    if (!best || reduced < best.score) best = { bins, binWidth, score: reduced };
  }
  if (!best) return null; // no candidate ever reached enough degrees of freedom to be trustworthy

  // One gate check on the single winner, against an EMPIRICALLY
  // calibrated threshold -- not the textbook chi-square critical value
  // (tried first, rejected): that assumes a correctly-specified,
  // untruncated MLE fit, but this search deliberately fits the
  // distribution AFTER percentile-trimming the core (see this function's
  // other comments for why), and an ordinary MLE on a trimmed remainder
  // is itself a biased estimator of that remainder's own spread -- it
  // doesn't correct for having removed the tails first. That bias is real
  // and systematic, not a bug to fix here: calibrated directly (several
  // hundred genuinely correctly-specified synthetic cases, Gaussian/
  // Laplace/Student-t(5), n=100-3000), this search's own best-achievable
  // reduced chi-square on truly clean, matching data has a MEDIAN around
  // 1.7-2.2 (not the textbook ~1) and a p99 around 3.8-4.2 depending on n
  // -- so a gate built on the textbook ~1 expectation rejects the vast
  // majority of genuinely good fits as "too poor," which is exactly what
  // happened when this was first tried (fell back to the formula on clean
  // data almost every time). GATE_THRESHOLD below sits just above that
  // calibration's p99 across all tested n -- loose enough to pass nearly
  // all genuinely matching data, tight enough to still catch the real
  // misfit case that motivated this gate in the first place (that case's
  // best score was 4.4).
  const GATE_THRESHOLD = 4.2;
  if (best.score > GATE_THRESHOLD) return null;

  // Tile the FULL range with the winning width from the reliable core --
  // this is what makes the returned count line up with what drawHistogram
  // actually renders (always over the full min-max), not just the core.
  const totalBins = Math.round(fullRange / best.binWidth);
  return { bins: Math.max(minBins, Math.min(maxTotalBins, totalBins)), distName };
}

// A handful of fits below (RANSAC, Theil-Sen on large datasets, Least
// Median/Trimmed Squares, trimmed exponential) sample random point pairs
// instead of checking every one. They used the browser's own Math.random(),
// which is NOT seeded — so calling the exact same fit on the exact same
// data twice could pick a different winner each time (confirmed: RANSAC
// picked a different model and a check R² of 0.042 vs -0.198 on the same
// dataset across two runs). mulberry32 is a small, fast, deterministic PRNG
// — seeding it with a fixed constant makes every one of these fits a pure
// function of its inputs again: same data in, same result out, every time.
function createSeededRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state |= 0;
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const RNG_SEED = 0x5eed1234;

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    if (Math.abs(M[col][col]) < 1e-12) throw new Error('Could not solve — data may be degenerate (e.g. all the same X value).');

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= factor * M[col][k];
    }
  }

  return M.map((row, i) => row[n] / row[i]);
}

// A^-1, by solving A*x = e_j (the j-th column of the identity matrix) for
// each j with the Gaussian elimination above — reuses solveLinearSystem
// rather than a separate Gauss-Jordan implementation. Only ever called on
// small matrices (one row/column per fitted parameter, so 2-5 in almost
// every model here), so re-solving from scratch per column is cheap.
function invertMatrix(A) {
  const n = A.length;
  const inv = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let col = 0; col < n; col++) {
    const e = new Array(n).fill(0);
    e[col] = 1;
    const x = solveLinearSystem(A, e);
    for (let row = 0; row < n; row++) inv[row][col] = x[row];
  }
  return inv;
}

// --- Student's t distribution (for exact OLS confidence intervals) ---
// Standard Numerical-Recipes-style implementation: log-gamma via Lanczos,
// then the regularized incomplete beta function via its continued-fraction
// expansion, which gives the t-distribution's CDF directly (the textbook
// identity P(T<=t) = 1 - 0.5*I_x(df/2, 1/2), x = df/(df+t^2), for t>0, and
// the mirror image for t<0). There's no closed-form inverse, so the
// critical value for a given confidence level is found by bisection on
// that CDF — cheap, since it converges to double precision in ~60 steps
// and this only ever runs once per fit, not once per bootstrap resample.
function logGamma(x) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function betaContinuedFraction(x, a, b) {
  const MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? bt * betaContinuedFraction(x, a, b) / a
    : 1 - bt * betaContinuedFraction(1 - x, b, a) / b;
}

function studentTCDF(t, df) {
  const x = df / (df + t * t);
  const ib = regularizedIncompleteBeta(x, df / 2, 0.5);
  return t > 0 ? 1 - 0.5 * ib : 0.5 * ib;
}

function studentTInverseCDF(p, df) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;
  let lo = -1000, hi = 1000;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (studentTCDF(mid, df) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// The exact, closed-form confidence interval for ordinary-least-squares
// coefficients: CI = estimate +/- t(df, 1-alpha/2) * SE, where SE comes
// from the residual variance and the normal-equations matrix A = X^T*X
// (every OLS fit in this file already builds A to solve for its
// coefficients in the first place — this just reuses it instead of
// rebuilding the design matrix). "Exact" under the standard OLS assumption
// of normally-distributed errors; still the standard, well-established
// approximation otherwise. Unlike bootstrapParameterCI, this is a single
// matrix inversion — no resampling, no iteration count to tune, and it's
// deterministic (no seed).
function olsParameterCI(A, ssRes, n, coeffs, confidence = 0.95) {
  const p = A.length;
  const df = n - p;
  // Need at least 1 residual degree of freedom for a residual variance to
  // exist at all — a model with as many parameters as data points has
  // nothing left over to estimate uncertainty from.
  if (df < 1) return null;
  const s2 = Math.max(0, ssRes) / df;
  const Ainv = invertMatrix(A);
  const tCritical = studentTInverseCDF(1 - (1 - confidence) / 2, df);
  const coeffs_ = coeffs.map((c, j) => {
    const se = Math.sqrt(Math.max(0, s2 * Ainv[j][j]));
    // t-test, H0: "this coefficient is really 0" — same estimate/se this CI
    // already computed, just read the other way. Two-tailed, using the
    // exact same Student-t distribution (and the same df) the CI's own
    // tCritical comes from, so the two always agree: p < 0.05 here iff 0
    // falls outside the 95% CI above.
    const t = se > 0 ? c / se : (c === 0 ? 0 : Infinity);
    const pValue = se > 0 ? 2 * (1 - studentTCDF(Math.abs(t), df)) : (c === 0 ? 1 : 0);
    return { estimate: c, se, lower: c - tCritical * se, upper: c + tCritical * se, t, pValue };
  });
  // "coeffs" (not coeffCIs) to match bootstrapParameterCI's result shape —
  // formatParameterCIs renders both the same way regardless of which one
  // produced them.
  return { exact: true, df, tCritical, params: {}, coeffs: coeffs_ };
}

// F-distribution CDF, via the same regularized-incomplete-beta relationship
// Student's t CDF above already uses: P(F <= f) =
// I_{d1*f/(d1*f+d2)}(d1/2, d2/2). No new numerical machinery needed — this
// is just a different way of reading the same beta function.
function fDistributionCDF(f, d1, d2) {
  if (f <= 0) return 0;
  const x = (d1 * f) / (d1 * f + d2);
  return regularizedIncompleteBeta(x, d1 / 2, d2 / 2);
}

// Overall F-test for a fitted model: H0 is "none of the predictors help at
// all — this model does no better than always predicting the mean".
// Compares the fitted model's SSE against the intercept-only baseline's SSE
// (= SST), the textbook F = ((SST-SSE)/p) / (SSE/(n-p-1)), generalized to
// whatever predictedYs/p a caller hands in — same formula whether that's a
// straight line (p=1), a polynomial (p=degree), or a multi-variable fit (p
// = number of X columns). This is the model-level complement to the
// per-coefficient t-tests above: a model can have one "significant"
// coefficient and still fail this, or vice versa, when predictors are
// correlated with each other.
function overallFTest(ys, predictedYs, p, weights = null) {
  const n = ys.length;
  const w = weights || new Array(n).fill(1);
  const wSum = w.reduce((s, v) => s + v, 0);
  const yMean = ys.reduce((s, y, i) => s + w[i] * y, 0) / wSum;
  let sse = 0, sst = 0;
  for (let i = 0; i < n; i++) { sse += w[i] * (ys[i] - predictedYs[i]) ** 2; sst += w[i] * (ys[i] - yMean) ** 2; }
  const df1 = p, df2 = n - p - 1;
  if (df2 < 1 || sse <= 0 || sst <= sse) return null;
  const F = ((sst - sse) / df1) / (sse / df2);
  if (!isFinite(F) || F < 0) return null;
  const pValue = 1 - fDistributionCDF(F, df1, df2);
  return { F, df1, df2, pValue };
}

// Fits y = c0 + c1*x + c2*x^2 + ... + cDegree*x^degree
//
// `weights` (optional, one per point) turns this into Weighted Least
// Squares: every normal-equations entry and every squared residual gets
// scaled by that point's weight before summing, same as replacing a point
// of weight 5 with 5 identical copies of it. Omitting weights (or passing
// all 1s) reproduces plain OLS exactly — every sum below is just the
// unweighted sum in that case. Typical sources for the weight: sample size
// (more observations behind a point = more precision = more weight, so
// weight = n) or measurement error (weight = 1/error^2, the standard
// inverse-variance convention — a point with twice the uncertainty gets a
// quarter the say).
function fitPolynomial(xs, ys, degree, weights = null) {
  const n = xs.length;
  const w = weights || new Array(n).fill(1);
  const terms = degree + 1;

  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const b = new Array(terms).fill(0);

  for (let i = 0; i < n; i++) {
    const powers = [1];
    for (let p = 1; p <= degree * 2; p++) powers.push(powers[p - 1] * xs[i]);
    for (let r = 0; r < terms; r++) {
      for (let c = 0; c < terms; c++) A[r][c] += w[i] * powers[r + c];
      b[r] += w[i] * powers[r] * ys[i];
    }
  }

  const coeffs = solveLinearSystem(A, b);

  const predict = x => coeffs.reduce((sum, c, p) => sum + c * Math.pow(x, p), 0);

  const r2 = computeR2Weighted(xs, ys, predict, w);
  let ssRes = 0;
  for (let i = 0; i < n; i++) ssRes += w[i] * (ys[i] - predict(xs[i])) ** 2;

  // A is already X^T*W*X (that's exactly what the double loop above built,
  // one normal-equations entry at a time) — olsParameterCI reuses it
  // directly rather than reconstructing a design matrix from scratch. The
  // same formula (s^2 = weighted ssRes / df, Var(coeffs) = s^2 * A^-1) is
  // the standard WLS parameter covariance whenever weights are proportional
  // to 1/variance, so nothing in olsParameterCI itself needs to change.
  const ci = olsParameterCI(A, ssRes, n, coeffs);

  return { degree, coeffs, r2, predict, ci };
}

function formatEquation(coeffs) {
  return coeffs
    .map((c, p) => {
      const rounded = c.toFixed(4);
      if (p === 0) return rounded;
      if (p === 1) return `${rounded}*x`;
      return `${rounded}*x^${p}`;
    })
    .reverse()
    .join(' + ')
    .replace(/\+ -/g, '- ');
}

// Tries degrees 1..maxDegree and returns the fit with the best R²,
// unless a specific degree is requested.
function runRegression(xs, ys, { degree = null, maxDegree = 3, weights = null } = {}) {
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');
  if (xs.length < 2) throw new Error('Need at least 2 data points.');

  if (degree !== null) {
    if (xs.length < degree + 1) throw new Error(`Need at least ${degree + 1} points to fit a degree-${degree} curve.`);
    return fitPolynomial(xs, ys, degree, weights);
  }

  let best = null;
  for (let d = 1; d <= Math.min(maxDegree, xs.length - 1); d++) {
    const fit = fitPolynomial(xs, ys, d, weights);
    if (!best || fit.r2 > best.r2 + 0.001) best = fit; // require a real improvement, not noise
  }
  return best;
}

// Fits y = a*x — a straight line forced through the origin (no intercept
// term). Different from the regular linear fit whenever the data doesn't
// actually cross zero at x=0 — this is for when you specifically want that
// constraint (e.g. "zero input must mean zero output").
function fitProportional(xs, ys, weights = null) {
  const w = weights || new Array(xs.length).fill(1);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += w[i] * xs[i] * ys[i]; sxx += w[i] * xs[i] * xs[i]; }
  if (sxx === 0) throw new Error('Proportional fit needs at least one nonzero X value.');
  const a = sxy / sxx;
  const predict = x => a * x;
  let ssRes = 0;
  for (let i = 0; i < xs.length; i++) ssRes += w[i] * (ys[i] - a * xs[i]) ** 2;
  const ci = olsParameterCI([[sxx]], ssRes, xs.length, [a]);
  return { type: 'proportional', a, r2: computeR2Weighted(xs, ys, predict, w), predict, ci };
}

function formatProportionalEquation({ a }) {
  return `${a.toFixed(4)}*x`;
}

// Fits y = a / (1 + e^(-(b*x + c))) — an S-shaped logistic growth curve.
// Unlike the polynomial fit above, there's no closed-form solution for a, b, c,
// so this uses gradient descent. X and Y are normalized first (each param would
// otherwise live on a wildly different scale — a on the data's scale, b/c on a
// small "logit" scale — which makes plain gradient descent diverge).
function fitLogisticCurve(xs, ys, { iters = 4000, lr = 0.3 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points to fit an S-curve.');

  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const xMean = mean(xs);
  const xStd = Math.sqrt(mean(xs.map(x => (x - xMean) ** 2))) || 1;
  const xn = xs.map(x => (x - xMean) / xStd);

  const yScale = arrayMax(ys.map(Math.abs)) || 1;
  const yn = ys.map(y => y / yScale);

  // corr sign gives a reasonable starting direction for b; a=1 covers the
  // normalized Y range, c=0 centers the curve.
  let corr = 0;
  for (let i = 0; i < n; i++) corr += xn[i] * yn[i];
  let a = 1, b = corr >= 0 ? 1 : -1, c = 0;

  const sigma = z => 1 / (1 + Math.exp(-z));

  for (let it = 0; it < iters; it++) {
    let ga = 0, gb = 0, gc = 0;
    for (let i = 0; i < n; i++) {
      const z = b * xn[i] + c;
      const s = sigma(z);
      const err = a * s - yn[i];
      ga += err * s;
      gb += err * a * s * (1 - s) * xn[i];
      gc += err * a * s * (1 - s);
    }
    a -= lr * (2 * ga / n);
    b -= lr * (2 * gb / n);
    c -= lr * (2 * gc / n);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
      throw new Error('The S-curve fit did not converge — this data may not follow a logistic shape.');
    }
  }

  // Undo the x/y normalization to get parameters in the original data's units.
  const finalA = a * yScale;
  const finalB = b / xStd;
  const finalC = c - (b * xMean) / xStd;

  const predict = x => finalA / (1 + Math.exp(-(finalB * x + finalC)));

  const yMean = mean(ys);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - predict(xs[i])) ** 2;
    ssTot += (ys[i] - yMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { type: 'logistic-curve', a: finalA, b: finalB, c: finalC, r2, predict };
}

function formatLogisticEquation({ a, b, c }) {
  const sign = c >= 0 ? '+' : '-';
  return `${a.toFixed(4)} / (1 + e^(-(${b.toFixed(4)}*x ${sign} ${Math.abs(c).toFixed(4)})))`;
}

// Average log-loss (binary cross-entropy) of a predict(x) function against
// actual Y values — Y can be any fraction in [0,1] (a draw scored 0.5 is a
// perfectly valid soft label), not just exact 0/1. Predictions are clamped
// away from the exact 0/1 boundary since log(0) is -Infinity — an
// unclamped model that ever predicts a hard 0 or 1 and is wrong would
// otherwise blow the whole score up to Infinity.
function logLoss(xs, ys, predict) {
  const eps = 1e-9;
  let total = 0;
  for (let i = 0; i < xs.length; i++) {
    const p = Math.min(1 - eps, Math.max(eps, predict(xs[i])));
    total += -(ys[i] * Math.log(p) + (1 - ys[i]) * Math.log(1 - p));
  }
  return total / xs.length;
}

// Fits y = 1 / (1 + e^(-(b*x + c))) — the S-curve with its ceiling fixed to
// 1 (a genuine 0-1 probability, never an amplitude that could drift above 1
// on new data) — by directly minimizing log-loss via gradient descent
// (this is exactly textbook logistic regression), rather than the
// least-squares fit fitLogisticCurve uses. Only makes sense, and is only
// offered in the UI, when Y is entirely within [0,1] — a win/draw/loss
// chess result (1/0.5/0) is the motivating case, and log-loss handles a
// 0.5 "soft" label as a first-class value, not a rounding compromise.
function fitLogisticProbability(xs, ys, { iters = 8000, lr = 0.3 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points to fit an S-curve.');
  if (ys.some(y => y < 0 || y > 1)) {
    throw new Error('Log-loss fitting needs every Y value between 0 and 1.');
  }

  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const xMean = mean(xs);
  const xStd = Math.sqrt(mean(xs.map(x => (x - xMean) ** 2))) || 1;
  const xn = xs.map(x => (x - xMean) / xStd);

  const sigma = z => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));

  let b = 0, c = 0;
  for (let it = 0; it < iters; it++) {
    let gb = 0, gc = 0;
    for (let i = 0; i < n; i++) {
      // Gradient of average log-loss w.r.t. z = b*x + c simplifies to
      // (predicted probability - actual label) — the classic logistic
      // regression gradient, no chain-rule mess like the least-squares
      // fit above needs.
      const err = sigma(b * xn[i] + c) - ys[i];
      gb += err * xn[i];
      gc += err;
    }
    b -= lr * (gb / n);
    c -= lr * (gc / n);
    if (!Number.isFinite(b) || !Number.isFinite(c)) {
      throw new Error('The log-loss S-curve fit did not converge — this data may not follow a logistic shape.');
    }
  }

  const finalB = b / xStd;
  const finalC = c - (b * xMean) / xStd;
  const predict = x => sigma(finalB * x + finalC);

  return { type: 'logistic-probability', a: 1, b: finalB, c: finalC, logLoss: logLoss(xs, ys, predict), r2: computeR2(xs, ys, predict), predict };
}

// Multinomial logistic regression (softmax regression): generalizes
// fitLogisticProbability above from a SINGLE probability (treating
// win/draw/loss as one "expected score" curve, which conflates win and
// draw probability into one number) to K separate, independently-shaped
// probability curves, one per outcome class. This is the genuinely
// different thing a true 3-class model can do that a 1-curve "expected
// score" fit structurally can't: chess theory says draw probability isn't
// monotonic in rating difference (it PEAKS between closely-matched
// players and falls off toward both lopsided extremes), a shape a single
// sigmoid has no way to represent even in principle, no matter how it's
// tuned -- a sigmoid is monotonic by construction.
//
// Standard softmax-regression setup: K classes get K sets of (weight,
// bias), but only K-1 are actually free -- the K-th class's logit is
// pinned at 0 (classes[K-1] is the reference class), since only the
// RELATIVE sizes of the logits affect the softmax probabilities; leaving
// all K free would make the solution non-unique (adding the same constant
// to every class's logit changes nothing). Fit by gradient descent on the
// categorical cross-entropy (log-loss generalized to K>2 classes) --
// same gradient form as fitLogisticProbability's binary case
// (predicted probability - actual indicator), just one such gradient per
// (class, non-reference) pair instead of one.
function fitMultinomialLogistic(xs, labels, { classLabels = null, iters = 4000, lr = 0.3 } = {}) {
  const n = xs.length;
  if (n < 8) throw new Error('Need at least 8 data points to fit a multinomial logistic model.');
  const classes = classLabels || Array.from(new Set(labels)).sort((a, b) => a - b);
  const K = classes.length;
  if (K < 2) throw new Error('Need at least 2 distinct outcome classes.');
  const classIdx = labels.map(l => classes.indexOf(l));
  if (classIdx.some(i => i < 0)) throw new Error('Found a label value not in classLabels.');

  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const xMean = mean(xs);
  const xStd = Math.sqrt(mean(xs.map(x => (x - xMean) ** 2))) || 1;
  const xn = xs.map(x => (x - xMean) / xStd);

  let w = new Array(K - 1).fill(0), b = new Array(K - 1).fill(0);
  function softmaxProbs(xVal) {
    const logits = new Array(K).fill(0); // last entry (reference class) stays pinned at 0
    for (let c = 0; c < K - 1; c++) logits[c] = w[c] * xVal + b[c];
    const maxLogit = Math.max(...logits); // subtract the max before exponentiating, for numerical stability -- standard softmax trick, doesn't change the result
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sum = exps.reduce((s, v) => s + v, 0);
    return exps.map(e => e / sum);
  }
  for (let it = 0; it < iters; it++) {
    const gw = new Array(K - 1).fill(0), gb = new Array(K - 1).fill(0);
    for (let i = 0; i < n; i++) {
      const probs = softmaxProbs(xn[i]);
      for (let c = 0; c < K - 1; c++) {
        const err = probs[c] - (classIdx[i] === c ? 1 : 0);
        gw[c] += err * xn[i];
        gb[c] += err;
      }
    }
    for (let c = 0; c < K - 1; c++) { w[c] -= lr * (gw[c] / n); b[c] -= lr * (gb[c] / n); }
    if (w.some(v => !Number.isFinite(v)) || b.some(v => !Number.isFinite(v))) {
      throw new Error('The multinomial logistic fit did not converge.');
    }
  }

  const finalW = w.map(v => v / xStd);
  const finalB = b.map((v, c) => v - (w[c] * xMean) / xStd);
  function predictProbsAt(x) {
    const logits = new Array(K).fill(0);
    for (let c = 0; c < K - 1; c++) logits[c] = finalW[c] * x + finalB[c];
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sum = exps.reduce((s, v) => s + v, 0);
    return exps.map(e => e / sum);
  }
  let correct = 0, totalLogLoss = 0;
  for (let i = 0; i < n; i++) {
    const probs = predictProbsAt(xs[i]);
    const predictedClass = probs.indexOf(Math.max(...probs));
    if (predictedClass === classIdx[i]) correct++;
    totalLogLoss -= Math.log(Math.max(1e-12, probs[classIdx[i]]));
  }
  return {
    type: 'multinomial-logistic', classes, w: finalW, b: finalB,
    accuracy: correct / n, logLoss: totalLogLoss / n,
    predictProbs: predictProbsAt,
    predictClass: x => classes[predictProbsAt(x).indexOf(Math.max(...predictProbsAt(x)))],
  };
}

// Fits y = a/(1 + e^(-(b*x + c))) + d — the S-curve above, but able to sit
// on a floor other than 0 (it ranges from d up to a+d instead of 0 to a).
// Adam (adaptive per-parameter step sizes) and multi-start over a/b's signs
// — same reasoning as the exponential-with-offset fit: an extra free
// parameter makes this surface harder to optimize than the plain S-curve,
// and whether the curve rises or falls depends on the sign of a and b
// together, not either one alone.
function fitLogisticOffset(xs, ys, { iters = 4000, lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
  const n = xs.length;
  if (n < 5) throw new Error('Need at least 5 data points to fit an S-curve with offset.');

  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const xMean = mean(xs);
  const xStd = Math.sqrt(mean(xs.map(x => (x - xMean) ** 2))) || 1;
  const xn = xs.map(x => (x - xMean) / xStd);

  const yScale = arrayMax(ys.map(Math.abs)) || 1;
  const yn = ys.map(y => y / yScale);
  const yMeanN = mean(yn);

  const sigma = z => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));

  let best = null;
  for (const a0 of [1, -1]) {
    for (const b0 of [1, -1]) {
      let a = a0, b = b0, c = 0, d = yMeanN;
      let ma = 0, mb = 0, mc = 0, md = 0, va = 0, vb = 0, vc = 0, vd = 0;
      let diverged = false;

      for (let it = 1; it <= iters; it++) {
        let ga = 0, gb = 0, gc = 0, gd = 0;
        for (let i = 0; i < n; i++) {
          const s = sigma(b * xn[i] + c);
          const err = (a * s + d) - yn[i];
          ga += err * s;
          gb += err * a * s * (1 - s) * xn[i];
          gc += err * a * s * (1 - s);
          gd += err;
        }
        ga = 2 * ga / n; gb = 2 * gb / n; gc = 2 * gc / n; gd = 2 * gd / n;

        ma = beta1 * ma + (1 - beta1) * ga; va = beta2 * va + (1 - beta2) * ga * ga;
        mb = beta1 * mb + (1 - beta1) * gb; vb = beta2 * vb + (1 - beta2) * gb * gb;
        mc = beta1 * mc + (1 - beta1) * gc; vc = beta2 * vc + (1 - beta2) * gc * gc;
        md = beta1 * md + (1 - beta1) * gd; vd = beta2 * vd + (1 - beta2) * gd * gd;

        const bc1 = 1 - beta1 ** it, bc2 = 1 - beta2 ** it;
        a -= lr * (ma / bc1) / (Math.sqrt(va / bc2) + eps);
        b -= lr * (mb / bc1) / (Math.sqrt(vb / bc2) + eps);
        c -= lr * (mc / bc1) / (Math.sqrt(vc / bc2) + eps);
        d -= lr * (md / bc1) / (Math.sqrt(vd / bc2) + eps);
        if (![a, b, c, d].every(Number.isFinite)) { diverged = true; break; }
      }
      if (diverged) continue;

      let ssRes = 0, ssTot = 0;
      for (let i = 0; i < n; i++) {
        ssRes += (yn[i] - (a * sigma(b * xn[i] + c) + d)) ** 2;
        ssTot += (yn[i] - yMeanN) ** 2;
      }
      const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
      if (!best || r2 > best.r2) best = { a, b, c, d, r2 };
    }
  }
  if (!best) throw new Error('The S-curve-with-offset fit did not converge for this data.');

  // Undo the x normalization: b*xn + c = b*(x-xMean)/xStd + c
  //                                     = (b/xStd)*x + (c - b*xMean/xStd)
  const finalB = best.b / xStd;
  const finalC = best.c - (best.b * xMean) / xStd;
  const finalA = best.a * yScale;
  const finalD = best.d * yScale;

  const predict = x => finalA / (1 + Math.exp(-(finalB * x + finalC))) + finalD;
  return { type: 'logistic-offset', a: finalA, b: finalB, c: finalC, d: finalD, r2: computeR2(xs, ys, predict), predict };
}

function formatLogisticOffsetEquation({ a, b, c, d }) {
  const cSign = c >= 0 ? '+' : '-';
  const dSign = d >= 0 ? '+' : '-';
  return `${a.toFixed(4)} / (1 + e^(-(${b.toFixed(4)}*x ${cSign} ${Math.abs(c).toFixed(4)}))) ${dSign} ${Math.abs(d).toFixed(4)}`;
}

// Splits the data into two groups by Y value — points equal to
// `constantValue` (a flat stretch, e.g. all zeros) and everything else — and
// fits a flat line to the first group, a straight line to the second. The
// fitted line only applies for X between the lowest and highest X seen among
// the non-constant points; everywhere else predict() just returns the
// constant. Good for data like "0% most of the time, ramping down over a
// specific stretch, then 0% again."
function fitPiecewiseConstantLinear(xs, ys, { constantValue = null } = {}) {
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');

  if (constantValue === null) {
    const counts = new Map();
    ys.forEach(y => counts.set(y, (counts.get(y) || 0) + 1));
    let bestVal = ys[0], bestCount = 0;
    for (const [val, count] of counts) {
      if (count > bestCount) { bestVal = val; bestCount = count; }
    }
    constantValue = bestVal;
  }

  const linXs = [], linYs = [];
  xs.forEach((x, i) => { if (ys[i] !== constantValue) { linXs.push(x); linYs.push(ys[i]); } });
  const constPoints = xs.length - linXs.length;

  // Both halves of the piecewise shape need enough points to actually mean
  // something — a "constant" backed by 1-2 points, or a "line" fit through
  // 1-2 points, is just noise wearing a shape. Require at least 3 points on
  // each side (6 total) or skip the model entirely rather than let it win
  // on an unearned perfect fit to a handful of points.
  if (constPoints < 3 || linXs.length < 3) {
    throw new Error(`Need at least 3 points on both the constant part (got ${constPoints}) and the linear part (got ${linXs.length}) to fit a piecewise constant+linear model.`);
  }

  const lineFit = fitPolynomial(linXs, linYs, 1);
  const activeMin = arrayMin(linXs);
  const activeMax = arrayMax(linXs);

  const predict = x => (x >= activeMin && x <= activeMax) ? lineFit.predict(x) : constantValue;

  const yMean = ys.reduce((a, v) => a + v, 0) / ys.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < xs.length; i++) {
    ssRes += (ys[i] - predict(xs[i])) ** 2;
    ssTot += (ys[i] - yMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return {
    type: 'piecewise-constant-linear',
    constantValue,
    activeMin,
    activeMax,
    slope: lineFit.coeffs[1],
    intercept: lineFit.coeffs[0],
    r2,
    predict,
  };
}

function formatPiecewiseEquation(fit) {
  return `${fit.constantValue}  (for x outside [${fit.activeMin}, ${fit.activeMax}])\n` +
         `  ${fit.slope.toFixed(4)}*x + ${fit.intercept.toFixed(4)}  (for x in [${fit.activeMin}, ${fit.activeMax}])`;
}

// Restricted ("natural") cubic spline regression: y = b0 + b1*x + sum_j
// bj*Xj, where each Xj is a cubic basis term tied to one interior knot
// (Harrell's standard construction). The key property this is built for:
// the fitted curve is a smooth cubic BETWEEN knots, but turns perfectly
// LINEAR before the first knot and after the last one. A plain,
// unrestricted spline doesn't have that property -- its last cubic piece
// can shoot off in any direction just past the edge of the data. Since
// this tool extrapolates fits beyond the observed X range, that would be
// unsafe; the restriction trades a little flexibility at the very ends for
// predictable, bounded extrapolation.
//
// Knot count defaults to a conservative function of N (3..6, roughly one
// knot per 6 points) since each extra knot is another parameter free to
// chase noise -- the same caution fitPiecewiseConstantLinear above takes
// with its "need 3 points on both sides" guard.
function fitNaturalCubicSpline(xs, ys, { numKnots = null } = {}) {
  const n = xs.length;
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');
  if (numKnots === null) numKnots = Math.max(3, Math.min(6, Math.floor(n / 6)));
  if (numKnots < 3) throw new Error(`Need at least 3 knots for a natural cubic spline (only room for ${numKnots} with ${n} points).`);

  // One extra parameter of slack beyond the knot count itself -- a flexible
  // shape fit to barely more points than it has parameters is just
  // interpolation wearing a model's clothes.
  const minPoints = numKnots + 5;
  if (n < minPoints) throw new Error(`Need at least ${minPoints} points for a ${numKnots}-knot spline (got ${n}).`);

  const sortedX = [...xs].sort((a, b) => a - b);
  const knots = [];
  for (let i = 0; i < numKnots; i++) {
    const p = i / (numKnots - 1); // evenly spaced percentiles, 0..1
    const idx = p * (n - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    knots.push(lo === hi ? sortedX[lo] : sortedX[lo] + (idx - lo) * (sortedX[hi] - sortedX[lo]));
  }
  // Guard against ties collapsing two knots onto the same X value (e.g. many
  // repeated X's) -- a zero-width span between knots divides by zero below.
  for (let i = 1; i < knots.length; i++) {
    if (knots[i] <= knots[i - 1]) knots[i] = knots[i - 1] + 1e-9;
  }

  const tLast = knots[numKnots - 1], tPenult = knots[numKnots - 2];
  const span = tLast - tPenult;
  const nonlinearTerms = numKnots - 2;
  const terms = 2 + nonlinearTerms; // intercept + linear + nonlinear basis terms

  function row(x) {
    const cube = v => (v > 0 ? v * v * v : 0);
    const r = [1, x];
    for (let j = 0; j < nonlinearTerms; j++) {
      const tj = knots[j];
      r.push(cube(x - tj) - cube(x - tPenult) * (tLast - tj) / span + cube(x - tLast) * (tPenult - tj) / span);
    }
    return r;
  }

  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const b = new Array(terms).fill(0);
  for (let i = 0; i < n; i++) {
    const r = row(xs[i]);
    for (let a = 0; a < terms; a++) {
      for (let c = 0; c < terms; c++) A[a][c] += r[a] * r[c];
      b[a] += r[a] * ys[i];
    }
  }

  const coeffs = solveLinearSystem(A, b);
  const predict = x => row(x).reduce((sum, v, idx) => sum + v * coeffs[idx], 0);
  const r2 = computeR2(xs, ys, predict);

  return { type: 'natural-cubic-spline', knots, coeffs, r2, predict, params: terms };
}

function formatSplineEquation(fit) {
  return `natural cubic spline, ${fit.knots.length} knots at x = [${fit.knots.map(k => k.toFixed(2)).join(', ')}] (linear beyond the first/last knot)`;
}

// Pool Adjacent Violators (PAVA), weighted -- the standard O(n) algorithm
// for isotonic regression. Walks the already-ordered values left to right,
// keeping a stack of "blocks" (each a run of points averaged into one
// level). Whenever the new point's value would be LOWER than the block
// before it (a "violation" of the non-decreasing constraint), it merges
// that block into the previous one (weighted average) and keeps merging
// backward until the constraint holds again. What's left when every point
// has been processed is the closest non-decreasing step function to the
// data, in the least-squares sense -- provably optimal, not a heuristic.
function poolAdjacentViolators(values, weights) {
  const n = values.length;
  const blockSum = [], blockWeight = [], blockCount = [];
  for (let i = 0; i < n; i++) {
    blockSum.push(values[i] * weights[i]);
    blockWeight.push(weights[i]);
    blockCount.push(1);
    while (blockSum.length > 1) {
      const top = blockSum.length - 1;
      const meanTop = blockSum[top] / blockWeight[top];
      const meanPrev = blockSum[top - 1] / blockWeight[top - 1];
      if (meanPrev <= meanTop) break;
      blockSum[top - 1] += blockSum[top];
      blockWeight[top - 1] += blockWeight[top];
      blockCount[top - 1] += blockCount[top];
      blockSum.pop(); blockWeight.pop(); blockCount.pop();
    }
  }
  const fitted = new Array(n);
  let idx = 0;
  for (let b = 0; b < blockSum.length; b++) {
    const mean = blockSum[b] / blockWeight[b];
    for (let c = 0; c < blockCount[b]; c++) fitted[idx++] = mean;
  }
  return fitted;
}

// Isotonic regression: the closest curve to the data that's only allowed to
// move in ONE direction as X increases (always non-decreasing, or always
// non-increasing -- whichever direction fits better is picked
// automatically unless `increasing` is forced). No formula, no smoothness
// assumption -- just a staircase of flat steps that respects that one rule
// and otherwise follows the data as closely as possible. Good for data you
// KNOW should only climb (or only fall), but whose exact shape you don't
// want to guess at (could jump, could level off, could creep). Bad for
// anything cyclical or genuinely reversing -- it will flatten that out.
// Repeated X values are pre-averaged (weighted by how many share that X) so
// the fitted step function stays a proper function of X; between knots, the
// prediction is linearly interpolated (still exactly monotonic, since the
// knot values already are); past the first/last knot, it holds flat rather
// than extrapolating a slope -- the safe, shape-respecting choice.
function fitIsotonic(xs, ys, { increasing = null } = {}) {
  const n = xs.length;
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');
  if (n < 5) throw new Error(`Need at least 5 points to fit an isotonic curve (got ${n}).`);

  const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  const sortedX = order.map(i => xs[i]);
  const sortedY = order.map(i => ys[i]);

  const knotX = [], groupedY = [], groupedW = [];
  for (let i = 0; i < n; i++) {
    if (knotX.length && knotX[knotX.length - 1] === sortedX[i]) {
      const last = knotX.length - 1;
      groupedY[last] = (groupedY[last] * groupedW[last] + sortedY[i]) / (groupedW[last] + 1);
      groupedW[last] += 1;
    } else {
      knotX.push(sortedX[i]);
      groupedY.push(sortedY[i]);
      groupedW.push(1);
    }
  }

  function predictFrom(knotYs) {
    return x => {
      if (x <= knotX[0]) return knotYs[0];
      if (x >= knotX[knotX.length - 1]) return knotYs[knotX.length - 1];
      let lo = 0, hi = knotX.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (knotX[mid] <= x) lo = mid; else hi = mid;
      }
      const t = (x - knotX[lo]) / (knotX[hi] - knotX[lo]);
      return knotYs[lo] + t * (knotYs[hi] - knotYs[lo]);
    };
  }

  const directions = increasing === null ? [true, false] : [increasing];
  let best = null;
  for (const inc of directions) {
    const vals = inc ? groupedY : groupedY.map(v => -v);
    const fittedRaw = poolAdjacentViolators(vals, groupedW);
    const knotYs = inc ? fittedRaw : fittedRaw.map(v => -v);
    const predict = predictFrom(knotYs);
    const r2 = computeR2(xs, ys, predict);
    if (!best || r2 > best.r2) best = { increasing: inc, knotYs, predict, r2 };
  }

  // Effective parameter count for adjusted-R²/AIC-style comparisons: the
  // number of distinct flat levels actually used, not the raw knot count --
  // a 20-point fit that pooled down to 3 levels is only about as flexible
  // as a 3-parameter model, not a 20-parameter one.
  const distinctLevels = new Set(best.knotYs.map(v => v.toFixed(9))).size;

  return { type: 'isotonic', increasing: best.increasing, knots: knotX, values: best.knotYs, r2: best.r2, predict: best.predict, params: Math.max(2, distinctLevels) };
}

function formatIsotonicEquation(fit) {
  return `isotonic (${fit.increasing ? 'non-decreasing' : 'non-increasing'}) step fit through ${fit.knots.length} x-values, ${fit.params} distinct levels (flat beyond the data range)`;
}

// Nadaraya-Watson kernel regression: at each query X, predict a weighted
// average of every Y in the data, where the weight falls off the farther
// that point's X is from the query (a Gaussian bell curve centered on the
// query X). No formula and no shape constraint at all -- purely "look at
// whatever's nearby and average it," which makes it flexible in a totally
// different way than the spline/isotonic fits above: it has no global
// structure, just local smoothing.
//
// `bandwidth` controls how wide "nearby" means. Too narrow and it just
// memorizes each point (overfits); too wide and it washes out into one
// flat average. Rather than pick a fixed default, the bandwidth is chosen
// per-dataset by leave-one-out cross-validation over a grid spanning very
// local to very smooth -- the same "let the data pick the knob" approach
// knnPredictionInterval elsewhere in this file uses, rather than a single
// magic constant that only suits some datasets.
//
// Like isotonic above, this has no sensible way to extrapolate -- the
// Gaussian weights vanish once X is many bandwidths outside the data, so
// prediction just holds flat at the boundary value past the edges, rather
// than guessing at a trend the method has no basis for.
function fitKernelRegression(xs, ys, { bandwidth = null } = {}) {
  const n = xs.length;
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');
  if (n < 6) throw new Error(`Need at least 6 points to fit a kernel regression (got ${n}).`);

  const xMin = arrayMin(xs), xMax = arrayMax(xs);
  const range = (xMax - xMin) || 1;

  function rawPredict(h) {
    return x => {
      let wSum = 0, wySum = 0;
      for (let i = 0; i < n; i++) {
        const z = (x - xs[i]) / h;
        const w = Math.exp(-0.5 * z * z);
        wSum += w; wySum += w * ys[i];
      }
      if (wSum < 1e-300) {
        let bestI = 0, bestD = Infinity;
        for (let i = 0; i < n; i++) { const d = Math.abs(x - xs[i]); if (d < bestD) { bestD = d; bestI = i; } }
        return ys[bestI];
      }
      return wySum / wSum;
    };
  }

  function clampedPredict(h) {
    const raw = rawPredict(h);
    return x => raw(x < xMin ? xMin : (x > xMax ? xMax : x));
  }

  let bestH = bandwidth, bestParams;
  if (bestH === null) {
    const grid = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0].map(f => f * range);
    let bestCV = Infinity;
    for (const h of grid) {
      let sse = 0, count = 0;
      for (let i = 0; i < n; i++) {
        let wSum = 0, wySum = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const z = (xs[i] - xs[j]) / h;
          const w = Math.exp(-0.5 * z * z);
          wSum += w; wySum += w * ys[j];
        }
        if (wSum < 1e-300) continue; // no neighbors at this bandwidth -- leave-one-out undefined for this point
        sse += (ys[i] - wySum / wSum) ** 2;
        count++;
      }
      if (count < n * 0.5) continue; // this bandwidth leaves too many points undefined to trust
      const cv = sse / count;
      if (cv < bestCV) { bestCV = cv; bestH = h; }
    }
    if (bestH === null) bestH = 0.25 * range; // every grid value failed the coverage check -- fall back to a moderate default
  }

  const predict = clampedPredict(bestH);
  const r2 = computeR2(xs, ys, predict);
  // Effective parameter count, same spirit as isotonic's distinct-level
  // count above: a narrower bandwidth acts like more independent local
  // "parameters" -- roughly how many non-overlapping bandwidth-widths fit
  // across the data's range, capped so it never claims more parameters
  // than there are data points.
  bestParams = Math.max(2, Math.min(n, Math.round(range / bestH)));

  return { type: 'kernel', bandwidth: bestH, r2, predict, params: bestParams };
}

function formatKernelEquation(fit) {
  return `Nadaraya-Watson kernel regression, bandwidth=${fit.bandwidth.toFixed(3)} (chosen by leave-one-out cross-validation; flat beyond the data range)`;
}

// R² in the original Y units, given any predict(x) function — used by the
// three curve-shape fits below, since each one fits a straight line to
// *transformed* data (so a plain fitPolynomial's own r2 would be measuring
// fit quality in the wrong (transformed) units, not the data's real units).
function computeR2(xs, ys, predict) {
  const yMean = ys.reduce((a, v) => a + v, 0) / ys.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < xs.length; i++) {
    ssRes += (ys[i] - predict(xs[i])) ** 2;
    ssTot += (ys[i] - yMean) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Weighted R²: same idea as computeR2, but every squared term (both the
// "unexplained" residual and the "total" spread around the mean) is scaled
// by that point's weight first — so a point given weight 5 counts as much
// toward both numerator and denominator as 5 copies of an ordinary point
// would. The mean itself is also weighted, for the same reason a weighted
// average uses weights in its denominator too. Reduces to plain computeR2
// exactly when every weight is 1 (verified by construction: sumW becomes n,
// yMeanW becomes the plain mean, and every term is unscaled).
function computeR2Weighted(xs, ys, predict, weights) {
  const sumW = weights.reduce((s, w) => s + w, 0);
  const yMeanW = ys.reduce((s, v, i) => s + weights[i] * v, 0) / sumW;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < xs.length; i++) {
    ssRes += weights[i] * (ys[i] - predict(xs[i])) ** 2;
    ssTot += weights[i] * (ys[i] - yMeanW) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Same idea as computeR2, but the denominator (ssTot, "how much is there to
// explain") comes from the FULL dataset's variance, not just the variance of
// whichever points are being scored. Plain computeR2 on a held-out test
// slice re-centers itself on that slice's own mean — if the slice happens to
// have near-zero variance (e.g. the tail of a curve that's already
// plateaued), even a small absolute miss gets divided by a near-zero
// denominator and produces an astronomically bad score that has nothing to
// do with how wrong the prediction actually was. Confirmed on this data:
// "PSI vs mean Polsby-Popper score"'s held-out test R² was -9408 by plain
// computeR2 (ssTot from 3 nearly-identical test points), because the model
// missed by only ~0.03-0.07 in absolute terms — computeR2FullVariance scores
// that same miss against the dataset's real overall spread instead, which is
// a fixed, stable "baseline to beat" per dataset rather than a moving target
// set by chance depending on which points landed in the holdout.
function computeR2FullVariance(allYs, testXs, testYs, predict) {
  const fullMean = allYs.reduce((a, v) => a + v, 0) / allYs.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < testXs.length; i++) ssRes += (testYs[i] - predict(testXs[i])) ** 2;
  for (let i = 0; i < allYs.length; i++) ssTot += (allYs[i] - fullMean) ** 2;
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Sequential (walk-forward) version of computeR2 / computeR2FullVariance:
// instead of calling predict(x) with no other context, each step also
// passes the REAL previous row's y (startY for the first test row, then the
// real testYs[i-1] for every row after) as a second argument. Every
// candidate's predict() either uses this (currently just AR(1), to do a
// genuine one-step-ahead forecast instead of a blind multi-step recursion)
// or ignores it (every other candidate type, unaffected). No leakage: the
// value passed for predicting row i is always from row i-1 or earlier,
// never row i itself or anything later — by the time predict() is asked
// about row i, row i's own true value hasn't been looked at yet.
function computeR2Sequential(startY, testXs, testYs, predict) {
  const yMean = testYs.reduce((a, v) => a + v, 0) / testYs.length;
  let ssRes = 0, ssTot = 0, prevY = startY;
  for (let i = 0; i < testXs.length; i++) {
    ssRes += (testYs[i] - predict(testXs[i], prevY)) ** 2;
    ssTot += (testYs[i] - yMean) ** 2;
    prevY = testYs[i];
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}
function computeR2FullVarianceSequential(allYs, startY, testXs, testYs, predict) {
  const fullMean = allYs.reduce((a, v) => a + v, 0) / allYs.length;
  let ssRes = 0, ssTot = 0, prevY = startY;
  for (let i = 0; i < testXs.length; i++) {
    ssRes += (testYs[i] - predict(testXs[i], prevY)) ** 2;
    prevY = testYs[i];
  }
  for (let i = 0; i < allYs.length; i++) ssTot += (allYs[i] - fullMean) ** 2;
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Fits y = a * e^(b*x) — steady growth/decay by a constant percentage each
// step. Linearized as ln(y) = ln(a) + b*x, so it reuses the straight-line
// fit above; only works when every Y is positive (e^anything is never <= 0).
function fitExponential(xs, ys, weights = null) {
  if (ys.some(y => y <= 0)) throw new Error('Exponential fit needs every Y value to be greater than 0.');
  const w = weights || new Array(xs.length).fill(1);
  // y = a*e^(bx) is really just OLS in disguise: ln(y) = ln(a) + b*x is a
  // straight line, so fitPolynomial's own exact CI (on ln(a) and b, in
  // that transformed space) is reused below rather than re-deriving one —
  // exp() is monotonic, so exponentiating the ln(a) bounds is still an
  // exact-coverage CI for a itself, and b needs no transform at all since
  // it's the same number in both spaces. Weights pass straight through —
  // they describe confidence in each (x, y) pair, which doesn't change
  // just because y got log-transformed for fitting purposes.
  const lineFit = fitPolynomial(xs, ys.map(Math.log), 1, w);
  const a = Math.exp(lineFit.coeffs[0]);
  const b = lineFit.coeffs[1];
  const predict = x => a * Math.exp(b * x);
  const ci = lineFit.ci && {
    exact: true, df: lineFit.ci.df, tCritical: lineFit.ci.tCritical,
    params: {
      a: { lower: Math.exp(lineFit.ci.coeffs[0].lower), upper: Math.exp(lineFit.ci.coeffs[0].upper) },
      b: { lower: lineFit.ci.coeffs[1].lower, upper: lineFit.ci.coeffs[1].upper },
    },
  };
  return { type: 'exponential', a, b, r2: computeR2Weighted(xs, ys, predict, w), predict, ci };
}

// --- Multiple independent variables ---
// X here is an array of ROWS, one per data point, each row itself an array
// of that point's values across all the independent variables — e.g.
// X[i] = [x1_i, x2_i, x3_i]. There's always exactly ONE constant/intercept
// for the whole fit, not one per variable — the same role `c` plays in the
// single-variable fits above, just shared across every predictor instead
// of belonging to any one of them.

// R² for a predict(xRow) function that takes a whole row (multiple
// variables) instead of a single x — same math as computeR2 above, just
// indexed by row instead of by scalar x.
function computeR2Multi(X, y, predict) {
  const n = y.length;
  const yMean = y.reduce((s, v) => s + v, 0) / n;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (y[i] - predict(X[i])) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Weighted version, same relationship to computeR2Multi as
// computeR2Weighted has to computeR2.
function computeR2MultiWeighted(X, y, predict, weights) {
  const n = y.length;
  const sumW = weights.reduce((s, w) => s + w, 0);
  const yMeanW = y.reduce((s, v, i) => s + weights[i] * v, 0) / sumW;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += weights[i] * (y[i] - predict(X[i])) ** 2;
    ssTot += weights[i] * (y[i] - yMeanW) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Ordinary least squares with any number of predictors: y = c + b1*x1 +
// b2*x2 + ... — the multivariate generalization of fitPolynomial's normal
// equations (design matrix [1, x1, x2, ...] instead of [1, x, x^2, ...]),
// reusing the same solveLinearSystem Gaussian elimination. This is the
// building block both fitMultiLeastTrimmedSquares and fitMultiExponential
// are made of.
function fitMultiLinear(X, y, weights = null) {
  const n = X.length;
  const k = X[0].length;
  const terms = k + 1;
  const w = weights || new Array(n).fill(1);

  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const b = new Array(terms).fill(0);
  for (let i = 0; i < n; i++) {
    const row = [1, ...X[i]];
    for (let r = 0; r < terms; r++) {
      for (let c = 0; c < terms; c++) A[r][c] += w[i] * row[r] * row[c];
      b[r] += w[i] * row[r] * y[i];
    }
  }

  const solved = solveLinearSystem(A, b);
  const intercept = solved[0];
  const coeffs = solved.slice(1);
  const predict = xRow => intercept + xRow.reduce((s, v, j) => s + v * coeffs[j], 0);

  let ssRes = 0;
  for (let i = 0; i < n; i++) ssRes += w[i] * (y[i] - predict(X[i])) ** 2;
  // olsParameterCI's coeffs order must match A/b's row order: [intercept, ...coeffs].
  const ci = olsParameterCI(A, ssRes, n, [intercept, ...coeffs]);

  return { intercept, coeffs, r2: computeR2MultiWeighted(X, y, predict, w), predict, ci };
}

// Multivariate Least Trimmed Squares: the same elemental-subset approach as
// the single-variable fitLeastTrimmedSquares below (fit an exact line
// through a random minimal subset, score it by trimmed squared residuals
// over ALL points, keep the best, then refit ordinary least squares on
// just the h points that subset says are inliers) — generalized so the
// "line" is a hyperplane through p = (number of variables + 1) points
// instead of 2. One shared intercept, never one per variable — trimming
// which POINTS count as outliers is a separate question from how many
// predictor variables there are, so there's nothing to "repeat" here.
function fitMultiLeastTrimmedSquares(X, y, { trimFraction = 0.90, iterations = 500, seed = RNG_SEED } = {}) {
  const n = X.length;
  if (!n) throw new Error('Need at least some data points for a multi-variable fit.');
  const k = X[0].length;
  const p = k + 1; // parameters to solve for: 1 intercept + k slopes
  if (n < p + 2) throw new Error(`Need at least ${p + 2} data points for a ${k}-variable least-trimmed-squares fit.`);
  const h = Math.max(p + 1, Math.round(n * trimFraction));
  const random = createSeededRandom(seed);

  let best = null;
  for (let it = 0; it < iterations; it++) {
    const idx = new Set();
    let attempts = 0;
    while (idx.size < p && attempts < 50) { idx.add(Math.floor(random() * n)); attempts++; }
    if (idx.size < p) continue;
    const rows = [...idx];

    const A = rows.map(i => [1, ...X[i]]);
    const bVec = rows.map(i => y[i]);
    let beta;
    try { beta = solveLinearSystem(A, bVec); } catch (e) { continue; } // subset was degenerate (e.g. collinear points) — try another

    const predict = xRow => beta[0] + xRow.reduce((s, v, j) => s + v * beta[j + 1], 0);
    const sqResiduals = X.map((xRow, i) => (y[i] - predict(xRow)) ** 2);
    const trimmedSum = [...sqResiduals].sort((a, c) => a - c).slice(0, h).reduce((s, v) => s + v, 0);
    if (!best || trimmedSum < best.trimmedSum) best = { beta, trimmedSum };
  }
  if (!best) throw new Error('Could not find a valid fit through this data — check for duplicate or collinear rows.');

  const bestPredict = xRow => best.beta[0] + xRow.reduce((s, v, j) => s + v * best.beta[j + 1], 0);
  const absResiduals = X.map((xRow, i) => Math.abs(y[i] - bestPredict(xRow)));
  const trimmedIdx = absResiduals.map((r, i) => i).sort((a, c) => absResiduals[a] - absResiduals[c]).slice(0, h);
  const refit = fitMultiLinear(trimmedIdx.map(i => X[i]), trimmedIdx.map(i => y[i]));

  const predict = xRow => refit.intercept + xRow.reduce((s, v, j) => s + v * refit.coeffs[j], 0);
  return {
    type: 'multi-lts', intercept: refit.intercept, coeffs: refit.coeffs,
    trimmedCount: h, totalCount: n, r2: computeR2Multi(X, y, predict), predict,
  };
}

// Multivariate exponential: y = a * e^(b1*x1 + b2*x2 + ...) — one constant
// `a` (the same role fitExponential's `a` plays), not one per variable.
// Linearizes exactly like the single-variable version: ln(y) = ln(a) +
// b1*x1 + b2*x2 + ..., which is just ordinary multivariate linear
// regression on ln(y) — reuses fitMultiLinear directly.
function fitMultiExponential(X, y, weights = null) {
  if (y.some(v => v <= 0)) throw new Error('Exponential fit needs every Y value to be greater than 0.');
  const w = weights || new Array(y.length).fill(1);
  const lineFit = fitMultiLinear(X, y.map(Math.log), w);
  const a = Math.exp(lineFit.intercept);
  const coeffs = lineFit.coeffs;
  const predict = xRow => a * Math.exp(xRow.reduce((s, v, j) => s + v * coeffs[j], 0));
  return { type: 'multi-exponential', a, coeffs, r2: computeR2MultiWeighted(X, y, predict, w), predict };
}

function formatMultiLinearEquation({ intercept, coeffs }, varNames) {
  const terms = coeffs.map((c, j) => `${c.toFixed(4)}*${varNames[j]}`);
  return `${intercept.toFixed(4)} + ${terms.join(' + ')}`;
}

// ---------------------------------------------------------------------------
// Weight penalization: ridge (L2), lasso (L1), and elastic net (a mix of
// both) for multi-variable regression. Unlike fitMultiLinear above, these
// don't just minimize squared error -- they minimize squared error PLUS a
// penalty on how large the coefficients get, which: (1) keeps a model with
// several correlated X columns from blowing up any one coefficient trying
// to split credit between them (ridge's specialty), and/or (2) can shrink
// a coefficient all the way to exactly 0, effectively dropping that
// variable (lasso's specialty -- useful as a form of automatic variable
// selection). Elastic net (alpha between 0 and 1) blends the two.
//
// Ridge has a closed-form solution (same normal-equations solve
// fitMultiLinear already uses, just with lambda added to the diagonal).
// Lasso/elastic net have no closed form because the L1 penalty isn't
// differentiable at 0, so those use coordinate descent -- the standard
// algorithm for this (same one glmnet/scikit-learn use under the hood):
// optimize one coefficient at a time, holding all the others fixed, via a
// "soft threshold" update, and repeat until nothing moves.

// Standardizes every X column (weighted mean 0, weighted variance 1) and
// centers y -- coordinate descent's soft-threshold step assumes every
// column is on a comparable scale, otherwise the penalty bites a
// large-raw-scale column harder than a small-raw-scale one for no
// principled reason. Shared by fitElasticNet and the auto-lambda search
// below so both standardize the exact same way.
function standardizeDesignMatrix(X, y, weights) {
  const n = X.length, k = X[0].length;
  const w = weights || new Array(n).fill(1);
  const wSum = w.reduce((s, v) => s + v, 0);
  const means = new Array(k), sds = new Array(k);
  const Xs = Array.from({ length: n }, () => new Array(k));
  for (let j = 0; j < k; j++) {
    let mean = 0; for (let i = 0; i < n; i++) mean += w[i] * X[i][j]; mean /= wSum;
    let variance = 0; for (let i = 0; i < n; i++) variance += w[i] * (X[i][j] - mean) ** 2; variance /= wSum;
    const sd = Math.sqrt(variance) || 1;
    means[j] = mean; sds[j] = sd;
    for (let i = 0; i < n; i++) Xs[i][j] = (X[i][j] - mean) / sd;
  }
  let yMean = 0; for (let i = 0; i < n; i++) yMean += w[i] * y[i]; yMean /= wSum;
  const ys = y.map(v => v - yMean);
  return { Xs, ys, means, sds, yMean, w, wSum, n, k };
}
function softThreshold(z, gamma) {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

// Elastic net: minimizes weighted squared error + lambda*alpha*sum(|b_j|)
// + lambda*(1-alpha)/2*sum(b_j^2). alpha=1 is pure lasso, alpha=0 is pure
// ridge (though fitRidge below uses ridge's own closed form instead, which
// is exact and doesn't need iterating), anything between is a genuine
// elastic-net blend. Coordinate descent tracks a running residual vector
// and updates it incrementally as each coefficient changes (the standard
// efficient implementation), rather than recomputing every prediction from
// scratch on every pass.
function fitElasticNet(X, y, { lambda = 1, alpha = 1, iterations = 1000, tol = 1e-8, weights = null } = {}) {
  const { Xs, ys, means, sds, yMean, w, k, n } = standardizeDesignMatrix(X, y, weights);
  const colSS = new Array(k).fill(0);
  for (let j = 0; j < k; j++) for (let i = 0; i < n; i++) colSS[j] += w[i] * Xs[i][j] * Xs[i][j];

  let beta = new Array(k).fill(0);
  let resid = ys.slice(); // ys - Xs*beta; starts at ys since beta starts at 0
  const l1 = lambda * alpha, l2 = lambda * (1 - alpha);

  for (let iter = 0; iter < iterations; iter++) {
    let maxChange = 0;
    for (let j = 0; j < k; j++) {
      let rho = 0;
      for (let i = 0; i < n; i++) rho += w[i] * Xs[i][j] * (resid[i] + beta[j] * Xs[i][j]);
      const z = colSS[j] + l2;
      const newBeta = z > 0 ? softThreshold(rho, l1) / z : 0;
      const delta = newBeta - beta[j];
      if (delta !== 0) { for (let i = 0; i < n; i++) resid[i] -= delta * Xs[i][j]; beta[j] = newBeta; }
      maxChange = Math.max(maxChange, Math.abs(delta));
    }
    if (maxChange < tol) break;
  }

  const coeffs = beta.map((b, j) => b / sds[j]);
  const intercept = yMean - coeffs.reduce((s, c, j) => s + c * means[j], 0);
  const predict = xRow => intercept + xRow.reduce((s, v, j) => s + v * coeffs[j], 0);
  let ssRes = 0; for (let i = 0; i < n; i++) ssRes += w[i] * (y[i] - predict(X[i])) ** 2;
  return { type: 'elastic-net', intercept, coeffs, lambda, alpha, nonzero: coeffs.filter(c => c !== 0).length, r2: computeR2MultiWeighted(X, y, predict, w), predict };
}
// Ridge alone: y = b0 + Xb, minimizing ||y-Xb||^2 + lambda*sum(b_j^2),
// intercept NOT penalized. Closed-form via the same normal-equations solve
// fitMultiLinear uses, just with lambda added to the diagonal for every
// coefficient except the intercept's own row/col (index 0) -- no iterating
// needed, unlike lasso/elastic net above.
function fitRidge(X, y, { lambda = 1, weights = null } = {}) {
  const n = X.length, k = X[0].length, terms = k + 1;
  const w = weights || new Array(n).fill(1);
  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const b = new Array(terms).fill(0);
  for (let i = 0; i < n; i++) {
    const row = [1, ...X[i]];
    for (let r = 0; r < terms; r++) { for (let c = 0; c < terms; c++) A[r][c] += w[i] * row[r] * row[c]; b[r] += w[i] * row[r] * y[i]; }
  }
  for (let j = 1; j < terms; j++) A[j][j] += lambda;
  const solved = solveLinearSystem(A, b);
  const intercept = solved[0], coeffs = solved.slice(1);
  const predict = xRow => intercept + xRow.reduce((s, v, j) => s + v * coeffs[j], 0);
  let ssRes = 0; for (let i = 0; i < n; i++) ssRes += w[i] * (y[i] - predict(X[i])) ** 2;
  return { type: 'ridge', intercept, coeffs, lambda, r2: computeR2MultiWeighted(X, y, predict, w), predict };
}

// lambda_max: the smallest lambda (at a given alpha) that drives every
// lasso-penalized coefficient to exactly 0 -- the standard glmnet
// convention for picking a sane SEARCH RANGE, since "large enough to zero
// everything out" depends entirely on the data's own scale and
// correlations, not a fixed number that would work the same on every
// dataset. On standardized columns, coordinate j first leaves 0 exactly
// when |rho_j| (at beta=0, rho_j = sum(w*Xs_j*ys)) exceeds lambda*alpha, so
// the max over j IS that threshold.
function lassoLambdaMax(X, y, alpha, weights = null) {
  const { Xs, ys, w, k, n } = standardizeDesignMatrix(X, y, weights);
  let maxRho = 0;
  for (let j = 0; j < k; j++) {
    let rho = 0; for (let i = 0; i < n; i++) rho += w[i] * Xs[i][j] * ys[i];
    maxRho = Math.max(maxRho, Math.abs(rho));
  }
  return alpha > 0 ? maxRho / alpha : maxRho || 1;
}

// Auto-selects lambda (and, if alpha isn't pinned, alpha too) via the same
// internal FIT/CHECK split + held-out-R² discipline as fitARIMAAuto and
// fitProphetStyleAuto -- consistent with every other "Auto" fit in this
// file. candidateAlphas defaults to lasso/elastic-net-mix/ridge so a single
// call can pick whichever of the three actually helps THIS dataset, rather
// than forcing the caller to already know which one to try.
const PENALTY_LAMBDA_FRACTIONS = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
function fitPenalizedAuto(X, y, { weights = null, candidateAlphas = [1, 0.5, 0] } = {}) {
  const n = X.length, k = X[0].length;
  if (n < k + 8) throw new Error(`Need more rows for penalized-regression auto lambda selection (have ${n}, ${k} X columns).`);
  const fitLen = Math.max(k + 4, Math.floor(n * 0.7));
  const fitX = X.slice(0, fitLen), fitY = y.slice(0, fitLen), fitW = weights ? weights.slice(0, fitLen) : null;
  const checkX = X.slice(fitLen), checkY = y.slice(fitLen), checkW = weights ? weights.slice(fitLen) : null;

  let best = null;
  if (checkX.length >= 3) {
    for (const alpha of candidateAlphas) {
      const lambdaMax = alpha > 0 ? lassoLambdaMax(fitX, fitY, alpha, fitW) : lassoLambdaMax(fitX, fitY, 1, fitW) * 10; // pure ridge has no lambda_max of its own -- borrow lasso's scale and search wider
      for (const frac of PENALTY_LAMBDA_FRACTIONS) {
        const lambda = lambdaMax * frac;
        let fit; try { fit = fitElasticNet(fitX, fitY, { lambda, alpha, weights: fitW }); } catch (e) { continue; }
        let ssRes = 0, ssTot = 0;
        const cw = checkW || new Array(checkX.length).fill(1);
        const checkMean = checkY.reduce((s, v, i) => s + cw[i] * v, 0) / cw.reduce((s, v) => s + v, 0);
        for (let i = 0; i < checkX.length; i++) { ssRes += cw[i] * (checkY[i] - fit.predict(checkX[i])) ** 2; ssTot += cw[i] * (checkY[i] - checkMean) ** 2; }
        const checkR2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
        if (!best || checkR2 > best.checkR2) best = { alpha, lambda, checkR2 };
      }
    }
  }
  if (!best) best = { alpha: 0.5, lambda: 0 }; // too little data to validate a choice -- fall back to plain OLS (lambda=0)
  const final = fitElasticNet(X, y, { lambda: best.lambda, alpha: best.alpha, weights });
  return { ...final, autoCheckR2: best.checkR2 };
}

// Regularized polynomial: fits a HIGHER-degree polynomial than
// runRegression's own discrete degree search normally risks (runRegression
// caps at maxDegree=3 by default specifically because an unpenalized
// degree-4/5 fit on a short series chases noise), then lets ridge's own
// penalty control how much of that extra flexibility actually gets used --
// auto-selected the same FIT/CHECK way fitPenalizedAuto already works,
// rather than a human picking a flat degree cutoff. This is the single-X
// candidate-pool's entry point into weight penalization: a different
// mechanism for controlling model complexity (shrinkage, not a discrete
// degree choice) than every other single-X candidate in the pool uses, so
// it's a genuinely new thing to test there, not just another curve shape.
// x is rescaled to roughly [-1,1] first (xMid/xHalfRange) purely for
// numerical conditioning -- raw x^5 on, say, a "year" column with x~2000
// would be astronomically large before standardizeDesignMatrix even gets a
// chance to rescale it, which can blow up floating-point precision well
// before the ridge penalty ever gets involved.
function fitRegularizedPolynomial(xs, ys, { maxDegree = 5, weights = null } = {}) {
  const n = xs.length;
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xMid = (xMin + xMax) / 2, xHalfRange = (xMax - xMin) / 2 || 1;
  const scaled = xs.map(x => (x - xMid) / xHalfRange);
  const X = scaled.map(x => Array.from({ length: maxDegree }, (_, p) => x ** (p + 1)));
  const fit = fitPenalizedAuto(X, ys, { weights, candidateAlphas: [0] }); // ridge only -- see comment above
  const predict = x => {
    const xs_ = (x - xMid) / xHalfRange;
    return fit.intercept + fit.coeffs.reduce((s, c, p) => s + c * xs_ ** (p + 1), 0);
  };
  let ssRes = 0, ssTot = 0;
  const w = weights || new Array(n).fill(1);
  const wSum = w.reduce((s, v) => s + v, 0);
  const yMean = ys.reduce((s, y, i) => s + w[i] * y, 0) / wSum;
  for (let i = 0; i < n; i++) { ssRes += w[i] * (ys[i] - predict(xs[i])) ** 2; ssTot += w[i] * (ys[i] - yMean) ** 2; }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  return { type: 'regularized-polynomial', maxDegree, lambda: fit.lambda, coeffs: fit.coeffs, intercept: fit.intercept, r2, predict };
}

function formatMultiExponentialEquation({ a, coeffs }, varNames) {
  const terms = coeffs.map((c, j) => `${c.toFixed(4)}*${varNames[j]}`);
  return `${a.toFixed(4)} * e^(${terms.join(' + ')})`;
}

// ---------------------------------------------------------------------------
// Standardizing features for IMPORTANCE comparison -- putting several X
// columns on the same scale so a multi-variable model's coefficients can be
// read as "which variable moves y the most", not just "which has the
// biggest raw units". A plain z-score ((x-mean)/sd) answers that correctly
// only when the feature's own spread is well-behaved; on a SKEWED feature, a
// handful of extreme values inflate the std, which squashes everyone else's
// z-score toward 0 and makes that variable look artificially UNimportant
// next to a better-behaved one with the same real effect size.
//
// Two methods:
//   'zscore' -- the usual z-score, but mean/sd are computed from the
//   PERCENTILE-TRIMMED core (5th-95th, same trim findOptimalHistogramBins
//   already uses elsewhere in this file) rather than the raw data, so a few
//   outliers don't single-handedly set everyone else's scale. Every point
//   still gets a z-score (nothing is dropped) -- only the mean/sd estimate
//   itself ignores the tails.
//   'normal-quantile' -- rank-based inverse-normal transform (aka RankIT /
//   Blom's formula, the same order-statistic formula shapiroWilkTest
//   already uses for its own reference distribution): convert each value to
//   its rank, map that rank to a quantile, then look up that quantile's
//   position on a standard normal curve. This is fully distribution-free --
//   it doesn't matter whether the original feature was skewed, bimodal, or
//   had a heavy tail, every transformed feature comes out with EXACTLY the
//   same (standard normal) shape, so coefficients built on top of them are
//   always comparing like with like. The tradeoff: it's a rank transform,
//   so it only preserves ORDER within a feature, not the original spacing
//   between values -- two points that were close together in the raw data
//   end up EXACTLY as far apart (in rank terms) as two points that were far
//   apart, as long as no other points sit between them.
function zScoreStandardize(values, { trimPercentile = 0.05 } = {}) {
  const n = values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(trimPercentile * n)];
  const hi = sorted[Math.min(n - 1, Math.ceil((1 - trimPercentile) * n) - 1)];
  const core = values.filter(v => v >= lo && v <= hi);
  const coreN = core.length || n;
  const corePool = core.length ? core : values;
  const mean = corePool.reduce((s, v) => s + v, 0) / coreN;
  const sd = Math.sqrt(corePool.reduce((s, v) => s + (v - mean) ** 2, 0) / coreN) || 1;
  return values.map(v => (v - mean) / sd);
}
function normalQuantileStandardize(values) {
  const n = values.length;
  // Average ranks for ties (the standard, textbook way to rank data with
  // repeated values) -- an untied rank would arbitrarily break ties by
  // original order, which would make the transform non-deterministic with
  // respect to which value "deserves" the higher quantile.
  const order = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    const avgRank = (i + j) / 2 + 1; // 1-indexed average rank across the tied block
    for (let k = i; k <= j; k++) ranks[order[k]] = avgRank;
    i = j + 1;
  }
  return ranks.map(r => studentTInverseNormalApprox((r - 0.375) / (n + 0.25)));
}
// Skewness (Fisher-Pearson, population formula -- consistent with this
// project's other skewness usage, e.g. skewNormalErrorFit's seed estimate).
function sampleSkewness(values) {
  const n = values.length;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  if (sd === 0) return 0;
  return values.reduce((s, v) => s + ((v - mean) / sd) ** 3, 0) / n;
}
// 'auto' picks the normal-quantile transform only when there's a REAL,
// detectable skew to protect against -- Shapiro-Wilk (reused as-is, same
// calibrated threshold the rest of this file already trusts) flags "this
// doesn't look like a symmetric, well-behaved distribution" ,and a plain
// skewness magnitude check confirms it's actually lopsided (SW alone would
// also fire on e.g. a heavy-tailed but SYMMETRIC distribution, where the
// trimmed z-score already handles the problem fine and a rank transform
// would just throw away real spacing information for no reason).
const SKEW_STANDARDIZE_THRESHOLD = 0.75;
function standardizeForImportance(values, { method = 'auto' } = {}) {
  let chosen = method;
  if (method === 'auto') {
    const sw = shapiroWilkTest(values);
    const skew = sampleSkewness(values);
    chosen = (sw && shapiroWilkLooksNonNull(sw.pValue) && Math.abs(skew) > SKEW_STANDARDIZE_THRESHOLD) ? 'normal-quantile' : 'zscore';
  }
  const standardized = chosen === 'normal-quantile' ? normalQuantileStandardize(values) : zScoreStandardize(values);
  return { method: chosen, values: standardized };
}

// Fits y = a * x^b — a constant-percentage relationship between X and Y
// (common for scaling laws). Linearized as ln(y) = ln(a) + b*ln(x); needs
// every X and Y to be positive.
function fitPower(xs, ys, weights = null) {
  if (xs.some(x => x <= 0)) throw new Error('Power fit needs every X value to be greater than 0.');
  if (ys.some(y => y <= 0)) throw new Error('Power fit needs every Y value to be greater than 0.');
  const w = weights || new Array(xs.length).fill(1);
  // Same reasoning as fitExponential's CI above: ln(y) = ln(a) + b*ln(x)
  // is a straight line in (ln x, ln y) space, so reuse that OLS fit's
  // exact CI instead of re-deriving one.
  const lineFit = fitPolynomial(xs.map(Math.log), ys.map(Math.log), 1, w);
  const a = Math.exp(lineFit.coeffs[0]);
  const b = lineFit.coeffs[1];
  const predict = x => a * Math.pow(x, b);
  const ci = lineFit.ci && {
    exact: true, df: lineFit.ci.df, tCritical: lineFit.ci.tCritical,
    params: {
      a: { lower: Math.exp(lineFit.ci.coeffs[0].lower), upper: Math.exp(lineFit.ci.coeffs[0].upper) },
      b: { lower: lineFit.ci.coeffs[1].lower, upper: lineFit.ci.coeffs[1].upper },
    },
  };
  return { type: 'power', a, b, r2: computeR2Weighted(xs, ys, predict, w), predict, ci };
}

// Fits y = a * ln(x) + b — fast initial change that levels off, without
// ever fully plateauing the way the S-curve does. Needs every X positive.
function fitLogarithmic(xs, ys, weights = null) {
  const w = weights || new Array(xs.length).fill(1);
  // ln(x) is undefined at x<=0, but a dataset whose X naturally starts at
  // or below 0 (e.g. "ply 0" in a game-tree dataset, or a year expressed
  // as an offset from some baseline) shouldn't be flatly rejected just
  // because of where its own index happens to start. Auto-shift X so
  // every value is positive instead: shift is fixed from this call's own
  // data (not recomputed per bootstrap resample, which would make the
  // transform itself wobble between resamples), chosen so the smallest
  // shifted X lands at exactly 1 (ln(1)=0, a clean, arbitrary-epsilon-free
  // anchor) rather than picking some tiny fudge factor.
  const minX = Math.min(...xs);
  const shift = minX > 0 ? 0 : 1 - minX;
  const shiftedXs = shift === 0 ? xs : xs.map(x => x + shift);
  // Plain OLS with ln(x+shift) as the input feature — unlike fitExponential/
  // fitPower, Y itself is never transformed, so lineFit's CI needs no
  // exp() back-transform at all, just relabeling (coeffs[0] is b here,
  // coeffs[1] is a — see the assignments right below).
  const lineFit = fitPolynomial(shiftedXs.map(Math.log), ys, 1, w);
  const b = lineFit.coeffs[0];
  const a = lineFit.coeffs[1];
  const predict = x => a * Math.log(x + shift) + b;
  const ci = lineFit.ci && {
    exact: true, df: lineFit.ci.df, tCritical: lineFit.ci.tCritical,
    params: {
      b: { lower: lineFit.ci.coeffs[0].lower, upper: lineFit.ci.coeffs[0].upper },
      a: { lower: lineFit.ci.coeffs[1].lower, upper: lineFit.ci.coeffs[1].upper },
    },
  };
  return { type: 'logarithmic', a, b, shift, r2: computeR2Weighted(xs, ys, predict, w), predict, ci };
}

function formatExponentialEquation({ a, b }) {
  return `${a.toFixed(4)} * e^(${b.toFixed(4)}*x)`;
}

// Fits y = a*e^(b*u) + c via Adam (adaptive per-parameter step sizes), for
// whatever `u` the caller supplies — u=x gives the exponential-with-offset
// fit below; u=ln(x) gives the power-law-with-offset fit (a*x^b+c) further
// down, since a*e^(b*ln(x)) = a*x^b. Shared because it's the same nonlinear
// shape and the same hard-to-optimize surface either way: plain gradient
// descent needed 50,000+ iterations to even approach the right answer on a
// test decay curve, because a, b, and c pull the loss at very different
// scales — Adam's per-parameter scaling gets the same case right in 4,000.
function fitAExpBUPlusC(us, ys, { iters = 4000, lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
  const n = us.length;
  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const uMean = mean(us);
  const uStd = Math.sqrt(mean(us.map(u => (u - uMean) ** 2))) || 1;
  const un = us.map(u => (u - uMean) / uStd);

  const yScale = arrayMax(ys.map(Math.abs)) || 1;
  const yn = ys.map(y => y / yScale);
  const yMeanN = mean(yn);

  // Clamps the exponent before exponentiating — without this, a bad step
  // early in training can send b*un past ~700 and overflow to Infinity,
  // which then propagates to NaN and never recovers.
  const safeExp = z => Math.exp(Math.max(-50, Math.min(50, z)));

  // Whether the curve rises or falls depends on the sign of a AND b
  // together (a<0 with b<0 still rises, just like a>0 with b>0 does) — so a
  // single "guess the direction from correlation" start can lock onto the
  // wrong region and never recover (verified: it did, on a decaying curve
  // with a<0). Trying all 4 sign combinations and keeping whichever
  // converges best sidesteps that instead of trying to out-guess it.
  let best = null;
  for (const a0 of [1, -1]) {
    for (const b0 of [1, -1]) {
      let a = a0, b = b0, c = yMeanN;
      let ma = 0, mb = 0, mc = 0, va = 0, vb = 0, vc = 0;
      let diverged = false;

      for (let it = 1; it <= iters; it++) {
        let ga = 0, gb = 0, gc = 0;
        for (let i = 0; i < n; i++) {
          const e = safeExp(b * un[i]);
          const err = (a * e + c) - yn[i];
          ga += err * e;
          gb += err * a * e * un[i];
          gc += err;
        }
        ga = 2 * ga / n; gb = 2 * gb / n; gc = 2 * gc / n;

        ma = beta1 * ma + (1 - beta1) * ga; va = beta2 * va + (1 - beta2) * ga * ga;
        mb = beta1 * mb + (1 - beta1) * gb; vb = beta2 * vb + (1 - beta2) * gb * gb;
        mc = beta1 * mc + (1 - beta1) * gc; vc = beta2 * vc + (1 - beta2) * gc * gc;

        const bc1 = 1 - beta1 ** it, bc2 = 1 - beta2 ** it;
        a -= lr * (ma / bc1) / (Math.sqrt(va / bc2) + eps);
        b -= lr * (mb / bc1) / (Math.sqrt(vb / bc2) + eps);
        c -= lr * (mc / bc1) / (Math.sqrt(vc / bc2) + eps);
        if (![a, b, c].every(Number.isFinite)) { diverged = true; break; }
      }
      if (diverged) continue;

      let ssRes = 0, ssTot = 0;
      for (let i = 0; i < n; i++) {
        ssRes += (yn[i] - (a * safeExp(b * un[i]) + c)) ** 2;
        ssTot += (yn[i] - yMeanN) ** 2;
      }
      const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
      if (!best || r2 > best.r2) best = { a, b, c, r2 };
    }
  }
  if (!best) return null;

  // Undo the u normalization: b*un = b*(u-uMean)/uStd = (b/uStd)*u - b*uMean/uStd
  // e^(that) = e^(-b*uMean/uStd) * e^((b/uStd)*u) — the constant factor just
  // folds into a. Undo the y scaling by multiplying a and c back out.
  const finalB = best.b / uStd;
  const finalA = best.a * Math.exp(-(best.b * uMean) / uStd) * yScale;
  const finalC = best.c * yScale;
  return { a: finalA, b: finalB, c: finalC };
}

// Fits y = a*e^(b*x) + c — like the plain exponential fit above, but lets
// the curve level off at any value c instead of being forced toward 0.
// (Desmos's exp(ax+b)+c is the same shape: exp(ax+b) = e^b * e^(ax), so
// that b just becomes this a's scale — same 3 free numbers either way.)
function fitExponentialOffset(xs, ys, opts) {
  if (xs.length < 4) throw new Error('Need at least 4 data points to fit an exponential-with-offset curve.');
  const fit = fitAExpBUPlusC(xs, ys, opts);
  if (!fit) throw new Error('The exponential-with-offset fit did not converge for this data.');
  const predict = x => fit.a * Math.exp(fit.b * x) + fit.c;
  return { type: 'exponential-offset', a: fit.a, b: fit.b, c: fit.c, r2: computeR2(xs, ys, predict), predict };
}

function formatExponentialOffsetEquation({ a, b, c }) {
  const sign = c >= 0 ? '+' : '-';
  return `${a.toFixed(4)} * e^(${b.toFixed(4)}*x) ${sign} ${Math.abs(c).toFixed(4)}`;
}
function formatPowerEquation({ a, b }) {
  return `${a.toFixed(4)} * x^${b.toFixed(4)}`;
}

// Fits y = a*x^b + c — the plain power fit, but able to level off at any
// value c instead of always decaying to 0. Since a*e^(b*ln(x)) = a*x^b,
// this is the exact same nonlinear shape as the exponential-with-offset fit
// above with u=ln(x) substituted for x — so it reuses that same Adam fit.
// Needs every X positive (ln(x) is undefined otherwise).
function fitPowerOffset(xs, ys, opts) {
  if (xs.some(x => x <= 0)) throw new Error('Power-with-offset fit needs every X value to be greater than 0.');
  if (xs.length < 4) throw new Error('Need at least 4 data points to fit a power-with-offset curve.');
  const fit = fitAExpBUPlusC(xs.map(Math.log), ys, opts);
  if (!fit) throw new Error('The power-with-offset fit did not converge for this data.');
  const predict = x => fit.a * Math.pow(x, fit.b) + fit.c;
  return { type: 'power-offset', a: fit.a, b: fit.b, c: fit.c, r2: computeR2(xs, ys, predict), predict };
}

function formatPowerOffsetEquation({ a, b, c }) {
  const sign = c >= 0 ? '+' : '-';
  return `${a.toFixed(4)} * x^${b.toFixed(4)} ${sign} ${Math.abs(c).toFixed(4)}`;
}
function formatLogarithmicEquation({ a, b, shift }) {
  const sign = b >= 0 ? '+' : '-';
  const xTerm = shift ? `(x + ${shift.toFixed(4)})` : '(x)';
  const shiftNote = shift ? `  (X auto-shifted by +${shift.toFixed(4)} so every value is > 0 before taking ln)` : '';
  return `${a.toFixed(4)} * ln${xTerm} ${sign} ${Math.abs(b).toFixed(4)}${shiftNote}`;
}

// Fits y = a/x + b — approaches a flat asymptote (b) as x grows, unlike the
// power fit which decays all the way to 0. Linearized as a straight line
// against the feature 1/x; needs every X to be nonzero.
function fitReciprocal(xs, ys) {
  if (xs.some(x => x === 0)) throw new Error('Reciprocal fit needs every X value to be nonzero.');
  const lineFit = fitPolynomial(xs.map(x => 1 / x), ys, 1);
  const b = lineFit.coeffs[0];
  const a = lineFit.coeffs[1];
  const predict = x => a / x + b;
  return { type: 'reciprocal', a, b, r2: computeR2(xs, ys, predict), predict };
}

function formatReciprocalEquation({ a, b }) {
  const sign = b >= 0 ? '+' : '-';
  return `${a.toFixed(4)} / x ${sign} ${Math.abs(b).toFixed(4)}`;
}

// Fits y = a*sin(b*x + c) + d — a repeating wave: a = height above/below the
// midline, b = how fast it cycles, c = how far it's shifted sideways, d =
// the midline itself. Frequency (b) is the hard part to find by gradient
// descent alone — a bad starting guess just settles into the nearest wrong
// cycle count — so this tries several starting frequencies (covering half a
// cycle up to 4 full cycles across the data) and keeps whichever converges
// to the best fit.
function fitSinusoidal(xs, ys, { iters = 5000, lr = 0.05 } = {}) {
  const n = xs.length;
  if (n < 4) throw new Error('Need at least 4 data points to fit a sinusoidal curve.');

  const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const xMean = mean(xs);
  const xStd = Math.sqrt(mean(xs.map(x => (x - xMean) ** 2))) || 1;
  const xn = xs.map(x => (x - xMean) / xStd);
  const yMean = mean(ys);
  const yAmpGuess = Math.sqrt(mean(ys.map(y => (y - yMean) ** 2))) * Math.SQRT2 || 1;
  const xnSpan = arrayMax(xn) - arrayMin(xn) || 1;

  let best = null;
  for (const cycles of [0.5, 1, 1.5, 2, 3, 4]) {
    let a = yAmpGuess, b = (2 * Math.PI * cycles) / xnSpan, c = 0, d = yMean;
    let diverged = false;

    for (let it = 0; it < iters; it++) {
      let ga = 0, gb = 0, gc = 0, gd = 0;
      for (let i = 0; i < n; i++) {
        const z = b * xn[i] + c;
        const s = Math.sin(z);
        const err = a * s + d - ys[i];
        ga += err * s;
        gb += err * a * Math.cos(z) * xn[i];
        gc += err * a * Math.cos(z);
        gd += err;
      }
      a -= lr * (2 * ga / n);
      b -= lr * (2 * gb / n);
      c -= lr * (2 * gc / n);
      d -= lr * (2 * gd / n);
      if (![a, b, c, d].every(Number.isFinite)) { diverged = true; break; }
    }
    if (diverged) continue;

    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < n; i++) {
      ssRes += (ys[i] - (a * Math.sin(b * xn[i] + c) + d)) ** 2;
      ssTot += (ys[i] - yMean) ** 2;
    }
    const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
    if (!best || r2 > best.r2) best = { a, b, c, d, r2 };
  }

  if (!best) throw new Error('The sinusoidal fit did not converge for this data.');

  // Undo the x normalization: z = b*xn + c = b*(x-xMean)/xStd + c
  //                              = (b/xStd)*x + (c - b*xMean/xStd)
  const finalB = best.b / xStd;
  const finalC = best.c - (best.b * xMean) / xStd;
  const predict = x => best.a * Math.sin(finalB * x + finalC) + best.d;

  return { type: 'sinusoidal', a: best.a, b: finalB, c: finalC, d: best.d, r2: computeR2(xs, ys, predict), predict };
}

function formatSinusoidalEquation({ a, b, c, d }) {
  const cSign = c >= 0 ? '+' : '-';
  const dSign = d >= 0 ? '+' : '-';
  return `${a.toFixed(4)} * sin(${b.toFixed(4)}*x ${cSign} ${Math.abs(c).toFixed(4)}) ${dSign} ${Math.abs(d).toFixed(4)}`;
}

// The three fits below are all still straight lines (y = slope*x + intercept)
// — what's different is how they handle outliers. Ordinary least squares
// (the plain "linear" fit) lets a single wild point drag the whole line
// toward it, since squaring the error makes big residuals count enormously.
// These three each limit that in a different way.

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Median absolute deviation, scaled by 1.4826 so it estimates the same
// thing a standard deviation would for normally-distributed data — the
// standard way to get a robust "typical spread" that a few outliers can't
// blow up (unlike a plain standard deviation, which they easily can).
function robustSigma(values) {
  const m = median(values);
  const mad = median(values.map(v => Math.abs(v - m)));
  return 1.4826 * mad;
}

// Turns continuous weights into an n-point multiset by replicating each
// point roughly weight-proportionally (largest-remainder rounding keeps
// the total exactly n) — e.g. a point with half the average weight of
// another ends up with roughly half as many copies. This is what lets
// genericWeightedRefit below do a Tukey-biweight IRLS refit against ANY
// fitFn(xs, ys), including every resistant/robust method in this file
// (RANSAC, Theil-Sen, Huber, ...), none of which take a weights argument —
// only fitPolynomial does. Deterministic (no new randomness introduced),
// so results stay reproducible under the same seed.
function weightedReplicate(xs, ys, weights) {
  const n = xs.length;
  const wsum = weights.reduce((s, w) => s + w, 0);
  if (wsum <= 0) return null;
  const raw = weights.map(w => (w / wsum) * n);
  const base = raw.map(Math.floor);
  let used = base.reduce((s, v) => s + v, 0);
  const remainders = raw.map((v, i) => ({ i, frac: v - base[i] })).sort((a, b) => b.frac - a.frac);
  for (let k = 0; used < n; k++) { base[remainders[k % n].i]++; used++; }
  const rxs = [], rys = [];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < base[i]; c++) { rxs.push(xs[i]); rys.push(ys[i]); }
  }
  return { rxs, rys };
}

// Refits ANY fitFn(xs, ys) against Tukey-biweight weights derived from its
// OWN residuals, iterating a few times (via weightedReplicate, since fitFn
// can't take a weights argument directly) — the same redescending M-
// estimator idea as fitTukeyBiweight/fitByIRLS above, just made to work
// with an arbitrary model instead of only a line. Exists specifically to
// counter skewed-outlier contamination inside movingBlockBootstrapParameterCI
// below: an earlier attempt there did one residual-threshold trim-and-
// refit pass and made the intercept CI WORSE (confirmed: wider, and the
// center moved further from the truth) because trimming off a fit that's
// already dragged off course by contamination misreads which points are
// the real outliers. Iterating lets a resample recover even when its first
// pass is thrown off. Validated against synthetic ground truth (ordinary
// least squares, n=60, a contiguous contaminated run of skewed/one-sided
// outliers, varying how much of n they cover): at a realistic 5% share,
// average CI width dropped from ~21 to ~1.9 while coverage of the true
// intercept improved (73/80 -> 76/80 trials); at 13% both coverage and
// width improved slightly; identical results on clean (uncontaminated)
// data in every trial, so this costs nothing when there's no contamination
// to correct for. Only degrades (slightly wider, coverage unaffected) once
// contamination reaches ~25% of the sample — past where any resampling
// method can be expected to recover the true line.
function genericWeightedRefit(fitFn, xs, ys, { iterations = 5, tuningConstant = 4.685, convergeTol = 1e-9 } = {}) {
  let fit;
  try { fit = fitFn(xs, ys); } catch (e) { return null; }
  if (typeof fit.predict !== 'function') return fit; // nothing to compute residuals from
  // Convergence probe: a handful of representative predict() outputs,
  // compared iteration to iteration — works for any fit shape (line,
  // polynomial, logistic, ...) without needing to know its parameters by
  // name. On clean (uncontaminated) data the weights are already ~1
  // everywhere, so this exits after one extra fit instead of looping the
  // full `iterations` every time — this function runs inside a
  // 150-resample bootstrap loop, so that difference matters.
  const probeX = [xs[0], xs[Math.floor(xs.length / 2)], xs[xs.length - 1]];
  let prevProbe = probeX.map(x => fit.predict(x));
  // Baseline spread from the UNWEIGHTED fit — the collapse guard below
  // measures each candidate refit against this, not against its own
  // runaway-shrunk residuals.
  const initialResiduals = xs.map((x, i) => ys[i] - fit.predict(x));
  const sigma0 = robustSigma(initialResiduals) || 1e-9;
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    const sigma = robustSigma(residuals) || 1e-9;
    const weights = residuals.map(r => {
      const u = r / (tuningConstant * sigma);
      return Math.abs(u) <= 1 ? (1 - u * u) ** 2 : 0;
    });
    const replicated = weightedReplicate(xs, ys, weights);
    if (!replicated) break; // weights collapsed to zero — keep the last good fit
    let next;
    try { next = fitFn(replicated.rxs, replicated.rys); } catch (e) { break; }
    if (typeof next.predict !== 'function') { fit = next; break; }
    // Collapse guard: on a handful of real datasets (confirmed via
    // leave-fold-out cross-validation against this project's own data
    // corpus — see data.txt's "Cash and Domestic Stock..." and "Movie
    // Gross" groups, both degree-3 polynomial fits), this loop can run
    // away instead of converging to a robust answer: each pass
    // concentrates weight onto a shrinking, increasingly duplicate-heavy
    // point cloud, whose own residuals vanish, which sharpens the next
    // round's weights further still, until it locks onto an exactly-zero
    // "fit" that reports a perfect (but meaningless) r2=1 and looks
    // converged (it reproduces itself forever). Checking the CANDIDATE
    // fit's residual spread against where the UNWEIGHTED fit started
    // catches this: a real robust fit narrows the spread somewhat, a
    // collapsed one obliterates it. Reject and stop rather than accept a
    // fit that's "improved" by orders of magnitude more than any genuine
    // outlier removal could explain.
    const candidateResiduals = xs.map((x, i) => ys[i] - next.predict(x));
    const candidateSigma = robustSigma(candidateResiduals);
    if (candidateSigma < sigma0 * 1e-6) break;
    fit = next;
    const nextProbe = probeX.map(x => fit.predict(x));
    const maxDelta = Math.max(...nextProbe.map((v, idx) => Math.abs(v - prevProbe[idx])));
    prevProbe = nextProbe;
    if (maxDelta < convergeTol) break;
  }
  return fit;
}

// Theil-Sen: the slope is the MEDIAN of the slopes between every pair of
// points. A handful of outlier points can only contribute a handful of
// bad pairwise slopes — the median shrugs them off as long as most of the
// data agrees on the trend. For large datasets, checking every pair
// (n²/2 of them) gets expensive, so this samples a large but bounded number
// of random pairs instead once n is big enough for that to matter.
function fitTheilSen(xs, ys, { maxPairs = 200000, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a Theil-Sen fit.');

  const totalPairs = (n * (n - 1)) / 2;
  const slopes = [];

  const addSlope = (i, j) => {
    if (xs[i] === xs[j]) return;
    slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
  };

  if (totalPairs <= maxPairs) {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) addSlope(i, j);
  } else {
    const random = createSeededRandom(seed);
    for (let k = 0; k < maxPairs; k++) {
      const i = Math.floor(random() * n);
      let j = Math.floor(random() * n);
      if (j === i) j = (j + 1) % n;
      addSlope(i, j);
    }
  }

  if (!slopes.length) throw new Error('Every X value is identical — cannot fit a line.');

  const slope = median(slopes);
  const intercept = median(ys.map((y, i) => y - slope * xs[i]));
  const predict = x => slope * x + intercept;

  return { type: 'theil-sen', slope, intercept, r2: computeR2(xs, ys, predict), predict };
}

// RANSAC: repeatedly picks 2 random points, draws the line through them,
// and counts how many OTHER points fall close to it (within a threshold
// set from the data's robust spread). Whichever random line collects the
// most "inliers" wins, and the final line is an ordinary least-squares fit
// through just that inlier set — so points that never inlier for any
// candidate line (the true outliers) never influence the answer at all.
// thresholdMultiplier was left at the generic textbook default of 2 (a
// conventional "2 robust standard deviations" inlier band) when RANSAC was
// first added, never swept against this corpus. Swept 0.5-10 here: 1.5 is
// the peak on both the full 56-dataset corpus (median test R² tied at
// 0.9259 everywhere; average peaks at 0.4566, vs. 2's 0.4555) and,
// independently, on a 7-dataset subset picked specifically because RANSAC
// and Least Trimmed Squares were each closest to winning there (median
// 0.7773 / average 0.6790 at 1.5, vs. 0.7771 / 0.6586 at 2). Values above
// ~3 make the inlier band so wide RANSAC just converges to plain OLS and
// stops winning anything at all (0 wins from 3 up through 10 tested).
function fitRANSAC(xs, ys, { iterations = 300, thresholdMultiplier = 1.5, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a RANSAC fit.');

  // A rough initial line just to measure a typical residual size from.
  const rough = fitPolynomial(xs, ys, 1);
  const roughResiduals = ys.map((y, i) => y - rough.predict(xs[i]));
  const sigma = robustSigma(roughResiduals) || (arrayMax(ys) - arrayMin(ys)) * 0.01 || 1;
  const threshold = thresholdMultiplier * sigma;

  let bestInliers = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * n);
    if (j === i) j = (j + 1) % n;
    if (xs[i] === xs[j]) continue;

    const slope = (ys[j] - ys[i]) / (xs[j] - xs[i]);
    const intercept = ys[i] - slope * xs[i];

    const inliers = [];
    for (let k = 0; k < n; k++) {
      if (Math.abs(ys[k] - (slope * xs[k] + intercept)) <= threshold) inliers.push(k);
    }
    if (!bestInliers || inliers.length > bestInliers.length) bestInliers = inliers;
  }

  if (!bestInliers || bestInliers.length < 2) throw new Error('RANSAC could not find a consistent line through this data.');

  const inlierXs = bestInliers.map(i => xs[i]);
  const inlierYs = bestInliers.map(i => ys[i]);
  const finalLine = fitPolynomial(inlierXs, inlierYs, 1);
  const [intercept, slope] = finalLine.coeffs;
  const predict = x => slope * x + intercept;

  return { type: 'ransac', slope, intercept, inlierCount: bestInliers.length, r2: computeR2(xs, ys, predict), predict };
}

// Same elemental random-pair search as fitRANSAC, but instead of an
// MAD-converted-to-SD threshold deciding how many points count as inliers
// (an adaptive count that can vary candidate to candidate), always keeps
// a FIXED trimFraction of the data — the same "keep 90%" idea
// fitLeastTrimmedSquares uses, just reusing RANSAC's own elemental-pair
// search and defaults rather than LTS's. Structurally, once the kept
// count is fixed instead of threshold-adaptive, scoring by inlier COUNT
// no longer differentiates candidates (every candidate keeps exactly the
// same number) — so candidates are scored by trimmed sum-of-squares
// instead, same as LTS.
//
// Tried in the Auto-selection pool (CANDIDATES in model_candidates.js)
// after checking whether this threshold choice specifically — not RANSAC
// vs. LMedS vs. LTS as whole algorithms — mattered on its own: offering
// both this and the real MAD-based RANSAC as separate candidates and
// letting the normal selection machinery pick per-dataset (via the check
// window, same as any other candidate pair) found this variant winning on
// 3 of 56 real datasets. One of those is a large, clear win: "Bachelor
// degree men salary" goes from test R² -2.91 (quadratic, the dataset's
// previous best fix via the noise-adaptive derivative gate) to +0.08. In
// isolation (against the then-14-candidate pool) this beat the pool
// without it in 247/300 bootstrap resamples (82%). But a full
// bootstrap-backed backward elimination pass over the whole pool (see the
// comment above CANDIDATES in model_candidates.js) found its removal
// win-rate climbs to ~60-65% once reciprocal/constant/proportional/LMedS
// are ALSO removed — once those weaker candidates are gone, whatever this
// was adding becomes redundant with plain RANSAC. Left out of CANDIDATES
// for that reason (per-candidate removal rankings don't compose
// additively — the same lesson this file's own comments have drawn
// before); kept here as a library function in case a future pool
// composition makes it useful again. A parallel fixed-90% variant of
// LMedS was tested the same way and never won a single dataset even when
// offered — not included anywhere, pure dead weight from the start.
function fitRANSACFixedFraction(xs, ys, { iterations = 300, trimFraction = 0.90, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a RANSAC fit.');
  const h = Math.max(2, Math.round(n * trimFraction));

  let best = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * n);
    if (j === i) j = (j + 1) % n;
    if (xs[i] === xs[j]) continue;

    const slope = (ys[j] - ys[i]) / (xs[j] - xs[i]);
    const intercept = ys[i] - slope * xs[i];
    const sqResiduals = xs.map((x, k) => (ys[k] - (slope * x + intercept)) ** 2);
    const trimmedSum = [...sqResiduals].sort((a, b) => a - b).slice(0, h).reduce((s, v) => s + v, 0);
    if (!best || trimmedSum < best.trimmedSum) best = { slope, intercept, trimmedSum };
  }
  if (!best) throw new Error('RANSAC could not find a consistent line through this data.');

  const absResiduals = xs.map((x, k) => Math.abs(ys[k] - (best.slope * x + best.intercept)));
  const inlierIdx = absResiduals.map((r, k) => k).sort((a, b) => absResiduals[a] - absResiduals[b]).slice(0, h);
  const inlierXs = inlierIdx.map(k => xs[k]), inlierYs = inlierIdx.map(k => ys[k]);
  const finalLine = fitPolynomial(inlierXs, inlierYs, 1);
  const [intercept, slope] = finalLine.coeffs;
  const predict = x => slope * x + intercept;

  return { type: 'ransac-fixed90', slope, intercept, inlierCount: inlierIdx.length, r2: computeR2(xs, ys, predict), predict };
}

// One weighted-least-squares step through (xs, ys, weights) — the shared
// core of every iteratively-reweighted fit below (Huber, Tukey, Andrews).
// Returns null if the weights collapse to nothing usable (e.g. every point
// got zero weight), so the caller can just keep its last good estimate.
function weightedLineStep(xs, ys, weights) {
  const n = xs.length;
  let sw = 0, swx = 0, swy = 0;
  for (let i = 0; i < n; i++) { sw += weights[i]; swx += weights[i] * xs[i]; swy += weights[i] * ys[i]; }
  if (sw === 0) return null;
  const xBar = swx / sw, yBar = swy / sw;

  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += weights[i] * (xs[i] - xBar) * (ys[i] - yBar);
    den += weights[i] * (xs[i] - xBar) ** 2;
  }
  if (den === 0) return null;
  const slope = num / den;
  return { slope, intercept: yBar - slope * xBar };
}

// Runs iteratively reweighted least squares to convergence: fit a line,
// recompute weights from the residuals using `weightFn`, refit, repeat.
// `weightFn(residual, sigma)` returns how much a point counts this round —
// sigma (the data's robust spread, re-estimated each round) is what lets
// the same weight formula adapt to any dataset's scale.
function fitByIRLS(xs, ys, weightFn, { iterations = 30 } = {}) {
  const n = xs.length;
  let { coeffs } = fitPolynomial(xs, ys, 1);
  let [intercept, slope] = coeffs;

  for (let it = 0; it < iterations; it++) {
    const residuals = ys.map((y, i) => y - (slope * xs[i] + intercept));
    const sigma = robustSigma(residuals) || 1e-9;
    const weights = residuals.map(r => weightFn(r, sigma));

    const step = weightedLineStep(xs, ys, weights);
    if (!step) break; // weights collapsed — keep the last good estimate

    const converged = Math.abs(step.slope - slope) < 1e-10 && Math.abs(step.intercept - intercept) < 1e-10;
    slope = step.slope; intercept = step.intercept;
    if (converged) break;
  }

  return { slope, intercept };
}

// Huber regression: points with a small residual get full weight (behaving
// just like ordinary least squares), points with a large residual get
// down-weighted in proportion to how far out they are — but never all the
// way to zero, so a point always has at least *some* pull on the line.
// `delta` (how big a residual counts as "large") is re-estimated from the
// data's own robust spread each round.
function fitHuber(xs, ys, { iterations = 30, deltaMultiplier = 1.345 } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a Huber fit.');

  const { slope, intercept } = fitByIRLS(xs, ys, (r, sigma) => {
    const delta = deltaMultiplier * sigma;
    return Math.abs(r) <= delta ? 1 : delta / Math.abs(r);
  }, { iterations });

  const predict = x => slope * x + intercept;
  return { type: 'huber', slope, intercept, r2: computeR2(xs, ys, predict), predict };
}

// Tukey's biweight (bisquare): like Huber, but "redescending" — past its
// cutoff, a point's weight doesn't just shrink, it hits exactly zero. A
// gross outlier ends up with zero pull on the line at all, rather than
// Huber's ever-smaller-but-still-nonzero pull. tuningConstant=4.685 is the
// standard choice (95% efficiency under normal errors).
function fitTukeyBiweight(xs, ys, { iterations = 30, tuningConstant = 4.685 } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a Tukey biweight fit.');

  const { slope, intercept } = fitByIRLS(xs, ys, (r, sigma) => {
    const u = r / (tuningConstant * sigma);
    return Math.abs(u) <= 1 ? (1 - u * u) ** 2 : 0;
  }, { iterations });

  const predict = x => slope * x + intercept;
  return { type: 'tukey-biweight', slope, intercept, r2: computeR2(xs, ys, predict), predict };
}

// Andrews' sine: another redescending weight function (zero past its
// cutoff, like Tukey's), just shaped differently in between — sin(u)/u
// instead of a squared parabola. tuningConstant=1.339 is the standard
// choice (95% efficiency under normal errors).
function fitAndrewsSine(xs, ys, { iterations = 30, tuningConstant = 1.339 } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for an Andrews sine fit.');

  const { slope, intercept } = fitByIRLS(xs, ys, (r, sigma) => {
    const u = r / (tuningConstant * sigma);
    if (Math.abs(u) >= Math.PI) return 0;
    return u === 0 ? 1 : Math.sin(u) / u;
  }, { iterations });

  const predict = x => slope * x + intercept;
  return { type: 'andrews-sine', slope, intercept, r2: computeR2(xs, ys, predict), predict };
}

// Hampel's M-estimator ("ABC" three-part redescending estimator): like
// Huber up to its first cutoff (a), full weight for small residuals, then
// a Huber-style 1/|u| taper between a and b, then a second linear taper
// down to exactly zero between b and c (like Tukey/Andrews' hard cutoff,
// but reached gradually instead of all at once past a single threshold).
// Standard tuning constants (a=1.7, b=3.4, c=8.5, in robust-sigma units)
// are the textbook default (Hampel et al., "Robust Statistics").
function fitHampel(xs, ys, { iterations = 30, a = 1.7, b = 3.4, c = 8.5 } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a Hampel fit.');

  const { slope, intercept } = fitByIRLS(xs, ys, (r, sigma) => {
    const u = Math.abs(r / sigma);
    if (u <= a) return 1;
    if (u <= b) return a / u;
    if (u <= c) return a * (c - u) / (u * (c - b));
    return 0;
  }, { iterations });

  const predict = x => slope * x + intercept;
  return { type: 'hampel', slope, intercept, r2: computeR2(xs, ys, predict), predict };
}

function formatLinearEquation({ slope, intercept }) {
  const sign = intercept >= 0 ? '+' : '-';
  return `${slope.toFixed(4)}*x ${sign} ${Math.abs(intercept).toFixed(4)}`;
}

// Least Median of Squares: like RANSAC, tries many random 2-point candidate
// lines — but instead of needing a distance threshold to decide who counts
// as an inlier, it just keeps whichever candidate has the smallest MEDIAN
// squared residual across all the points. As long as under half the data is
// contaminated, the true line's candidates will always have a small median
// error and a bad line's won't — no threshold to guess. After picking the
// winner, this refines it: estimates a robust noise scale from that winning
// median, flags points within 2.5x of it as inliers, and refits ordinary
// least squares through just those — the standard cleanup step used by
// R's lqs()/lmsreg (a raw random 2-point line is noisier than it needs to be).
function fitLeastMedianSquares(xs, ys, { iterations = 500, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a least-median-of-squares fit.');

  let best = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * n);
    if (j === i) j = (j + 1) % n;
    if (xs[i] === xs[j]) continue;

    const slope = (ys[j] - ys[i]) / (xs[j] - xs[i]);
    const intercept = ys[i] - slope * xs[i];
    const medSq = median(xs.map((x, k) => (ys[k] - (slope * x + intercept)) ** 2));
    if (!best || medSq < best.medSq) best = { slope, intercept, medSq };
  }
  if (!best) throw new Error('Could not find a valid line through this data.');

  const scale = 1.4826 * (1 + 5 / (n - 2)) * Math.sqrt(best.medSq);
  const residuals = xs.map((x, k) => ys[k] - (best.slope * x + best.intercept));
  const inlierIdx = residuals.map((r, k) => k).filter(k => Math.abs(residuals[k]) <= (scale || 1e-9) * 2.5);

  let { slope, intercept } = best;
  if (inlierIdx.length >= 2) {
    const refit = fitPolynomial(inlierIdx.map(k => xs[k]), inlierIdx.map(k => ys[k]), 1);
    [intercept, slope] = refit.coeffs;
  }

  const predict = x => slope * x + intercept;
  return { type: 'lms', slope, intercept, inlierCount: inlierIdx.length, r2: computeR2(xs, ys, predict), predict };
}

// Least Trimmed Squares: same random 2-point sampling as LMedS, but scores
// each candidate by the SUM of its smallest `trimFraction` share of squared
// residuals (75% by default) rather than the median of all of them. The
// worst-fitting quarter of points never get a say at all, no matter how bad
// their residual is — which is what makes this resistant to outliers, and
// also what makes it more statistically efficient than LMedS on clean data
// (it isn't throwing away information from every point, just the ones that
// look like outliers). The final line is an ordinary least-squares refit
// through exactly the trimmed set the winning candidate identified.
// trimFraction is really a KEEP fraction (points kept, not points trimmed —
// the name is inherited from the original fit's `h` variable, which is a
// count of points kept). Swept 85%-95% kept against this file's data.txt
// corpus (forcing every dataset through LTS only, no model selection):
// 90% scored best (median held-out test R² -0.8918, tied with 91% and
// 94%), clearly beating the old 75% default (-1.2364). Picked 90% as the
// cleanest of the tied-best values — trims the fewest points while still
// hitting the best score, i.e. ~10% of a typical dataset is being treated
// as an outlier.
function fitLeastTrimmedSquares(xs, ys, { iterations = 500, trimFraction = 0.90, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a least-trimmed-squares fit.');
  const h = Math.max(2, Math.round(n * trimFraction));

  let best = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * n);
    if (j === i) j = (j + 1) % n;
    if (xs[i] === xs[j]) continue;

    const slope = (ys[j] - ys[i]) / (xs[j] - xs[i]);
    const intercept = ys[i] - slope * xs[i];
    const sqResiduals = xs.map((x, k) => (ys[k] - (slope * x + intercept)) ** 2);
    const trimmedSum = [...sqResiduals].sort((a, b) => a - b).slice(0, h).reduce((s, v) => s + v, 0);
    if (!best || trimmedSum < best.trimmedSum) best = { slope, intercept, trimmedSum };
  }
  if (!best) throw new Error('Could not find a valid line through this data.');

  const absResiduals = xs.map((x, k) => Math.abs(ys[k] - (best.slope * x + best.intercept)));
  const trimmedIdx = absResiduals.map((r, k) => k).sort((a, b) => absResiduals[a] - absResiduals[b]).slice(0, h);
  const refit = fitPolynomial(trimmedIdx.map(k => xs[k]), trimmedIdx.map(k => ys[k]), 1);
  const [intercept, slope] = refit.coeffs;

  const predict = x => slope * x + intercept;
  return { type: 'lts', slope, intercept, trimmedCount: h, totalCount: n, r2: computeR2(xs, ys, predict), predict };
}

// Same elemental-subset-and-trim approach as fitLeastTrimmedSquares above,
// but for y = a*e^(b*x) instead of a straight line. The elemental fit is
// an exact line through two points' (x, ln(y)) — exponential's usual
// linearization — but residuals for trimming/scoring are computed back in
// original Y units, same as everywhere else in this file. Tested against
// this file's data.txt corpus (every eligible dataset — Y > 0 throughout —
// forced through exponential only, no model selection): plain
// least-squares exponential scored a median held-out test R² of -2.5946;
// trimming to 90% kept improved that to roughly -1.3 to -1.4 (repeated
// runs vary slightly since the elemental search is randomized). 89% kept
// scored marginally better in both runs (~-1.20 to -1.25), but the gap
// was within the run-to-run noise, so 90% was kept to match
// fitLeastTrimmedSquares' default rather than add an inconsistent value.
function fitTrimmedExponential(xs, ys, { iterations = 500, trimFraction = 0.90, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (ys.some(y => y <= 0)) throw new Error('Exponential fit needs every Y value to be greater than 0.');
  if (n < 4) throw new Error('Need at least 4 data points for a trimmed exponential fit.');
  const h = Math.max(2, Math.round(n * trimFraction));
  const lnYs = ys.map(Math.log);

  let best = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * n);
    if (j === i) j = (j + 1) % n;
    if (xs[i] === xs[j]) continue;

    const slope = (lnYs[j] - lnYs[i]) / (xs[j] - xs[i]);
    const intercept = lnYs[i] - slope * xs[i];
    const a = Math.exp(intercept), b = slope;
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const predict = x => a * Math.exp(b * x);
    const sqResiduals = xs.map((x, k) => (ys[k] - predict(x)) ** 2);
    const trimmedSum = [...sqResiduals].sort((p, q) => p - q).slice(0, h).reduce((s, v) => s + v, 0);
    if (Number.isFinite(trimmedSum) && (!best || trimmedSum < best.trimmedSum)) best = { a, b, trimmedSum };
  }
  if (!best) throw new Error('Could not find a valid exponential fit through this data.');

  const bestPredict = x => best.a * Math.exp(best.b * x);
  const absResiduals = xs.map((x, k) => Math.abs(ys[k] - bestPredict(x)));
  const trimmedIdx = absResiduals.map((r, k) => k).sort((p, q) => absResiduals[p] - absResiduals[q]).slice(0, h);
  const refit = fitExponential(trimmedIdx.map(k => xs[k]), trimmedIdx.map(k => ys[k]));

  return { type: 'trimmed-exponential', a: refit.a, b: refit.b, trimmedCount: h, totalCount: n, r2: computeR2(xs, ys, refit.predict), predict: refit.predict };
}

// Same idea as fitTrimmedExponential, applied to the S-curve: no closed-form
// 2-point fit exists for a logistic (3 free parameters, fit by gradient
// descent), so each trial instead runs a fast, low-iteration fitLogisticCurve
// on a random subset of the data, scores ALL points by trimmed squared
// residual against that trial fit, and keeps whichever trial's inlier set
// scored best. The winning trial's inlier points are then refit with a full,
// default-iteration fitLogisticCurve for the final answer — same
// "cheap search, expensive final refit" split fitTrimmedExponential uses.
function fitTrimmedLogisticCurve(xs, ys, { iterations = 200, trimFraction = 0.90, subsetFrac = 0.5, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 5) throw new Error('Need at least 5 data points for a trimmed S-curve fit.');
  const h = Math.max(3, Math.round(n * trimFraction));
  const subsetSize = Math.max(3, Math.round(n * subsetFrac));

  let best = null;
  const random = createSeededRandom(seed);
  for (let it = 0; it < iterations; it++) {
    const idx = new Set();
    let attempts = 0;
    while (idx.size < subsetSize && attempts < 50) { idx.add(Math.floor(random() * n)); attempts++; }
    if (idx.size < 3) continue;
    const rows = [...idx];

    let trialFit;
    try { trialFit = fitLogisticCurve(rows.map(k => xs[k]), rows.map(k => ys[k]), { iters: 200 }); } catch (e) { continue; }
    if (!Number.isFinite(trialFit.a) || !Number.isFinite(trialFit.b) || !Number.isFinite(trialFit.c)) continue;

    const sqResiduals = xs.map((x, k) => (ys[k] - trialFit.predict(x)) ** 2);
    const trimmedSum = [...sqResiduals].sort((p, q) => p - q).slice(0, h).reduce((s, v) => s + v, 0);
    if (Number.isFinite(trimmedSum) && (!best || trimmedSum < best.trimmedSum)) best = { predict: trialFit.predict, trimmedSum };
  }
  if (!best) throw new Error('Could not find a valid S-curve fit through this data.');

  const absResiduals = xs.map((x, k) => Math.abs(ys[k] - best.predict(x)));
  const trimmedIdx = absResiduals.map((r, k) => k).sort((p, q) => absResiduals[p] - absResiduals[q]).slice(0, h);
  const refit = fitLogisticCurve(trimmedIdx.map(k => xs[k]), trimmedIdx.map(k => ys[k]));

  return { type: 'trimmed-logistic', a: refit.a, b: refit.b, c: refit.c, trimmedCount: h, totalCount: n, r2: computeR2(xs, ys, refit.predict), predict: refit.predict };
}

// Persistence baselines — not real candidates in Auto-selection, but the
// standard sanity check in time-series forecasting evaluation: before
// trusting any fitted model, check whether a forecast that ignores the
// model entirely (just carries the last known value forward, or continues
// its trend) would have done just as well or better. If a fitted model
// can't beat this, the model isn't adding anything.

// "Naive"/flat persistence: predict the last training Y for every point,
// no matter what X is. Assumes the series is closer to a random walk than
// to any of the fitted shapes.
function fitPersistence(xs, ys) {
  if (!ys.length) throw new Error('Need at least 1 data point for a persistence fit.');
  const lastY = ys[ys.length - 1];
  const predict = () => lastY;
  return { type: 'persistence', lastY, r2: computeR2(xs, ys, predict), predict };
}

// "Drift"/trend persistence: same idea, but continues the straight-line
// trend from the first to the last training point instead of holding flat
// — the standard drift method from time-series forecasting (Hyndman &
// Athanasopoulos). Only uses the two endpoints, not a full regression, so
// it stays a "did you even need to fit anything" baseline rather than
// competing with fitted lines on their own terms.
function fitPersistenceDrift(xs, ys) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a persistence-drift fit.');
  const firstX = xs[0], firstY = ys[0];
  const lastX = xs[n - 1], lastY = ys[n - 1];
  if (lastX === firstX) throw new Error('Every X value is identical — cannot continue a trend.');
  const slope = (lastY - firstY) / (lastX - firstX);
  const predict = x => lastY + slope * (x - lastX);
  return { type: 'persistence-drift', lastX, lastY, slope, r2: computeR2(xs, ys, predict), predict };
}

function formatPersistenceEquation(fit) {
  return `y = ${fit.lastY.toFixed(4)}  (last training value, held constant)`;
}

function formatPersistenceDriftEquation(fit) {
  const sign = fit.slope >= 0 ? '+' : '-';
  return `y = ${fit.lastY.toFixed(4)} ${sign} ${Math.abs(fit.slope).toFixed(4)}*(x - ${fit.lastX.toFixed(4)})  (training trend continued linearly)`;
}

// AR(1): y_t = c + phi*y_(t-1) + noise — the simplest model that actually
// targets autocorrelated residuals directly (fit via ordinary least squares
// of y[1:] on y[:-1]), motivated by finding strong positive autocorrelation
// (Durbin-Watson well below 1, lag-1 autocorrelation 0.55-0.98) in the
// residuals of this corpus's worst-extrapolating datasets — none of the
// existing candidates model serial structure at all, they're all plain
// y=f(x) curves.
//
// predict(x) has to bridge that gap: every other candidate treats x as an
// arbitrary continuous input, but AR(1) only knows how to step forward one
// ROW at a time. Following the same convention fitPersistence/
// fitPersistenceDrift already use (rows are an implicit, evenly-spaced
// sequence), predict(x) converts x into a whole number of steps past the
// last training row (using the training data's own average X spacing) and
// applies the recursion that many times from the last observed Y. Steps <= 0
// (asking for a point at or before the last training row) just return that
// last Y unchanged — this model has no way to distinguish separate points
// within a single step.
// `shrinkage` (k) pulls phi toward 0 (a Bayesian ridge-style prior — a
// weaker "assume no autocorrelation until proven otherwise" belief that
// fades as more pairs accumulate): phi_used = phi_ols * pairs/(pairs+k).
// Motivated by exactly the failure this model has on tiny FIT windows: with
// only a handful of pairs, phi_ols can land close to or past 1 just from
// sampling noise, and the recursive predict() then diverges explosively
// over even a short CHECK/TEST horizon (confirmed: "Average high and low
// temperature by day"'s FIT-half phi came out at 0.9965 — barely stable —
// and predict() blew past the real data by TEST's end). Shrinking phi
// toward 0 with few pairs makes the recursion converge toward the mean
// faster instead, trading some fit quality for a lot more stability
// exactly when the estimate is least trustworthy; with many pairs the
// shrinkage fades out and phi_used approaches the plain OLS estimate.
// Default of 1.5 found by sweeping k against this file's corpus (as a
// CANDIDATES pool member, scored on held-out test R²): a narrow peak —
// 1.2-1.4 still let an occasional badly-regularized AR(1) through to
// disaster (aggregate average swings as low as -5), while 1.5-1.6 give the
// best median found in this whole investigation (0.9259, vs 0.8827 without
// AR(1) at all) with a healthy, stable average (~0.455); higher values
// (1.8-4) are safe but give up some of that gain.
function fitAR1(xs, ys, { shrinkage = 1.5 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for an AR(1) fit.');
  const yLag = ys.slice(0, -1), yNow = ys.slice(1);
  const line = fitPolynomial(yLag, yNow, 1);
  const [, phiOls] = line.coeffs;
  const pairs = yLag.length;
  const phi = phiOls * pairs / (pairs + shrinkage);
  const yLagMean = yLag.reduce((a, v) => a + v, 0) / pairs;
  const yNowMean = yNow.reduce((a, v) => a + v, 0) / pairs;
  const c = yNowMean - phi * yLagMean;

  const lastY = ys[n - 1], lastX = xs[n - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;
  // Same chart-only stability bound as fitARIMA's forecastW (see its
  // comment) -- phi is shrunk toward 0 but never hard-capped below 1, so a
  // long recursive extrapolation can still drift arbitrarily far from the
  // real data on a chart that samples far past the last training row.
  // Doesn't touch oneStepPredict/r2 below, so it can't affect selection or
  // the reported accuracy.
  const yDataRange = (Math.max(...ys) - Math.min(...ys)) || 1;
  const forecastBoundMultiplier = 3;
  const clampForecast = (v) => {
    const delta = v - lastY;
    const bound = forecastBoundMultiplier * yDataRange;
    return delta > bound ? lastY + bound : delta < -bound ? lastY - bound : v;
  };

  // Second argument (knownPrevY) is optional and only meant for backtesting
  // against real, already-observed data: when the caller actually knows the
  // true value one row back (e.g. walking forward through a held-out CHECK
  // or TEST window), pass it here to get a genuine one-step-ahead forecast
  // instead of the multi-step recursion below. Confirmed empirically (see
  // ar1_onestep_vs_multistep.js / ar1_onestep_selected_subset.js) that this
  // matters a lot once more than ~1 step separates the last training point
  // from a test point: the recursive version has no real data to check
  // itself against as it goes, so phi<1 pulls every step further toward the
  // unconditional mean c/(1-phi) regardless of what the real series does,
  // while re-anchoring to the real previous value every step keeps it
  // honest. On the 18 real datasets where AR(1) is selected or nearly
  // selected by selectByExtrapolation, this took trimmed-average held-out
  // test R² from 0.7698 to 0.8613 and the median from 0.70 to 0.91, with no
  // leakage: knownPrevY is always a value from BEFORE the point being
  // predicted, never the point itself or anything later.
  // Without knownPrevY (e.g. drawing the extrapolated curve on the chart
  // past the end of all real data, or the single "Predict at X" field),
  // there IS no real previous value to anchor to that far out, so it falls
  // back to the original multi-step recursion from the last training point.
  // The one-step-ahead fitted curve across the training rows themselves
  // (row 0 anchored to its own real value, since there's no row before it
  // to forecast from; every row after that is c + phi*previous row's real
  // y) -- reused below so the chart's line actually traces something
  // through the fit range, instead of degenerating to a flat line at the
  // last training value (see predict()'s own comment for why that
  // happened: steps=round((x-lastX)/avgStep) is exactly 0 for every x at
  // or before the last training row, since (x-lastX)<=0 gets clamped to 0
  // by the Math.max — the recursion loop below never even runs for those
  // x's, so it just returns lastY unchanged for the ENTIRE fit range).
  // Row 0 and the LAST row both anchor to their own real value (not the
  // model's one-step prediction of them) -- row 0 because there's no prior
  // row to forecast it from, and the last row so the chart's fitted curve
  // ends EXACTLY where the extrapolation recursion begins (which anchors
  // to the real lastY, same as it always has). Without this, the two
  // pieces of the chart's line visually discontinue at x=lastX: the fitted
  // curve ends at the model's own one-step guess for the last row
  // (c + phi*ys[n-2]), which generally isn't the same number as the real
  // ys[n-1] the extrapolation then starts from, producing a visible jump
  // right at the fit/extrapolation seam (confirmed directly: a test case
  // jumped from 5.0056 to 5.9998 crossing that boundary before this fix).
  // This array is for the CHART ONLY -- oneStepPredict below (used for the
  // FIT r2) stays on the unmodified one-step residuals, since forcing the
  // last row to "predict" itself exactly would artificially zero out that
  // row's real residual and inflate r2.
  const oneStepFittedForChart = ys.map((y, i) => (i === 0 || i === n - 1) ? y : c + phi * ys[i - 1]);
  const oneStepFitted = ys.map((y, i) => (i === 0 ? y : c + phi * ys[i - 1]));

  const predict = (x, knownPrevY) => {
    if (typeof knownPrevY === 'number' && Number.isFinite(knownPrevY)) {
      return c + phi * knownPrevY;
    }
    // Within (or at) the training range: interpolate the real one-step
    // fitted curve between training rows, rather than the flat lastY bug
    // described above. Charts call predict() at a dense, continuous sweep
    // of x's, most of which don't land exactly on a real row.
    // `forward` handles a DESCENDING X column the same way fitARIMA's
    // predict does (see its own comment for the full story and how it was
    // confirmed) -- without it, a descending series' lastX is actually the
    // smallest x, so almost the whole chart range wrongly fell into the
    // extrapolation branch below instead of this one.
    const forward = avgStep >= 0;
    if (forward ? x <= lastX : x >= lastX) {
      if (forward ? x <= xs[0] : x >= xs[0]) return oneStepFittedForChart[0];
      for (let i = 1; i < n; i++) {
        if (forward ? x <= xs[i] : x >= xs[i]) {
          const span = xs[i] - xs[i - 1];
          const frac = span !== 0 ? (x - xs[i - 1]) / span : 0;
          return oneStepFittedForChart[i - 1] + frac * (oneStepFittedForChart[i] - oneStepFittedForChart[i - 1]);
        }
      }
      return oneStepFittedForChart[n - 1];
    }
    // Extrapolation: AR(1) only has a defined value at each whole step past
    // the last training row (lastX + k*avgStep) -- interpolating linearly
    // BETWEEN consecutive whole-step forecasts gives a continuous-looking
    // curve for the chart, instead of snapping each x to its NEAREST whole
    // step (Math.round), which is what produced the staircase: many
    // adjacent x's round to the same integer step and get an identical y,
    // then the next x jumps straight to the next step's y with no
    // transition.
    const rawSteps = (x - lastX) / avgStep;
    const stepsLo = Math.max(0, Math.floor(rawSteps));
    let yLo = lastY;
    for (let s = 0; s < stepsLo; s++) yLo = clampForecast(c + phi * yLo);
    const yHi = clampForecast(c + phi * yLo);
    const frac = rawSteps - stepsLo;
    return clampForecast(yLo + frac * (yHi - yLo));
  };

  // FIT r2 is measured on the one-step-ahead relationship (y_now vs.
  // predicted from y_lag) — the thing AR(1) actually fits — not on
  // predict(x)'s multi-step recursion, which isn't what least squares
  // minimized.
  const oneStepPredict = oneStepFitted.slice(1); // same formula as above, reused rather than recomputed
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < yNow.length; i++) {
    ssRes += (yNow[i] - oneStepPredict[i]) ** 2;
    ssTot += (yNow[i] - yNowMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { type: 'AR(1)', c, phi, r2, predict };
}

function formatAR1Equation(fit) {
  const sign = fit.c >= 0 ? '+' : '-';
  return `y_next = ${fit.phi.toFixed(4)}*y_previous ${sign} ${Math.abs(fit.c).toFixed(4)}  (one step per row, stepping forward from the last training row)`;
}

// AR(2): y_t = c + phi1*y_(t-1) + phi2*y_(t-2) + noise — one more lag than
// AR(1), for the autocorrelated cases a single lag doesn't fully capture
// (e.g. residual structure with damped oscillation or a slower-decaying
// echo, which AR(1) can't represent no matter what phi it picks). Fit by
// OLS of y[2:] on (y[1:-1], y[:-2]) via fitMultiLinear, same idea as AR(1)'s
// regression of y_now on y_lag but with a second predictor.
// Same shrinkage trick as AR(1) and for the same reason: with only a
// handful of pairs (this pool's FIT windows are tiny), the OLS (phi1, phi2)
// pair can land in or near the unstable region and predict()'s recursion
// then diverges. Both coefficients are shrunk toward 0 together by the same
// pairs/(pairs+shrinkage) factor AR(1) uses — shrinking the whole
// coefficient vector toward the "no autocorrelation" origin rather than
// trying to shrink toward stationarity boundaries directly.
function fitAR2(xs, ys, { shrinkage = 1.5 } = {}) {
  const n = xs.length;
  if (n < 5) throw new Error('Need at least 5 data points for an AR(2) fit.');
  const yLag1 = ys.slice(1, -1), yLag2 = ys.slice(0, -2), yNow = ys.slice(2);
  const pairs = yNow.length;
  const X = yLag1.map((v, i) => [v, yLag2[i]]);
  const fit = fitMultiLinear(X, yNow);
  const shrink = pairs / (pairs + shrinkage);
  const phi1 = fit.coeffs[0] * shrink;
  const phi2 = fit.coeffs[1] * shrink;
  const yLag1Mean = yLag1.reduce((a, v) => a + v, 0) / pairs;
  const yLag2Mean = yLag2.reduce((a, v) => a + v, 0) / pairs;
  const yNowMean = yNow.reduce((a, v) => a + v, 0) / pairs;
  const c = yNowMean - phi1 * yLag1Mean - phi2 * yLag2Mean;

  const lastY = ys[n - 1], prevY = ys[n - 2], lastX = xs[n - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;

  // Same bug/fix as fitAR1's own oneStepFittedForChart comment: without
  // this, predict(x) (the only form the chart ever calls) is flat for
  // every x at or before the last training row (steps=0 there), then
  // staircases past it. Rows 0 and 1 have no two-lag one-step prediction
  // available yet, so they just show the real value; the last row is
  // forced to its real value too, so the chart curve ends exactly where
  // the extrapolation recursion below begins.
  const oneStepFittedForChart = ys.map((y, i) => (i < 2 || i === n - 1) ? y : c + phi1 * ys[i - 1] + phi2 * ys[i - 2]);

  const predict = x => {
    if (x <= lastX) {
      if (x <= xs[0]) return oneStepFittedForChart[0];
      for (let i = 1; i < n; i++) {
        if (x <= xs[i]) {
          const span = xs[i] - xs[i - 1];
          const frac = span !== 0 ? (x - xs[i - 1]) / span : 0;
          return oneStepFittedForChart[i - 1] + frac * (oneStepFittedForChart[i] - oneStepFittedForChart[i - 1]);
        }
      }
      return oneStepFittedForChart[n - 1];
    }
    // Extrapolation: interpolate BETWEEN consecutive whole-step forecasts
    // instead of snapping each x to its nearest whole step (same staircase
    // bug and fix as fitAR1).
    const rawSteps = (x - lastX) / avgStep;
    const stepsLo = Math.max(0, Math.floor(rawSteps));
    const stepsHi = stepsLo + 1;
    let y2 = prevY, y1 = lastY;
    let yAtLo = lastY; // step 0 is just the real last training value, before any recursion
    for (let s = 0; s < stepsHi; s++) {
      const next = c + phi1 * y1 + phi2 * y2;
      y2 = y1; y1 = next;
      if (s === stepsLo - 1) yAtLo = next; // captures the forecast AT step stepsLo, if stepsLo >= 1
    }
    const yAtHi = y1;
    const frac = rawSteps - stepsLo;
    return yAtLo + frac * (yAtHi - yAtLo);
  };

  const oneStepPredict = yLag1.map((v, i) => c + phi1 * v + phi2 * yLag2[i]);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < yNow.length; i++) {
    ssRes += (yNow[i] - oneStepPredict[i]) ** 2;
    ssTot += (yNow[i] - yNowMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { type: 'AR(2)', c, phi1, phi2, r2, predict };
}

function formatAR2Equation(fit) {
  const sign1 = fit.phi1 >= 0 ? '+' : '-';
  const sign2 = fit.phi2 >= 0 ? '+' : '-';
  const signC = fit.c >= 0 ? '+' : '-';
  return `y_next = ${sign1 === '+' ? '' : '-'}${Math.abs(fit.phi1).toFixed(4)}*y_previous ${sign2} ${Math.abs(fit.phi2).toFixed(4)}*y_two_back ${signC} ${Math.abs(fit.c).toFixed(4)}  (one step per row, stepping forward from the last two training rows)`;
}

// ARMA(1,1): y_t = c + phi*y_(t-1) + theta*e_(t-1) + e_t. Adds a
// moving-average term on top of AR(1)'s autoregressive one — useful when a
// single shock's effect on the residual decays after just one step (MA
// side) on top of the slower, persistent echo AR(1) alone captures. Fit via
// the Hannan-Rissanen two-step method (no iterative MLE/nonlinear solver
// needed, consistent with the rest of this file's plain-OLS-based fits):
//   1) Fit a longer AR model (order = min(5, floor(n/3)), at least 2) by
//      OLS to get proxy innovations (residuals) e_t.
//   2) Regress y_t on (y_(t-1), e_(t-1)) — using the step-1 residuals as a
//      stand-in for the true unobserved e_(t-1) — via fitMultiLinear to get
//      phi and theta directly.
// Same shrink-toward-0 prior as AR(1)/AR(2) and for the same reason (tiny
// FIT windows make raw OLS phi/theta unstable), applied to both phi and
// theta together.
function fitARMA11(xs, ys, { shrinkage = 1.5 } = {}) {
  const n = xs.length;
  if (n < 6) throw new Error('Need at least 6 data points for an ARMA(1,1) fit.');
  const arOrder = Math.max(2, Math.min(5, Math.floor(n / 3)));

  // Step 1: long AR(arOrder) by OLS to get proxy residuals.
  const yLagsMatrix = [];
  const yTargetLong = [];
  for (let t = arOrder; t < n; t++) {
    yTargetLong.push(ys[t]);
    const row = [];
    for (let l = 1; l <= arOrder; l++) row.push(ys[t - l]);
    yLagsMatrix.push(row);
  }
  const longAr = fitMultiLinear(yLagsMatrix, yTargetLong);
  const eProxy = yTargetLong.map((y, i) => y - longAr.predict(yLagsMatrix[i])); // e_t for t = arOrder..n-1

  // Step 2: regress y_t on (y_(t-1), e_(t-1)). e_(t-1) is only known from
  // t = arOrder+1 onward (e_arOrder is the first available proxy residual).
  const X = [], yTarget = [];
  for (let t = arOrder + 1; t < n; t++) {
    const ePrev = eProxy[t - 1 - arOrder];
    X.push([ys[t - 1], ePrev]);
    yTarget.push(ys[t]);
  }
  if (X.length < 2) throw new Error('Not enough data left after the AR warm-up for an ARMA(1,1) fit.');
  const step2 = fitMultiLinear(X, yTarget);
  const pairs = X.length;
  const shrink = pairs / (pairs + shrinkage);
  const phi = step2.coeffs[0] * shrink;
  const theta = step2.coeffs[1] * shrink;
  const yLagMean = X.reduce((a, row) => a + row[0], 0) / pairs;
  const eLagMean = X.reduce((a, row) => a + row[1], 0) / pairs;
  const yNowMean = yTarget.reduce((a, v) => a + v, 0) / pairs;
  const c = yNowMean - phi * yLagMean - theta * eLagMean;

  const lastY = ys[n - 1], lastX = xs[n - 1];
  const lastE = eProxy[eProxy.length - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;

  // Same bug/fix as fitAR1's own oneStepFittedForChart comment: without
  // this, predict(x) (the only form the chart ever calls) is flat for
  // every x at or before the last training row, then staircases past it.
  // Rows 0..arOrder have no one-step prediction available yet (that's the
  // warm-up window used to build the proxy residuals), so they just show
  // the real value; the last row is forced to its real value too, so the
  // chart curve ends exactly where the extrapolation recursion begins.
  const oneStepFittedForChart = ys.map((y, i) => {
    if (i <= arOrder || i === n - 1) return y;
    const ePrev = eProxy[i - 1 - arOrder];
    return c + phi * ys[i - 1] + theta * ePrev;
  });

  // Forecasting an MA term past one step uses e=0 (its expected value) —
  // the last known residual only affects the very first predicted step.
  const predict = x => {
    if (x <= lastX) {
      if (x <= xs[0]) return oneStepFittedForChart[0];
      for (let i = 1; i < n; i++) {
        if (x <= xs[i]) {
          const span = xs[i] - xs[i - 1];
          const frac = span !== 0 ? (x - xs[i - 1]) / span : 0;
          return oneStepFittedForChart[i - 1] + frac * (oneStepFittedForChart[i] - oneStepFittedForChart[i - 1]);
        }
      }
      return oneStepFittedForChart[n - 1];
    }
    // Extrapolation: interpolate BETWEEN consecutive whole-step forecasts
    // instead of snapping each x to its nearest whole step (same staircase
    // bug and fix as fitAR1).
    const rawSteps = (x - lastX) / avgStep;
    const stepsLo = Math.max(0, Math.floor(rawSteps));
    const stepsHi = stepsLo + 1;
    let y = lastY, e = lastE;
    let yAtLo = lastY;
    for (let s = 0; s < stepsHi; s++) {
      y = c + phi * y + theta * e;
      e = 0;
      if (s === stepsLo - 1) yAtLo = y;
    }
    const yAtHi = y;
    const frac = rawSteps - stepsLo;
    return yAtLo + frac * (yAtHi - yAtLo);
  };

  const oneStepPredict = X.map(row => c + phi * row[0] + theta * row[1]);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < yTarget.length; i++) {
    ssRes += (yTarget[i] - oneStepPredict[i]) ** 2;
    ssTot += (yTarget[i] - yNowMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { type: 'ARMA(1,1)', c, phi, theta, r2, predict };
}

function formatARMA11Equation(fit) {
  const signPhi = fit.phi >= 0 ? '+' : '-';
  const signTheta = fit.theta >= 0 ? '+' : '-';
  const signC = fit.c >= 0 ? '+' : '-';
  return `y_next = ${signPhi === '+' ? '' : '-'}${Math.abs(fit.phi).toFixed(4)}*y_previous ${signTheta} ${Math.abs(fit.theta).toFixed(4)}*e_previous ${signC} ${Math.abs(fit.c).toFixed(4)}  (one step per row; e_previous decays to 0 beyond the first forecast step)`;
}

// ARIMA(p,d,q): generalizes AR(1)/AR(2)/ARMA(1,1) above with the one piece
// they're missing — the "I" (differencing) term, for genuinely NON-
// STATIONARY series (a stochastic trend/random walk), where fitting a
// mean-reverting AR/ARMA model directly to the levels is the wrong model.
// Differences the series d times, fits AR(p)+MA(q) on the DIFFERENCED
// series via the same Hannan-Rissanen two-step OLS method ARMA(1,1) uses
// (generalized to arbitrary small p/q), then integrates forecasts back up
// (cumulative-sum) to the original Y scale. Same shrink-toward-0 prior on
// the AR/MA coefficients as the other AR-family models here, for the same
// reason (tiny FIT windows destabilize raw OLS).
//
// predict(x, knownPrevY) follows the exact same convention fitAR1
// introduced (see its own comment above): when a caller walking forward
// through real data reveals the REAL previous y (via
// computeR2Sequential), this re-derives the REAL differenced history and
// REAL one-step residuals from that actual data and forecasts exactly ONE
// step — rather than recursing blind from the end of training. This
// matters more here than for AR(1)/AR(2): differencing then integrating
// compounds recursive error through a cumulative sum on top of the
// recursion itself, and an early prototype of this model, backtested
// without real-data re-anchoring, suffered catastrophic blowups on longer
// series (e.g. a ~400-step blind recursion on "Movie Gross" scored
// R²=-198) purely from that compounding, not from any flaw in the
// underlying ARIMA math — confirmed by the fact that adding this exact
// re-anchoring took that same dataset to R²=0.999 with no other change.
function differenceOnceForARIMA(series) {
  const out = [];
  for (let t = 1; t < series.length; t++) out.push(series[t] - series[t - 1]);
  return out;
}
function fitARIMA(xs, ys, { p = 1, d = 1, q = 0, shrinkage = 1.5 } = {}) {
  const n = xs.length;
  const lastLevels = []; // last value of the k-times-differenced series (k=0..d-1), needed to integrate forecasts back up
  let series = ys.slice();
  for (let k = 0; k < d; k++) { lastLevels.push(series[series.length - 1]); series = differenceOnceForARIMA(series); }
  const w = series; // the series actually modeled (length n-d)
  const minNeeded = Math.max(p, q) + p + q + 3;
  if (w.length < minNeeded) throw new Error(`Need at least ${minNeeded + d} data points for ARIMA(${p},${d},${q}).`);

  const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
  let phis = new Array(p).fill(0), thetas = new Array(q).fill(0), c = 0, eProxy = null, arOrderUsed = 0;

  if (p === 0 && q === 0) {
    c = meanOf(w);
  } else if (q === 0) {
    const yLags = [], yTarget = [];
    for (let t = p; t < w.length; t++) { yTarget.push(w[t]); const row = []; for (let l = 1; l <= p; l++) row.push(w[t - l]); yLags.push(row); }
    const fit = fitMultiLinear(yLags, yTarget);
    const pairs = yTarget.length;
    const shrink = pairs / (pairs + shrinkage);
    phis = fit.coeffs.map(v => v * shrink);
    const means = Array.from({ length: p }, (_, l) => meanOf(yLags.map(row => row[l])));
    c = meanOf(yTarget) - phis.reduce((s, phi, l) => s + phi * means[l], 0);
  } else {
    arOrderUsed = Math.max(p + q + 2, Math.min(6, Math.floor(w.length / 3)));
    const yLagsLong = [], yTargetLong = [];
    for (let t = arOrderUsed; t < w.length; t++) { yTargetLong.push(w[t]); const row = []; for (let l = 1; l <= arOrderUsed; l++) row.push(w[t - l]); yLagsLong.push(row); }
    const longAr = fitMultiLinear(yLagsLong, yTargetLong);
    eProxy = yTargetLong.map((y, i) => y - longAr.predict(yLagsLong[i]));

    const X = [], yTarget2 = [];
    const start = arOrderUsed + q;
    for (let t = start; t < w.length; t++) {
      const row = [];
      for (let l = 1; l <= p; l++) row.push(w[t - l]);
      for (let l = 1; l <= q; l++) row.push(eProxy[t - l - arOrderUsed]);
      X.push(row); yTarget2.push(w[t]);
    }
    if (X.length < p + q + 2) throw new Error(`Not enough data left after warm-up for ARIMA(${p},${d},${q}).`);
    const step2 = fitMultiLinear(X, yTarget2);
    const pairs = X.length;
    const shrink = pairs / (pairs + shrinkage);
    phis = step2.coeffs.slice(0, p).map(v => v * shrink);
    thetas = step2.coeffs.slice(p, p + q).map(v => v * shrink);
    const colMeans = Array.from({ length: p + q }, (_, j) => meanOf(X.map(row => row[j])));
    c = meanOf(yTarget2) - colMeans.reduce((s, m, j) => s + step2.coeffs[j] * shrink * m, 0);
  }

  let oneStepPredict, target, oneStepStartInW;
  if (p === 0 && q === 0) { oneStepStartInW = 0; target = w; oneStepPredict = w.map(() => c); }
  else if (q === 0) { oneStepStartInW = p; target = w.slice(p); oneStepPredict = target.map((_, i) => c + phis.reduce((s, phi, l) => s + phi * w[p + i - 1 - l], 0)); }
  else {
    oneStepStartInW = arOrderUsed + q;
    target = w.slice(oneStepStartInW);
    oneStepPredict = target.map((_, i) => {
      const t = oneStepStartInW + i; let v = c;
      for (let l = 1; l <= p; l++) v += phis[l - 1] * w[t - l];
      for (let l = 1; l <= q; l++) v += thetas[l - 1] * eProxy[t - l - arOrderUsed];
      return v;
    });
  }
  const targetMean = meanOf(target);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < target.length; i++) { ssRes += (target[i] - oneStepPredict[i]) ** 2; ssTot += (target[i] - targetMean) ** 2; }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  const lastX = xs[n - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;
  // Bound for forecastW's blind multi-step recursion below -- ONLY used to
  // keep the CHART's drawn curve on-screen past the last real row; never
  // touches oneStepPredict/r2 above (what selection and the reported R²
  // actually score), so this can't change which model wins or its reported
  // accuracy. Needed because forecastW has no built-in stationarity
  // guarantee (phis/thetas are only shrunk toward 0, not hard-capped), and
  // a chart can ask for hundreds of steps past the last training row (e.g.
  // a long monthly series charted over its own full X range) -- confirmed
  // directly on "congress poll" (415 monthly rows): a mildly unstable
  // ARIMA(1,1,1) fit recursed over ~400 steps and reached y=1248 against a
  // real data range of 10-86, visibly shooting off the top of the chart.
  const yDataRange = (Math.max(...ys) - Math.min(...ys)) || 1;
  const forecastBoundMultiplier = 3;
  const clampForecast = (v) => {
    const delta = v - ys[n - 1];
    const bound = forecastBoundMultiplier * yDataRange;
    return delta > bound ? ys[n - 1] + bound : delta < -bound ? ys[n - 1] - bound : v;
  };
  const wHistoryInit = w.slice(), eHistoryInit = eProxy ? eProxy.slice() : [];
  let yHistory = ys.slice();

  // --- Chart-only one-step-fitted curve, in Y-space (not w-space), for
  // every original row -- same fix and same reasoning as fitAR1's own
  // oneStepFittedForChart above: without this, predict(x) (no knownPrevY,
  // the only form the chart ever calls) returns a flat ys[n-1] for every x
  // at or before the last training row (steps=round((x-lastX)/avgStep) is
  // exactly 0 there), then jumps to a staircase of whole-step forecasts
  // past it. Fix needs one extra piece ARIMA has that AR(1) doesn't: the
  // one-step forecasts above (oneStepPredict) live in DIFFERENCED (w)
  // space, so each one has to be integrated back up through d levels using
  // the REAL historical level values at the right position -- not a
  // fictional recursively-built level the way forecastW does for genuine
  // extrapolation -- to land back on the original Y scale at the right row.
  const levelsStatic = [ys];
  for (let k = 0; k < d; k++) levelsStatic.push(differenceOnceForARIMA(levelsStatic[levelsStatic.length - 1]));
  // levelsStatic[k][m] is the real k-times-differenced value whose right
  // edge is original row (m+k) -- so levelsStatic[k][T-1-k] is the real
  // value at level k immediately before row T, for any k.
  const oneStepFittedForChart = ys.map((y, i) => i); // placeholder shape, filled below
  for (let T = 0; T < n; T++) {
    const j = T - d - oneStepStartInW; // index into oneStepPredict/target, if this row has one
    if (j >= 0 && j < oneStepPredict.length && T !== n - 1) {
      let v = oneStepPredict[j];
      for (let k = d - 1; k >= 0; k--) v = levelsStatic[k][T - 1 - k] + v;
      oneStepFittedForChart[T] = v;
    } else {
      // No model-based one-step value available yet (still inside the
      // warm-up window) -- OR this is the last row, forced to the real
      // value so the chart's fitted curve ends exactly where the
      // extrapolation recursion below begins (same real anchor, same
      // "seamless join" fix as fitAR1's chart curve uses).
      oneStepFittedForChart[T] = ys[T];
    }
  }

  function computeWLevels(yArr) { let s = yArr.slice(); for (let k = 0; k < d; k++) s = differenceOnceForARIMA(s); return s; }
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
    for (let t = minHist; t < wHist.length; t++) { const forecast = oneStepWForecast(wHist.slice(0, t), eh); eh.push(wHist[t] - forecast); }
    return eh;
  }
  function forecastW(steps) {
    const wh = wHistoryInit.slice(), eh = eHistoryInit.slice();
    const forecasts = [];
    for (let s = 0; s < steps; s++) {
      let v = c;
      for (let l = 1; l <= p; l++) v += phis[l - 1] * wh[wh.length - l];
      for (let l = 1; l <= q; l++) v += thetas[l - 1] * (eh.length >= l ? eh[eh.length - l] : 0);
      wh.push(v); eh.push(0);
      forecasts.push(v);
    }
    return forecasts;
  }

  const predict = (x, knownPrevY) => {
    if (typeof knownPrevY === 'number' && Number.isFinite(knownPrevY)) {
      if (knownPrevY !== yHistory[yHistory.length - 1]) yHistory.push(knownPrevY); // skip the harness's first call, which just re-reveals the last training point already in yHistory
      const levels = [yHistory];
      for (let k = 0; k < d; k++) levels.push(differenceOnceForARIMA(levels[levels.length - 1]));
      const wHist = levels[d];
      const minNeededHere = Math.max(p, 1) + (q > 0 ? q : 0);
      if (wHist.length <= minNeededHere) return yHistory[yHistory.length - 1];
      const eHist = rebuildEHistory(wHist);
      const wForecast = oneStepWForecast(wHist, eHist);
      let v = wForecast;
      for (let k = d - 1; k >= 0; k--) v = levels[k][levels[k].length - 1] + v;
      return v;
    }
    // Within (or at) the training range: interpolate the real one-step
    // fitted curve between training rows, instead of the flat-line bug
    // described above.
    // "Within training range" and the bracket search below both compare x
    // against xs[i] assuming xs runs ascending (row 0 = earliest). This
    // project's own Desmos range shorthand also supports a DESCENDING X
    // column (e.g. "2026,2025.9167...1991.55", most-recent-first -- see
    // expandRanges' comment) -- avgStep's sign already reflects that (it's
    // negative when xs runs high-to-low), so `forward` below just asks
    // "which direction does row order actually go in X" and flips every
    // comparison to match, instead of assuming ascending. Without this,
    // a descending series' `lastX` (xs[n-1], the row-order-wise last point)
    // is actually the SMALLEST x, so nearly every x the chart asks about
    // (everything above that minimum) wrongly fell into the blind
    // extrapolation branch below -- confirmed on "congress poll" (415
    // monthly rows, descending X): the resulting rawSteps came out wildly
    // negative (training data was being treated as centuries of
    // extrapolation in the wrong direction), which the clamp below can't
    // fully absorb since it's multiplied by that same runaway `frac`.
    const forward = avgStep >= 0;
    if (forward ? x <= lastX : x >= lastX) {
      if (forward ? x <= xs[0] : x >= xs[0]) return oneStepFittedForChart[0];
      for (let i = 1; i < n; i++) {
        if (forward ? x <= xs[i] : x >= xs[i]) {
          const span = xs[i] - xs[i - 1];
          const frac = span !== 0 ? (x - xs[i - 1]) / span : 0;
          return oneStepFittedForChart[i - 1] + frac * (oneStepFittedForChart[i] - oneStepFittedForChart[i - 1]);
        }
      }
      return oneStepFittedForChart[n - 1];
    }
    // Extrapolation: interpolate linearly BETWEEN consecutive whole-step
    // forecasts, instead of snapping each x to its NEAREST whole step
    // (same staircase bug and same fix as fitAR1's chart curve).
    const rawSteps = (x - lastX) / avgStep;
    const stepsLo = Math.max(0, Math.floor(rawSteps));
    const stepsHi = stepsLo + 1;
    const wForecasts = forecastW(stepsHi);
    let levelForecasts = wForecasts;
    for (let k = d - 1; k >= 0; k--) {
      let running = lastLevels[k];
      const next = [];
      for (const v of levelForecasts) { running += v; next.push(running); }
      levelForecasts = next;
    }
    levelForecasts = levelForecasts.map(clampForecast);
    const yLo = stepsLo === 0 ? ys[n - 1] : levelForecasts[stepsLo - 1];
    const yHi = levelForecasts[stepsHi - 1];
    const frac = rawSteps - stepsLo;
    return clampForecast(yLo + frac * (yHi - yLo));
  };

  return { type: `ARIMA(${p},${d},${q})`, p, d, q, phis, thetas, c, r2, predict };
}

function formatARIMAEquation(fit) {
  const lagTerms = fit.phis.map((phi, i) => `${phi >= 0 ? '+' : '-'} ${Math.abs(phi).toFixed(4)}*w_(t-${i + 1})`).join(' ');
  const maTerms = fit.thetas.map((th, i) => `${th >= 0 ? '+' : '-'} ${Math.abs(th).toFixed(4)}*e_(t-${i + 1})`).join(' ');
  return `w_t = ${fit.c.toFixed(4)} ${lagTerms} ${maTerms}  (w = y differenced ${fit.d} time${fit.d === 1 ? '' : 's'}; forecasts are integrated back up to y)`;
}

// Auto-selects (p,d,q) from a small, deliberately conservative grid via an
// internal FIT/CHECK split carved out of whatever (xs, ys) it's given —
// same discipline as selectByExtrapolation's own FIT/CHECK split, so this
// can act as a single CANDIDATES pool entry (like AR(1)) rather than
// bloating the pool with several near-duplicate fixed-config ARIMA
// entries. Scored via computeR2Sequential (walk-forward, real-data
// re-anchored) on the internal CHECK split, consistent with how this
// model is meant to be evaluated everywhere else.
const ARIMA_AUTO_CONFIGS = [{ p: 1, d: 1, q: 0 }, { p: 0, d: 1, q: 1 }, { p: 1, d: 1, q: 1 }, { p: 2, d: 1, q: 0 }];
function fitARIMAAuto(xs, ys) {
  const n = xs.length;
  if (n < 12) throw new Error('Need at least 12 data points for ARIMA auto-selection.');
  const fitLen = Math.max(8, Math.floor(n * 0.7));
  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen), checkYs = ys.slice(fitLen);

  let chosenCfg = null;
  if (checkXs.length >= 3) {
    let bestCheckR2 = -Infinity;
    for (const cfg of ARIMA_AUTO_CONFIGS) {
      let fit; try { fit = fitARIMA(fitXs, fitYs, cfg); } catch (e) { continue; }
      const r2 = computeR2Sequential(fitYs[fitYs.length - 1], checkXs, checkYs, fit.predict);
      if (r2 > bestCheckR2) { bestCheckR2 = r2; chosenCfg = cfg; }
    }
  }
  if (!chosenCfg) chosenCfg = { p: 1, d: 1, q: 0 }; // not enough data to internally validate a choice — fall back to the simplest config
  return fitARIMA(xs, ys, chosenCfg);
}

// ---------------------------------------------------------------------------
// SARIMAX: Seasonal ARIMA with eXogenous regressors. ARIMA above still
// handles only a TREND (a stochastic drift, via differencing) — it has no
// notion of a repeating CYCLE (e.g. "every 12 rows the pattern repeats"),
// and no way to let another column help predict y. SARIMAX adds both:
//   - Seasonal: an extra differencing step at lag `s` (on top of the
//     regular lag-1 differencing), plus seasonal AR/MA terms at multiples
//     of `s`, generalizing fitARIMA's single differencing stage into a
//     CHAIN of stages, each with its own lag.
//   - eXogenous: other columns, known at the same row as y, handled the
//     standard "regression with ARMA errors" way — regress y on them first,
//     then run the seasonal-ARIMA machinery on the LEFTOVER residual
//     series, adding the exogenous contribution back at the end.
// This is a hand-written JS equivalent of Python's statsmodels SARIMAX, not
// a port — there's no JavaScript SARIMAX to build on, and running the real
// (Python-only) one would mean a separate Python process, which this
// project (100% client-side JS, confirmed with the user before building
// this) deliberately doesn't have.
//
// exog, if given, must be known for every forecast row too — this is a
// real, unavoidable limitation of adding exogenous regressors to ANY
// forecasting tool, not something specific to this implementation.
// predict's 3rd argument lets a caller supply that future row; without it,
// the exogenous contribution just holds at its last training-row value
// (clearly a simplification, not a real forecast of those other columns).
function diffByLag(series, lag) {
  const out = [];
  for (let t = lag; t < series.length; t++) out.push(series[t] - series[t - lag]);
  return out;
}
function fitSARIMA(xs, ys, { p = 1, d = 1, q = 0, P = 0, D = 0, Q = 0, s = 1, exog = null, shrinkage = 1.5 } = {}) {
  const n = xs.length;
  if (!Number.isInteger(s) || s < 1) s = 1;
  if (exog && exog.length && exog.some(col => col.length !== n)) {
    throw new Error('SARIMAX exogenous columns must be the same length as y.');
  }

  // --- Step 1: regress out the exogenous columns (if any); everything
  // below models the LEFTOVER residual series.
  let exogFit = null, lastExogRow = null, Xexog = null;
  let yForArima = ys;
  if (exog && exog.length) {
    Xexog = ys.map((_, i) => exog.map(col => col[i]));
    exogFit = fitMultiLinear(Xexog, ys);
    yForArima = ys.map((y, i) => y - exogFit.predict(Xexog[i]));
    lastExogRow = Xexog[n - 1];
  }

  // --- Step 2: the differencing CHAIN -- D seasonal diffs at lag s, then d
  // regular diffs at lag 1 -- generalizing fitARIMA's single lag-1 chain.
  // levelsStatic[0] = yForArima; levelsStatic[k] = that series differenced
  // k times by the stage lags below; levelsStatic[L] (L = stageLags.length)
  // is `w`, the series actually modeled.
  const stageLags = [...new Array(D).fill(s), ...new Array(d).fill(1)];
  const L = stageLags.length;
  const levelsStatic = [yForArima.slice()];
  for (const lag of stageLags) levelsStatic.push(diffByLag(levelsStatic[levelsStatic.length - 1], lag));
  const w = levelsStatic[L];
  const cumLag = [0];
  for (const lag of stageLags) cumLag.push(cumLag[cumLag.length - 1] + lag);

  const arLags = [...(p > 0 ? Array.from({ length: p }, (_, i) => i + 1) : []), ...(P > 0 ? Array.from({ length: P }, (_, i) => (i + 1) * s) : [])];
  const maLags = [...(q > 0 ? Array.from({ length: q }, (_, i) => i + 1) : []), ...(Q > 0 ? Array.from({ length: Q }, (_, i) => (i + 1) * s) : [])];
  const maxArLag = arLags.length ? Math.max(...arLags) : 0;
  const maxMaLag = maLags.length ? Math.max(...maLags) : 0;
  const minNeeded = maxArLag + maxMaLag + arLags.length + maLags.length + 3;
  if (w.length < minNeeded) {
    throw new Error(`Need more data for SARIMA(${p},${d},${q})(${P},${D},${Q})[${s}] -- have ${n} rows, need more.`);
  }

  const meanOf = a => a.reduce((sum, v) => sum + v, 0) / a.length;
  let phis = new Array(arLags.length).fill(0), thetas = new Array(maLags.length).fill(0), c = 0, eProxy = null, longArOrder = 0;

  if (arLags.length === 0 && maLags.length === 0) {
    c = meanOf(w);
  } else if (maLags.length === 0) {
    const yLags = [], yTarget = [];
    for (let t = maxArLag; t < w.length; t++) { yTarget.push(w[t]); yLags.push(arLags.map(l => w[t - l])); }
    const fit = fitMultiLinear(yLags, yTarget);
    const pairs = yTarget.length, shrink = pairs / (pairs + shrinkage);
    phis = fit.coeffs.map(v => v * shrink);
    const means = arLags.map((_, j) => meanOf(yLags.map(row => row[j])));
    c = meanOf(yTarget) - phis.reduce((sum, phi, j) => sum + phi * means[j], 0);
  } else {
    longArOrder = Math.max(maxArLag + maLags.length + 2, Math.min(10, Math.floor(w.length / 3)));
    const yLagsLong = [], yTargetLong = [];
    for (let t = longArOrder; t < w.length; t++) { yTargetLong.push(w[t]); const row = []; for (let l = 1; l <= longArOrder; l++) row.push(w[t - l]); yLagsLong.push(row); }
    const longAr = fitMultiLinear(yLagsLong, yTargetLong);
    eProxy = yTargetLong.map((y, i) => y - longAr.predict(yLagsLong[i]));

    const X = [], yTarget2 = [];
    const start = longArOrder + maxMaLag;
    for (let t = start; t < w.length; t++) {
      const row = [];
      for (const l of arLags) row.push(w[t - l]);
      for (const l of maLags) row.push(eProxy[t - l - longArOrder]);
      X.push(row); yTarget2.push(w[t]);
    }
    if (X.length < arLags.length + maLags.length + 2) throw new Error(`Not enough data left after warm-up for SARIMA(${p},${d},${q})(${P},${D},${Q})[${s}].`);
    const step2 = fitMultiLinear(X, yTarget2);
    const pairs = X.length, shrink = pairs / (pairs + shrinkage);
    phis = step2.coeffs.slice(0, arLags.length).map(v => v * shrink);
    thetas = step2.coeffs.slice(arLags.length).map(v => v * shrink);
    const colMeans = Array.from({ length: arLags.length + maLags.length }, (_, j) => meanOf(X.map(row => row[j])));
    c = meanOf(yTarget2) - colMeans.reduce((sum, m, j) => sum + step2.coeffs[j] * shrink * m, 0);
  }

  function oneStepWForecast(wHist, eHist) {
    let v = c;
    arLags.forEach((lag, idx) => { v += phis[idx] * wHist[wHist.length - lag]; });
    maLags.forEach((lag, idx) => { v += thetas[idx] * (eHist.length >= lag ? eHist[eHist.length - lag] : 0); });
    return v;
  }
  function rebuildEHistory(wHist) {
    if (!maLags.length) return [];
    const eh = [];
    const minHist = arLags.length ? Math.max(...arLags) : 1;
    for (let t = minHist; t < wHist.length; t++) { eh.push(wHist[t] - oneStepWForecast(wHist.slice(0, t), eh)); }
    return eh;
  }
  function forecastWSteps(wHistoryInit, eHistoryInit, steps) {
    const wh = wHistoryInit.slice(), eh = eHistoryInit.slice();
    const forecasts = [];
    for (let i = 0; i < steps; i++) { const v = oneStepWForecast(wh, eh); wh.push(v); eh.push(0); forecasts.push(v); }
    return forecasts;
  }
  // Integrates ONE new w-level value all the way back up to yForArima scale,
  // appending the new value at every stage's history along the way --
  // generalizes fitARIMA's "integrate back up" trick to a chain of stages
  // with possibly-different lags (lag s for seasonal stages, lag 1 for
  // regular ones), instead of always undoing a single lag-1 diff.
  function appendForecastStep(stageHistories, wValue) {
    const vals = new Array(L + 1);
    vals[L] = wValue;
    for (let k = L - 1; k >= 0; k--) {
      const hist = stageHistories[k];
      vals[k] = hist[hist.length - stageLags[k]] + vals[k + 1];
    }
    for (let k = 0; k <= L; k++) stageHistories[k] = stageHistories[k].concat([vals[k]]);
    return vals[0];
  }

  let oneStepPredict, target, oneStepStartInW;
  if (arLags.length === 0 && maLags.length === 0) { oneStepStartInW = 0; target = w; oneStepPredict = w.map(() => c); }
  else if (maLags.length === 0) {
    oneStepStartInW = maxArLag; target = w.slice(maxArLag);
    oneStepPredict = target.map((_, i) => { const t = maxArLag + i; let v = c; arLags.forEach((l, idx) => v += phis[idx] * w[t - l]); return v; });
  } else {
    oneStepStartInW = longArOrder + maxMaLag; target = w.slice(oneStepStartInW);
    oneStepPredict = target.map((_, i) => {
      const t = oneStepStartInW + i; let v = c;
      arLags.forEach((l, idx) => v += phis[idx] * w[t - l]);
      maLags.forEach((l, idx) => v += thetas[idx] * eProxy[t - l - longArOrder]);
      return v;
    });
  }
  const targetMean = meanOf(target);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < target.length; i++) { ssRes += (target[i] - oneStepPredict[i]) ** 2; ssTot += (target[i] - targetMean) ** 2; }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  const lastX = xs[n - 1];
  const avgStep = (xs[n - 1] - xs[0]) / (n - 1) || 1;
  let yHistory = ys.slice();
  const exogAt = row => exogFit ? exogFit.predict(row) : 0;

  // Chart-only one-step-fitted curve in original Y units, for every row --
  // same reasoning as fitARIMA's own oneStepFittedForChart (see its
  // comment): reintegrates each one-step w-prediction back up through the
  // REAL historical stage values (not a recursively-built fictional one),
  // then adds the real exogenous contribution for that row back on top.
  const oneStepFittedForChart = ys.map((y, i) => i);
  for (let T = 0; T < n; T++) {
    const j = T - cumLag[L] - oneStepStartInW;
    if (j >= 0 && j < oneStepPredict.length && T !== n - 1) {
      let v = oneStepPredict[j];
      for (let k = L - 1; k >= 0; k--) v = levelsStatic[k][T - cumLag[k + 1]] + v;
      oneStepFittedForChart[T] = v + exogAt(Xexog ? Xexog[T] : null);
    } else {
      oneStepFittedForChart[T] = ys[T];
    }
  }

  const predict = (x, knownPrevY, futureExogRow) => {
    if (typeof knownPrevY === 'number' && Number.isFinite(knownPrevY)) {
      if (knownPrevY !== yHistory[yHistory.length - 1]) yHistory.push(knownPrevY);
      const yHistForArima = exogFit && futureExogRow ? yHistory.map((y, i) => i === yHistory.length - 1 ? y - exogFit.predict(futureExogRow) : y) : yHistory;
      const levels = [yHistForArima];
      for (const lag of stageLags) levels.push(diffByLag(levels[levels.length - 1], lag));
      const wHist = levels[L];
      const minNeededHere = maxArLag + (maLags.length ? maxMaLag : 0) + 1;
      if (wHist.length <= minNeededHere) return yHistory[yHistory.length - 1];
      const eHist = rebuildEHistory(wHist);
      let v = oneStepWForecast(wHist, eHist);
      for (let k = L - 1; k >= 0; k--) v = levels[k][levels[k].length - stageLags[k]] + v;
      return v + exogAt(futureExogRow || lastExogRow);
    }
    if (x <= lastX) {
      if (x <= xs[0]) return oneStepFittedForChart[0];
      for (let i = 1; i < n; i++) {
        if (x <= xs[i]) {
          const span = xs[i] - xs[i - 1];
          const frac = span !== 0 ? (x - xs[i - 1]) / span : 0;
          return oneStepFittedForChart[i - 1] + frac * (oneStepFittedForChart[i] - oneStepFittedForChart[i - 1]);
        }
      }
      return oneStepFittedForChart[n - 1];
    }
    const rawSteps = (x - lastX) / avgStep;
    const stepsLo = Math.max(0, Math.floor(rawSteps));
    const stepsHi = stepsLo + 1;
    const wForecasts = forecastWSteps(w, eProxy ? eProxy.slice() : [], stepsHi);
    const stageHistories = levelsStatic.map(arr => arr.slice());
    const levelForecasts = wForecasts.map(wv => appendForecastStep(stageHistories, wv) + exogAt(futureExogRow || lastExogRow));
    const yLo = stepsLo === 0 ? ys[n - 1] : levelForecasts[stepsLo - 1];
    const yHi = levelForecasts[stepsHi - 1];
    const frac = rawSteps - stepsLo;
    return yLo + frac * (yHi - yLo);
  };

  return { type: `SARIMA(${p},${d},${q})(${P},${D},${Q})[${s}]`, p, d, q, P, D, Q, s, phis, thetas, c, beta: exogFit ? exogFit.coeffs : null, r2, predict };
}

function formatSARIMAEquation(fit) {
  const regLagCount = fit.p, seasLagCount = fit.P;
  const arTerms = fit.phis.map((phi, i) => {
    const lag = i < regLagCount ? i + 1 : (i - regLagCount + 1) * fit.s;
    return `${phi >= 0 ? '+' : '-'} ${Math.abs(phi).toFixed(4)}*w_(t-${lag})`;
  }).join(' ');
  const regMaCount = fit.q;
  const maTerms = fit.thetas.map((th, i) => {
    const lag = i < regMaCount ? i + 1 : (i - regMaCount + 1) * fit.s;
    return `${th >= 0 ? '+' : '-'} ${Math.abs(th).toFixed(4)}*e_(t-${lag})`;
  }).join(' ');
  const exogNote = fit.beta ? `, plus exogenous terms ${fit.beta.map((b, i) => `${b >= 0 ? '+' : '-'} ${Math.abs(b).toFixed(4)}*x${i + 1}`).join(' ')}` : '';
  return `w_t = ${fit.c.toFixed(4)} ${arTerms} ${maTerms}  (w = y seasonally differenced ${fit.D} time(s) at lag ${fit.s}, then differenced ${fit.d} more time(s)${exogNote})`;
}

// Auto-selects the seasonal period and a small (p,d,q)(P,D,Q) grid via the
// same internal FIT/CHECK split + walk-forward scoring discipline as
// fitARIMAAuto -- so this can act as a single CANDIDATES pool entry.
// Period detection: the strongest autocorrelation peak (lag 2..n/2) in the
// LINEARLY DETRENDED series -- a raw trend would otherwise dominate every
// lag's correlation and mask any real cycle. Returns null (no seasonal
// component tried) if nothing clears SEASON_MIN_R, rather than forcing a
// guess onto data with no real cycle.
const SEASON_MIN_R = 0.3;
function detectSeasonalPeriod(xs, ys, maxLag = Math.floor(ys.length / 2)) {
  const n = ys.length;
  if (n < 8) return null;
  let detrended;
  try { const lin = fitPolynomial(xs, ys, 1); detrended = ys.map((y, i) => y - lin.predict(xs[i])); }
  catch (e) { detrended = ys.slice(); }
  const mean = detrended.reduce((s, v) => s + v, 0) / n;
  const centered = detrended.map(v => v - mean);
  const var0 = centered.reduce((s, v) => s + v * v, 0);
  if (var0 <= 0) return null;
  let bestLag = null, bestR = SEASON_MIN_R;
  const cap = Math.min(maxLag, Math.floor(n / 2));
  for (let lag = 2; lag <= cap; lag++) {
    let cov = 0;
    for (let t = lag; t < n; t++) cov += centered[t] * centered[t - lag];
    const r = cov / var0;
    if (r > bestR) { bestR = r; bestLag = lag; }
  }
  return bestLag;
}
function fitSARIMAAuto(xs, ys, { exog = null } = {}) {
  const n = xs.length;
  if (n < 16) throw new Error('Need at least 16 data points for SARIMAX auto-selection.');
  const period = detectSeasonalPeriod(xs, ys);
  const fitLen = Math.max(10, Math.floor(n * 0.7));
  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen), checkYs = ys.slice(fitLen);
  const fitExog = exog ? exog.map(col => col.slice(0, fitLen)) : null;

  const configs = [{ p: 1, d: 1, q: 0, P: 0, D: 0, Q: 0, s: 1 }, { p: 0, d: 1, q: 1, P: 0, D: 0, Q: 0, s: 1 }];
  if (period && period * 2 + 6 <= fitLen) {
    configs.push(
      { p: 1, d: 1, q: 0, P: 1, D: 0, Q: 0, s: period },
      { p: 1, d: 0, q: 0, P: 0, D: 1, Q: 0, s: period },
      { p: 1, d: 1, q: 0, P: 0, D: 1, Q: 0, s: period },
      { p: 0, d: 1, q: 1, P: 0, D: 1, Q: 1, s: period },
    );
  }

  let chosenCfg = null;
  if (checkXs.length >= 3) {
    let bestCheckR2 = -Infinity;
    for (const cfg of configs) {
      let fit; try { fit = fitSARIMA(fitXs, fitYs, { ...cfg, exog: fitExog }); } catch (e) { continue; }
      const r2 = computeR2Sequential(fitYs[fitYs.length - 1], checkXs, checkYs, fit.predict);
      if (r2 > bestCheckR2) { bestCheckR2 = r2; chosenCfg = cfg; }
    }
  }
  if (!chosenCfg) chosenCfg = { p: 1, d: 1, q: 0, P: 0, D: 0, Q: 0, s: 1 };
  return fitSARIMA(xs, ys, { ...chosenCfg, exog });
}

// ---------------------------------------------------------------------------
// Prophet-style decomposition: y = TREND (piecewise-linear, with a handful
// of automatically-placed changepoints where the slope is allowed to bend)
// + SEASONALITY (an additive Fourier series at an auto-detected period).
// This is the same two-part decomposition idea Facebook/Meta's Prophet
// uses, written from scratch in plain JS -- NOT a port of Prophet's actual
// code (which is Python/R, fit by a full Bayesian MAP optimizer via Stan).
// The real differences from actual Prophet: changepoints are shrunk toward
// zero slope-change with the same OLS-shrinkage trick as this file's AR
// models (Prophet uses a proper Laplace prior fit by Stan), and there's no
// holiday-effects term. Confirmed with the user before building this that a
// from-scratch JS equivalent (rather than running real Python Prophet as a
// separate process) is what's wanted here.
//
// Trend basis: intercept + slope*x + a "hinge" term max(0, x - cp_j) for
// each changepoint cp_j -- a continuous piecewise-linear curve whose slope
// can change at each cp_j, fit in one shot by ordinary least squares
// (hinges are just extra columns, no different from any other regressor).
// Changepoints are placed evenly over the first 80% of the X range (same
// `changepoint_range` default Prophet itself uses -- bending the trend
// right at the last few points is usually overfitting noise, not a real
// change in direction).
function fitProphetStyle(xs, ys, { nChangepoints = 5, period = null, fourierOrder = 3, shrinkage = 2 } = {}) {
  const n = xs.length;
  if (n < nChangepoints + 4) throw new Error(`Need more data for a Prophet-style fit with ${nChangepoints} changepoints.`);
  // Nyquist cap: with evenly-spaced integer rows, a harmonic k >= period/2
  // doesn't add real information — sin(2*pi*k*x/period) is identically 0
  // for every integer x whenever period divides 2k, and at exactly that
  // boundary cos(2*pi*k*x/period) duplicates a LOWER harmonic's cos column
  // exactly (e.g. period=2: cos(3*pi*x) === cos(pi*x) for every integer
  // x) — a collinearity the variance filter below can't catch on its own,
  // since a duplicate column isn't zero-variance, just redundant.
  if (period) fourierOrder = Math.max(1, Math.min(fourierOrder, Math.floor(period / 2)));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const range = xMax - xMin || 1;
  const changepoints = Array.from({ length: nChangepoints }, (_, i) => xMin + range * 0.8 * ((i + 1) / (nChangepoints + 1)));

  const basisRow = x => {
    const row = [x, ...changepoints.map(cp => Math.max(0, x - cp))];
    if (period) for (let k = 1; k <= fourierOrder; k++) { row.push(Math.sin(2 * Math.PI * k * x / period)); row.push(Math.cos(2 * Math.PI * k * x / period)); }
    return row;
  };
  const Xfull = xs.map(basisRow);
  // With evenly-spaced integer rows, some Fourier harmonic at a small
  // period can be IDENTICALLY zero for every row (e.g. sin(2*pi*k*x/period)
  // when period divides 2k exactly) -- a degenerate, all-constant column
  // that would otherwise hand fitMultiLinear a singular matrix and crash
  // the solver. Drop any column with ~0 variance before fitting, then
  // reinsert a 0 coefficient for it afterward so every other index below
  // (cpDeltas, seasonalCoeffs) still lines up the same way regardless of
  // which columns got dropped.
  const nCols = Xfull[0].length;
  const keepCol = new Array(nCols).fill(true);
  for (let c = 0; c < nCols; c++) {
    const vals = Xfull.map(row => row[c]);
    const colMean = meanOfArr(vals);
    const variance = vals.reduce((s, v) => s + (v - colMean) ** 2, 0) / vals.length;
    if (variance < 1e-12) keepCol[c] = false;
  }
  const keptIdx = []; for (let c = 0; c < nCols; c++) if (keepCol[c]) keptIdx.push(c);
  const X = Xfull.map(row => keptIdx.map(c => row[c]));
  const fitKept = fitMultiLinear(X, ys);
  const fullCoeffs = new Array(nCols).fill(0);
  keptIdx.forEach((c, i) => { fullCoeffs[c] = fitKept.coeffs[i]; });
  const fit = { intercept: fitKept.intercept, coeffs: fullCoeffs };

  // Shrink the hinge (changepoint-slope-change) coefficients toward 0, the
  // same "pairs/(pairs+shrinkage)" prior this file's AR models use for the
  // same reason: a handful of changepoints fit by plain OLS on a short
  // series will happily bend the trend to chase noise unless nudged back
  // toward "no change here" by default. The overall slope and seasonal
  // terms are NOT shrunk -- only the changepoint deltas are a "maybe
  // nothing happened here" parameter in the way Prophet's own prior treats
  // them.
  const shrink = n / (n + shrinkage);
  const slope = fit.coeffs[0];
  const cpDeltas = fit.coeffs.slice(1, 1 + nChangepoints).map(v => v * shrink);
  const seasonalCoeffs = fit.coeffs.slice(1 + nChangepoints);
  const intercept = fit.intercept;

  const predict = x => {
    let v = intercept + slope * x;
    changepoints.forEach((cp, i) => { v += cpDeltas[i] * Math.max(0, x - cp); });
    if (period) for (let k = 1; k <= fourierOrder; k++) {
      v += seasonalCoeffs[2 * (k - 1)] * Math.sin(2 * Math.PI * k * x / period);
      v += seasonalCoeffs[2 * (k - 1) + 1] * Math.cos(2 * Math.PI * k * x / period);
    }
    return v;
  };
  let ssRes = 0, ssTot = 0;
  const yMean = meanOfArr(ys);
  for (let i = 0; i < n; i++) { ssRes += (ys[i] - predict(xs[i])) ** 2; ssTot += (ys[i] - yMean) ** 2; }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { type: 'Prophet-style', changepoints, slope, cpDeltas, intercept, period, fourierOrder, seasonalCoeffs, r2, predict };
}
function meanOfArr(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

function formatProphetEquation(fit) {
  const cpTerms = fit.changepoints.map((cp, i) => `${fit.cpDeltas[i] >= 0 ? '+' : '-'} ${Math.abs(fit.cpDeltas[i]).toFixed(4)}*max(0, x-${cp.toFixed(2)})`).join(' ');
  const seasonTerms = fit.period ? ' + seasonal terms at period ' + fit.period.toFixed(2) : ' (no seasonal component detected)';
  return `${fit.intercept.toFixed(4)} + ${fit.slope.toFixed(4)}*x ${cpTerms}${seasonTerms}`;
}

// Auto-picks the number of changepoints and the seasonal period/order via
// the same internal FIT/CHECK split as fitARIMAAuto -- except, unlike the
// AR-family models, a Prophet-style prediction is a direct function of x
// (no recursive one-step-ahead state), so plain static R² on the held-out
// CHECK rows is the right score here, not computeR2Sequential's
// walk-forward re-anchoring.
const PROPHET_CHANGEPOINT_GRID = [0, 2, 4, 6, 10];
function fitProphetStyleAuto(xs, ys) {
  const n = xs.length;
  if (n < 10) throw new Error('Need at least 10 data points for a Prophet-style auto fit.');
  const period = detectSeasonalPeriod(xs, ys);
  const fitLen = Math.max(8, Math.floor(n * 0.7));
  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen), checkYs = ys.slice(fitLen);

  let chosenCp = null, bestCheckR2 = -Infinity;
  const cpGrid = PROPHET_CHANGEPOINT_GRID.filter(cp => cp + 4 <= fitLen);
  if (checkXs.length >= 3) {
    for (const nChangepoints of cpGrid) {
      let fit; try { fit = fitProphetStyle(fitXs, fitYs, { nChangepoints, period }); } catch (e) { continue; }
      let ssRes = 0, ssTot = 0; const checkMean = meanOfArr(checkYs);
      for (let i = 0; i < checkXs.length; i++) { ssRes += (checkYs[i] - fit.predict(checkXs[i])) ** 2; ssTot += (checkYs[i] - checkMean) ** 2; }
      const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
      if (r2 > bestCheckR2) { bestCheckR2 = r2; chosenCp = nChangepoints; }
    }
  }
  if (chosenCp === null) chosenCp = Math.min(4, Math.max(0, Math.floor(n / 10)));
  return fitProphetStyle(xs, ys, { nChangepoints: chosenCp, period });
}

// Confidence intervals for a fitted model's parameters, via bootstrap
// resampling — works the same way for every fit function in this file,
// whether it has a closed-form solution (polynomial, linear), is fit
// iteratively (S-curve, sinusoidal), or is a resistant/robust method
// (RANSAC, Theil-Sen, Huber...). Rather than hand-deriving a separate
// standard-error formula per model type, refit the SAME model on many
// resamples of the data (same size n, drawn with replacement, seeded so
// results are reproducible) and see how much each parameter moves across
// resamples. The middle 95% of that spread (2.5th to 97.5th percentile) is
// the reported interval — the standard "percentile bootstrap" CI.
//
// fitFn must take (xs, ys) and return a fit object (extra fixed options —
// e.g. a fixed polynomial degree — should already be closed over by the
// caller, e.g. `(rxs, rys) => fitPolynomial(rxs, rys, degree)`).
function percentileCI(vals) {
  const sorted = [...vals].sort((a, b) => a - b);
  const loIdx = Math.max(0, Math.floor(0.025 * (sorted.length - 1)));
  const hiIdx = Math.min(sorted.length - 1, Math.ceil(0.975 * (sorted.length - 1)));
  return { lower: sorted[loIdx], upper: sorted[hiIdx] };
}

// Only ever called once per run, for the single model actually being
// reported to the user (never for every candidate a selection step
// considered) — but that one call still means refitting the SAME model
// `iterations` times, so it's worth keeping modest. 150 is plenty for a
// stable 2.5th/97.5th percentile; callers fitting an expensive iterative
// model (gradient descent, RANSAC, Theil-Sen) should also pass a lighter
// fitFn (e.g. fewer gradient steps) — see the /regression page's ciBlock()
// call sites for examples. The final displayed fit is never affected by
// this — only the resamples used to build the CI are cheaper.
function bootstrapParameterCI(fitFn, xs, ys, { iterations = 150, seed = RNG_SEED } = {}) {
  const n = xs.length;
  // Too few points and the resampled fits are just noise reshuffled — the
  // "interval" would be meaningless rather than merely wide.
  if (n < 5) return null;

  const random = createSeededRandom(seed);
  const scalarSamples = {};
  const coeffSamples = [];
  const SKIP_KEYS = new Set(['type', 'r2', 'logLoss', 'inlierCount', 'trimmedCount', 'totalCount', 'coeffs', 'predict', 'degree']);
  let successes = 0;

  for (let i = 0; i < iterations; i++) {
    const rxs = new Array(n), rys = new Array(n);
    for (let j = 0; j < n; j++) {
      const idx = Math.floor(random() * n);
      rxs[j] = xs[idx]; rys[j] = ys[idx];
    }
    let fit;
    // A resample can easily be degenerate for a given model (e.g. every
    // resampled X identical, or every resampled Y on one side of a
    // threshold a robust method needs both sides of) even when the real
    // data fits fine — skip it rather than letting one bad resample crash
    // the whole CI.
    try { fit = fitFn(rxs, rys); } catch (e) { continue; }
    successes++;
    for (const key of Object.keys(fit)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = fit[key];
      if (typeof val === 'number' && isFinite(val)) {
        (scalarSamples[key] = scalarSamples[key] || []).push(val);
      }
    }
    if (Array.isArray(fit.coeffs)) {
      fit.coeffs.forEach((c, idx) => {
        if (typeof c === 'number' && isFinite(c)) (coeffSamples[idx] = coeffSamples[idx] || []).push(c);
      });
    }
  }

  // Need enough surviving resamples for a 2.5th/97.5th percentile to mean
  // anything — if most resamples failed, the model type is a poor fit for
  // this data's shape and no CI should be reported at all.
  const minNeeded = Math.max(20, iterations * 0.3);
  if (successes < minNeeded) return null;

  const params = {};
  for (const key of Object.keys(scalarSamples)) {
    if (scalarSamples[key].length < minNeeded) continue;
    params[key] = percentileCI(scalarSamples[key]);
  }
  const coeffs = coeffSamples.length
    ? coeffSamples.map(vals => (vals && vals.length >= minNeeded) ? percentileCI(vals) : null)
    : null;

  return { params, coeffs, resamples: successes, requested: iterations };
}

// Moving-block bootstrap: like bootstrapParameterCI, but instead of
// resampling single (x,y) points with replacement (which destroys any
// serial structure — fine for IID data, wrong when residuals are
// autocorrelated), it resamples contiguous BLOCKS of `blockLength`
// consecutive points with replacement and concatenates them (wrapping
// around the end of the data, "circular" block bootstrap) until it has n
// points again. Blocks preserve whatever lag-1 (and shorter) dependence
// exists inside each block, so the resample-to-resample spread reflects the
// real uncertainty of autocorrelated data instead of understating it the
// way a plain IID bootstrap would. blockLength defaults to round(n/3) —
// found by a direct block-length SWEEP (ci_gate_coverage_test.js), not a
// textbook rule of thumb. Two textbook defaults were tried and rejected on
// this project's own data first: n^(1/3) (e.g. length 4 on a 60-row
// dataset) and log(n) both badly under-cover — this corpus's real flagged
// datasets run as high as phi=0.98 lag-1 autocorrelation (see the comment
// near line ~1495), far more persistence than either formula's short
// blocks can capture. sqrt(n) was adopted as an interim fix, but the swept
// comparison found n/3 beats it in EVERY tested scenario (autocorrelation
// at phi=0.3/0.5/0.7/0.9 AND heteroscedasticity, at n=30/60/120) — e.g. at
// phi=0.9, n=120: sqrt(n)'s ~71-75% coverage vs. n/3's ~86-91%. n/4 is a
// close second everywhere; n/2 and anything shorter than n/4 is worse than
// n/3 on both sides. Still falls short of the 95% target at the most
// severe end (phi=0.9), but meaningfully closer than any prior default —
// see that test file for the full swept numbers. chooseParameterCI below
// now supplements this with an AR(1) analytic floor (ar1InflateExactCI) AND
// an empirical coverage calibration table (ar1CoverageCorrectionFactor) on
// top of that. Held-out validation (n/phi values off both calibration
// grids): at phi=0.9 coverage goes from the raw block bootstrap's ~65-90%
// (worse at small n) up to ~76-94% with both corrections applied -- still
// short of 95% at small n, but meaningfully closer at every size tested;
// see ar1_analytic_floor_test.js and ar1CoverageCorrectionFactor's comment.
function movingBlockBootstrapParameterCI(fitFn, xs, ys, { iterations = 150, seed = RNG_SEED, blockLength = null } = {}) {
  const n = xs.length;
  if (n < 5) return null;
  const L = Math.max(2, Math.min(n, blockLength || Math.round(n / 3)));
  const numBlocks = Math.ceil(n / L);

  const random = createSeededRandom(seed);
  const scalarSamples = {};
  const coeffSamples = [];
  const SKIP_KEYS = new Set(['type', 'r2', 'logLoss', 'inlierCount', 'trimmedCount', 'totalCount', 'coeffs', 'predict', 'degree']);
  let successes = 0;

  for (let i = 0; i < iterations; i++) {
    const rxs = [], rys = [];
    for (let b = 0; b < numBlocks; b++) {
      const start = Math.floor(random() * n); // circular: wraps past the end
      for (let k = 0; k < L && rxs.length < n; k++) {
        const idx = (start + k) % n;
        rxs.push(xs[idx]); rys.push(ys[idx]);
      }
    }
    // genericWeightedRefit (see its own comment) reweights this resample's
    // fit against its own residuals so one skewed-outlier-contaminated
    // block can't pull this resample's intercept as far off as a single
    // unweighted fit would — see that function's comment for the
    // before/after validation numbers.
    const fit = genericWeightedRefit(fitFn, rxs, rys);
    if (!fit) continue;
    successes++;
    for (const key of Object.keys(fit)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = fit[key];
      if (typeof val === 'number' && isFinite(val)) {
        (scalarSamples[key] = scalarSamples[key] || []).push(val);
      }
    }
    if (Array.isArray(fit.coeffs)) {
      fit.coeffs.forEach((c, idx) => {
        if (typeof c === 'number' && isFinite(c)) (coeffSamples[idx] = coeffSamples[idx] || []).push(c);
      });
    }
  }

  const minNeeded = Math.max(20, iterations * 0.3);
  if (successes < minNeeded) return null;

  const params = {};
  for (const key of Object.keys(scalarSamples)) {
    if (scalarSamples[key].length < minNeeded) continue;
    params[key] = percentileCI(scalarSamples[key]);
  }
  const coeffs = coeffSamples.length
    ? coeffSamples.map(vals => (vals && vals.length >= minNeeded) ? percentileCI(vals) : null)
    : null;

  return { params, coeffs, resamples: successes, requested: iterations, method: 'block-bootstrap', blockLength: L };
}

// Hat-matrix diagonal (leverage) for a plain polynomial design [1, x, x^2,
// ..., x^degree] — only ever used by wildBootstrapParameterCI's leverage
// correction below, and only when the fit being bootstrapped actually has
// that design (see hasLinearDesign there). Reuses solveLinearSystem via
// invertMatrix exactly like olsParameterCI does, rather than a separate
// matrix routine.
function polynomialHatDiagonal(xs, degree) {
  const n = xs.length, terms = degree + 1;
  const X = xs.map(x => {
    const row = [1];
    for (let p = 1; p <= degree; p++) row.push(row[p - 1] * x);
    return row;
  });
  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < terms; r++) for (let c = 0; c < terms; c++) A[r][c] += X[i][r] * X[i][c];
  }
  const Ainv = invertMatrix(A);
  return X.map(row => {
    let h = 0;
    for (let i = 0; i < terms; i++) for (let j = 0; j < terms; j++) h += row[i] * Ainv[i][j] * row[j];
    return h;
  });
}

// Builds a CI centered on the ORIGINAL fit's own estimate (not the resample
// mean, which can drift from it under heteroscedasticity/small samples) and
// scaled by Student-t-critical * resample standard deviation, instead of
// reading the resample distribution's own 2.5th/97.5th percentiles
// (percentileCI) directly. Falls back to the plain percentile method
// whenever there isn't a usable degrees-of-freedom count (tCrit null) —
// see wildBootstrapParameterCI's comment for why this matters there.
function tScaledOrPercentileCI(vals, center, tCrit) {
  if (tCrit === null || !isFinite(center)) return percentileCI(vals);
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, vals.length - 1));
  return { lower: center - tCrit * sd, upper: center + tCrit * sd };
}

// Wild bootstrap: for heteroscedastic (not autocorrelated) residuals. A
// plain IID bootstrap resamples (x,y) PAIRS, which is valid but throws away
// the fitted shape and relies on enough points near every x to get the
// local spread right — noisy when the data isn't dense everywhere. The wild
// bootstrap instead keeps the original xs fixed and the original FIT fixed,
// and only resamples the ERROR: each resampled y*_i = predict(x_i) +
// residual_i * w_i, where w_i is an independent Rademacher weight (+1 or -1
// with equal probability, seeded). Multiplying by +-1 keeps residual_i's
// own magnitude at that specific x_i (so a point with a big residual stays
// big, a point with a small residual stays small) — exactly the structure
// that matters under heteroscedasticity, where the error SIZE itself
// depends on x. This is the standard residual-based wild bootstrap
// (Rademacher weights, aka the Mammen two-point alternative's simpler
// cousin) used for inference under an unknown form of heteroscedasticity.
//
// Two corrections below were added after a direct coverage sweep
// (wild_bootstrap_leverage_t_test.js) found this function under-covering
// even on CLEAN heteroscedastic data with no outliers at all — ~83-94% vs.
// the 95% target, worse at small n. That's a baseline CALIBRATION problem,
// not an outlier-sensitivity one — a separate attempt to fix outlier
// sensitivity here by winsorizing/capping residuals was tried and REJECTED
// (it only narrows an already-too-narrow interval further, making coverage
// *worse*, not better — see that investigation's own notes).
//
// 1. Leverage adjustment: a fitted model's residual at point i is
//    systematically smaller than that point's TRUE error, by a factor of
//    sqrt(1-h_i), where h_i is the point's leverage (hat-matrix diagonal) —
//    the fit was partly pulled toward that very point, so its own residual
//    understates how much it could plausibly have varied. Reusing raw
//    residuals for sign-flip resampling therefore understates the true
//    error spread. Scaling residual_i by 1/sqrt(1-h_i) before sign-flipping
//    (the standard "modified" wild bootstrap correction) fixes this. Only
//    computable when the fit actually has a plain polynomial design (a
//    numeric `degree` field alongside `coeffs`, i.e. runRegression /
//    fitPolynomial) — other fit types (exponential, logistic, AR/ARIMA,
//    etc.) skip this step and get correction #2 alone, the same
//    partial-applicability pattern as the AR(1) analytic floor above (exact
//    CI only for linear/polynomial fits, bootstrap alone otherwise).
// 2. t-scaled interval instead of raw percentile: a plain percentile
//    bootstrap CI is known to run too narrow at small sample sizes (the
//    resample distribution's own tails underestimate the true sampling
//    distribution's). Centering on the ORIGINAL fit's estimate and scaling
//    by Student-t-critical * resample-sd (same df = n - p convention the
//    exact OLS CI uses) instead of reading off percentiles closes most of
//    the remaining gap, and applies regardless of fit type (df only needs a
//    parameter count, not a design matrix).
//
// Combined validated result (wild_bootstrap_leverage_t_test.js, 400-2000
// trials/condition, heteroscedastic synthetic linear data): coverage at
// n=15 goes from ~83-88% (uncorrected) to ~91-97% across 0/5/10/20%
// contamination; at n=60 both are already close to nominal (~93-96%
// either way). Width cost is modest and shrinks with n (~25% more at n=15,
// ~5% more at n=60) — and critically, coverage under contamination is as
// good as or better than the uncorrected version at every rate tested, so
// this doesn't reintroduce the outlier-sensitivity problem the rejected
// capping attempt was trying (and failing) to fix.
function wildBootstrapParameterCI(fitFn, xs, ys, predict, { iterations = 150, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 5) return null;
  const residuals = xs.map((x, i) => ys[i] - predict(x));

  const originalFit = fitFn(xs, ys);
  const SKIP_KEYS = new Set(['type', 'r2', 'logLoss', 'inlierCount', 'trimmedCount', 'totalCount', 'coeffs', 'predict', 'degree']);
  const scalarParamCount = Object.keys(originalFit).filter(k => SKIP_KEYS.has(k) ? false : (typeof originalFit[k] === 'number' && isFinite(originalFit[k]))).length;
  const p = Array.isArray(originalFit.coeffs) ? originalFit.coeffs.length : scalarParamCount;
  const df = n - p;
  const tCrit = (p > 0 && df >= 1) ? studentTInverseCDF(0.975, df) : null;

  const hasLinearDesign = Array.isArray(originalFit.coeffs) && typeof originalFit.degree === 'number';
  const scaledResiduals = hasLinearDesign
    ? (() => {
        const h = polynomialHatDiagonal(xs, originalFit.degree);
        return residuals.map((r, i) => r / Math.sqrt(Math.max(0.01, 1 - h[i])));
      })()
    : residuals;

  const random = createSeededRandom(seed);
  const scalarSamples = {};
  const coeffSamples = [];
  let successes = 0;

  for (let i = 0; i < iterations; i++) {
    const rys = new Array(n);
    for (let j = 0; j < n; j++) {
      const w = random() < 0.5 ? -1 : 1;
      rys[j] = predict(xs[j]) + scaledResiduals[j] * w;
    }
    let fit;
    try { fit = fitFn(xs, rys); } catch (e) { continue; }
    successes++;
    for (const key of Object.keys(fit)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = fit[key];
      if (typeof val === 'number' && isFinite(val)) {
        (scalarSamples[key] = scalarSamples[key] || []).push(val);
      }
    }
    if (Array.isArray(fit.coeffs)) {
      fit.coeffs.forEach((c, idx) => {
        if (typeof c === 'number' && isFinite(c)) (coeffSamples[idx] = coeffSamples[idx] || []).push(c);
      });
    }
  }

  const minNeeded = Math.max(20, iterations * 0.3);
  if (successes < minNeeded) return null;

  const params = {};
  for (const key of Object.keys(scalarSamples)) {
    if (scalarSamples[key].length < minNeeded) continue;
    params[key] = tScaledOrPercentileCI(scalarSamples[key], originalFit[key], tCrit);
  }
  const coeffs = coeffSamples.length
    ? coeffSamples.map((vals, idx) => (vals && vals.length >= minNeeded) ? tScaledOrPercentileCI(vals, originalFit.coeffs[idx], tCrit) : null)
    : null;

  return { params, coeffs, resamples: successes, requested: iterations, method: 'wild-bootstrap', df };
}

// The CI-method gate: picks which of the three CI methods above actually
// applies to THIS fit's residuals, instead of always using the same one
// regardless of whether its assumptions hold. Runs Durbin-Watson and
// White's test on the fit's own residuals (the same two diagnostics shown
// to the user elsewhere on the page) and routes accordingly:
//   - autocorrelated (DW p < 0.1)      -> moving-block bootstrap
//   - heteroscedastic only (White p < 0.1) -> wild bootstrap
//   - neither flagged                  -> whatever exactOrPlainCI gives
//     (the classical exact OLS CI when the fit has one, else the plain IID
//     bootstrap) — unchanged behavior for well-behaved residuals.
// Autocorrelation is checked first and wins ties (a fit can be flagged by
// both): a dependence-structure violation breaks the plain wild bootstrap's
// independence assumption too, while the block bootstrap still tolerates
// some heteroscedasticity inside each block.
// Direct sample estimate of residual lag-1 autocorrelation (phi in an
// AR(1) model e_t = phi*e_{t-1} + eta_t), used by the AR(1) analytic CI
// floor below. Standard ratio estimator sum(e_t*e_{t-1}) / sum(e_t^2) --
// equivalent at large n to the r1 = 1 - DW/2 approximation durbinWatsonTest
// already uses, computed directly here instead of backed out from DW so it
// isn't tied to DW's own large-n approximation error.
function estimateLag1Autocorrelation(residuals) {
  const n = residuals.length;
  if (n < 3) return 0;
  let num = 0, denom = 0;
  for (let t = 1; t < n; t++) num += residuals[t] * residuals[t - 1];
  for (let t = 0; t < n; t++) denom += residuals[t] * residuals[t];
  if (denom === 0) return 0;
  return num / denom;
}

// AR(1) analytic CI floor: the moving-block bootstrap above is the primary
// fix for autocorrelated residuals, but a direct coverage sweep
// (ci_gate_coverage_test.js) found it still under-covers at the severe end
// (phi=0.9: ~86-91% vs. the 95% target -- see the comment above
// movingBlockBootstrapParameterCI). This supplies an independent, classical
// check: under an AR(1) error model the variance of the OLS coefficient
// estimator inflates by roughly (1+phi)/(1-phi) relative to the IID case
// (the same "effective sample size" correction n_eff = n*(1-phi)/(1+phi)
// used for correlated time series elsewhere, e.g. Quenouille 1952), so the
// classical exact-OLS standard error, inflated by the sqrt of that factor,
// gives a second, analytically-motivated estimate of how wide the interval
// should be. Only ever WIDENS -- negative phi would imply deflation, which
// isn't the problem being fixed here, so the inflation factor is floored at
// 1 -- and phi is capped at 0.97 so a near-unit-root residual series can't
// blow the interval up toward +-Infinity. Only applicable when an exact OLS
// CI exists (linear/polynomial fits with a design matrix); nonlinear models
// fall back to the block bootstrap alone, same as before.
function ar1InflateExactCI(exactCI, phi) {
  if (!exactCI || !exactCI.exact || !Array.isArray(exactCI.coeffs)) return null;
  const phiClamped = Math.max(-0.97, Math.min(0.97, phi));
  const inflation = Math.max(1, (1 + phiClamped) / (1 - phiClamped));
  const factor = Math.sqrt(inflation);
  const coeffs = exactCI.coeffs.map(c => {
    if (!c) return c;
    const se = c.se * factor;
    return { estimate: c.estimate, se, lower: c.estimate - exactCI.tCritical * se, upper: c.estimate + exactCI.tCritical * se };
  });
  return { ...exactCI, coeffs, phi: phiClamped, inflation };
}

// Widens `ci` (the moving-block bootstrap result) so no parameter's
// interval is narrower than the matching one in `floorCi` (the AR(1)
// analytic estimate above) -- a union of the two intervals, not an average,
// so the result is never LESS conservative than either method alone. The
// empirical block-bootstrap interval is kept as-is whenever it's already at
// least this wide, since it makes no normality assumption and reflects this
// specific dataset's own resampled spread.
function widenParameterCIWithFloor(ci, floorCi) {
  if (!ci || !floorCi) return ci;
  const widenPair = (a, b) => {
    if (!a) return b || a;
    if (!b) return a;
    return { ...a, lower: Math.min(a.lower, b.lower), upper: Math.max(a.upper, b.upper) };
  };
  const coeffs = Array.isArray(ci.coeffs)
    ? ci.coeffs.map((c, i) => widenPair(c, floorCi.coeffs && floorCi.coeffs[i]))
    : ci.coeffs;
  const params = {};
  for (const key of Object.keys(ci.params || {})) {
    params[key] = widenPair(ci.params[key], floorCi.params && floorCi.params[key]);
  }
  return { ...ci, coeffs, params, ar1Floor: { phi: floorCi.phi, inflation: floorCi.inflation } };
}

// Empirical (phi, n) -> extra correction-factor table for the COMBINED
// (block-bootstrap union AR(1)-analytic) CI above, built the same way this
// project calibrated SW_SIZE_ADJUST for Shapiro-Wilk: simulate AR(1) data
// with a KNOWN true coefficient at a grid of (n, phi), measure how far the
// combined CI's own center actually lands from the truth relative to its
// own half-width, and take the 95th-percentile ratio as the factor that
// would have been needed to hit 95% coverage on that calibration run (see
// ar1_analytic_floor_test.js). Held-out n/phi values NOT on this grid
// (n=30/60/120/240, phi=0.1/0.3/0.5/0.7/0.9 vs. the grid's 20/40/80/160/320
// and 0/0.2/0.4/0.6/0.8/0.95) were checked separately before this table was
// wired in: it closes several more points of the remaining gap at the
// severe end (phi=0.9: +4 to +10 points of coverage across all four
// held-out n) while leaving the already-adequate mild/moderate cases
// essentially untouched. Floored at 1 in ar1CoverageCorrectionFactor below
// -- same "only ever widen" rule as the rest of this gate.
const AR1_COVERAGE_TABLE = {
  n: [20, 40, 80, 160, 320],
  phi: [0, 0.2, 0.4, 0.6, 0.8, 0.95],
  factor: [
    [0.957, 0.889, 1.037, 1.205, 1.671, 2.866],
    [0.891, 0.973, 0.975, 1.014, 1.385, 2.474],
    [0.874, 0.927, 0.959, 0.969, 1.281, 1.998],
    [0.928, 0.925, 0.941, 0.971, 1.052, 1.432],
    [0.892, 0.882, 0.917, 0.889, 0.894, 1.141],
  ],
};
// Bilinear interpolation over log(n) x phi -- log(n) because the table's own
// row-to-row shrinkage (e.g. phi=0.95: 2.87 -> 1.14 from n=20 to n=320) is
// much closer to linear on a log scale than a linear one.
function ar1CoverageCorrectionFactor(phi, n) {
  const { n: nGrid, phi: phiGrid, factor } = AR1_COVERAGE_TABLE;
  const nClamped = Math.max(nGrid[0], Math.min(nGrid[nGrid.length - 1], n));
  const phiClamped = Math.max(phiGrid[0], Math.min(phiGrid[phiGrid.length - 1], Math.abs(phi)));
  const logN = Math.log(nClamped);
  let i0 = 0;
  while (i0 < nGrid.length - 2 && Math.log(nGrid[i0 + 1]) < logN) i0++;
  const i1 = i0 + 1;
  const logN0 = Math.log(nGrid[i0]), logN1 = Math.log(nGrid[i1]);
  const tN = logN1 > logN0 ? (logN - logN0) / (logN1 - logN0) : 0;
  let j0 = 0;
  while (j0 < phiGrid.length - 2 && phiGrid[j0 + 1] < phiClamped) j0++;
  const j1 = j0 + 1;
  const phi0 = phiGrid[j0], phi1 = phiGrid[j1];
  const tPhi = phi1 > phi0 ? (phiClamped - phi0) / (phi1 - phi0) : 0;
  const f00 = factor[i0][j0], f01 = factor[i0][j1], f10 = factor[i1][j0], f11 = factor[i1][j1];
  const f0 = f00 + (f01 - f00) * tPhi;
  const f1 = f10 + (f11 - f10) * tPhi;
  return Math.max(1, f0 + (f1 - f0) * tN);
}

// Widens every coeff/param in `ci` around ITS OWN center (not the original
// point estimate) by `factor` -- matches exactly how ar1CoverageCorrectionFactor
// was calibrated, against the combined CI's own midpoint and half-width.
function inflateParameterCIAroundCenter(ci, factor) {
  if (!ci || factor <= 1) return ci;
  const scale = (c) => {
    if (!c) return c;
    const center = (c.lower + c.upper) / 2;
    const halfWidth = (c.upper - c.lower) / 2 * factor;
    return { ...c, lower: center - halfWidth, upper: center + halfWidth };
  };
  const coeffs = Array.isArray(ci.coeffs) ? ci.coeffs.map(scale) : ci.coeffs;
  const params = {};
  for (const key of Object.keys(ci.params || {})) params[key] = scale(ci.params[key]);
  return { ...ci, coeffs, params };
}

function chooseParameterCI(fitFn, xs, ys, predict, exactOrPlainCI, { iterations = 150, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 8) return exactOrPlainCI; // too little data for DW/White to mean anything
  const residuals = xs.map((x, i) => ys[i] - predict(x));
  const dw = durbinWatsonTest(residuals);
  const wt = whiteTest(xs, residuals);
  if (dw && dw.pValue < 0.1) {
    const ci = movingBlockBootstrapParameterCI(fitFn, xs, ys, { iterations, seed });
    if (ci) {
      const phi = estimateLag1Autocorrelation(residuals);
      const analyticFloor = ar1InflateExactCI(exactOrPlainCI, phi);
      const widened = analyticFloor ? widenParameterCIWithFloor(ci, analyticFloor) : ci;
      const correctionFactor = ar1CoverageCorrectionFactor(phi, n);
      const corrected = inflateParameterCIAroundCenter(widened, correctionFactor);
      const ar1Floor = widened.ar1Floor ? { ...widened.ar1Floor, correctionFactor } : undefined;
      return { ...corrected, ar1Floor, gateReason: 'autocorrelated residuals (Durbin-Watson)' };
    }
  } else if (wt && wt.pValue < 0.1) {
    const ci = wildBootstrapParameterCI(fitFn, xs, ys, predict, { iterations, seed });
    if (ci) return { ...ci, gateReason: 'heteroscedastic residuals (White\'s test)' };
  }
  return exactOrPlainCI;
}

// Plain-text rendering of a bootstrapParameterCI() result, for the /regression
// page and any CLI output. Returns '' (not a blank line) when ci is null, so
// callers can always just append it without an extra conditional.
function formatParameterCIs(ci) {
  if (!ci) return '';
  const header = ci.exact
    ? `  95% confidence intervals (exact, Student's t with ${ci.df} degrees of freedom):`
    : ci.method === 'block-bootstrap'
    ? `  95% confidence intervals (moving-block bootstrap, block length ${ci.blockLength}, ${ci.resamples}/${ci.requested} resamples used):`
    : ci.method === 'wild-bootstrap'
    ? `  95% confidence intervals (wild bootstrap, ${ci.resamples}/${ci.requested} resamples used):`
    : `  95% confidence intervals (bootstrap resampling, ${ci.resamples}/${ci.requested} resamples used):`;
  const lines = [header];
  if (ci.gateReason) lines.push(`    (switched from the usual method: ${ci.gateReason})`);
  if (ci.ar1Floor) {
    const extra = ci.ar1Floor.correctionFactor > 1.001 ? `, then widened another ${ci.ar1Floor.correctionFactor.toFixed(2)}x by a coverage calibration table` : '';
    lines.push(`    (widened to match an AR(1) analytic estimate, phi~${ci.ar1Floor.phi.toFixed(2)}${extra})`);
  }
  for (const key of Object.keys(ci.params || {})) {
    const { lower, upper } = ci.params[key];
    lines.push(`    ${key}: ${lower.toFixed(4)} to ${upper.toFixed(4)}`);
  }
  if (ci.coeffs) {
    ci.coeffs.forEach((c, idx) => {
      if (!c) return;
      // t-test p-value only exists on the exact (Student-t) path above —
      // a bootstrap CI has no classical standard error/t-ratio to build
      // one from, so this is silently skipped there rather than showing a
      // number built a different, inconsistent way.
      const sigTest = c.pValue !== undefined
        ? `   t=${c.t.toFixed(3)}, p=${c.pValue.toFixed(4)}${c.pValue < 0.05 ? ' *' : ''}`
        : '';
      lines.push(`    coeffs[${idx}]: ${c.lower.toFixed(4)} to ${c.upper.toFixed(4)}${sigTest}`);
    });
  }
  return '\n\n' + lines.join('\n');
}

// Plain-text rendering of overallFTest()'s result, for the same callers as
// formatParameterCIs. H0 here is "none of the predictors help at all" —
// different question from any single coefficient's own t-test above (a
// model can fail this while still having one "significant" coefficient, or
// pass it with none, when predictors overlap with each other).
function formatFTest(ft) {
  if (!ft) return '';
  const sig = ft.pValue < 0.05 ? ' <-- the model as a whole explains real variation (not just noise)' : ' (not distinguishable from "none of the predictors help")';
  return `\n  Overall F-test (H0: no predictor helps at all): F(${ft.df1},${ft.df2})=${ft.F.toFixed(3)}, p=${ft.pValue.toFixed(4)}${sig}`;
}

// --- Two-sample difference-in-means test (Boxplot view's column-vs-column
// comparison). Welch's t-test (unequal-variance t-test) is the default: it
// drops the classic Student's t-test's "both groups have the same variance"
// assumption, but like any t-test it still leans on the Central Limit
// Theorem to make its p-value trustworthy when the underlying data isn't
// actually normal -- fine once n is reasonably large per group, shakier on
// small, skewed, or heavy-tailed samples. H0: the two groups have the same
// population mean.
function welchTTest(a, b) {
  const nA = a.length, nB = b.length;
  if (nA < 2 || nB < 2) return null;
  const meanA = a.reduce((s, v) => s + v, 0) / nA;
  const meanB = b.reduce((s, v) => s + v, 0) / nB;
  const varA = a.reduce((s, v) => s + (v - meanA) ** 2, 0) / (nA - 1);
  const varB = b.reduce((s, v) => s + (v - meanB) ** 2, 0) / (nB - 1);
  const seA2 = varA / nA, seB2 = varB / nB;
  const se = Math.sqrt(seA2 + seB2);
  const diff = meanA - meanB;
  if (se === 0) return { meanA, meanB, nA, nB, diff, se: 0, t: diff === 0 ? 0 : Infinity, df: nA + nB - 2, pValue: diff === 0 ? 1 : 0 };
  const t = diff / se;
  // Welch-Satterthwaite approximate degrees of freedom -- NOT nA+nB-2 (that's
  // the pooled-variance Student's-t df; Welch's whole point is not pooling).
  const df = (seA2 + seB2) ** 2 / ((seA2 ** 2) / (nA - 1) + (seB2 ** 2) / (nB - 1));
  const pValue = 2 * (1 - studentTCDF(Math.abs(t), df));
  return { meanA, meanB, nA, nB, diff, se, t, df, pValue };
}

// Distribution-free fallback for the same H0 (equal population means), used
// when either group fails a normality check and Welch's CLT cushion can't
// be trusted. Recenters each group onto a common mean (enforcing H0
// directly) WITHOUT pooling them together first, then resamples each group
// SEPARATELY (with replacement) and recomputes the mean difference each
// time -- the actual null distribution, built straight from the data, with
// each group keeping its own shape and spread.
//
// This replaced a plain permutation test (pool both groups, shuffle labels,
// re-split) that was tried first and found to have a real calibration
// problem: shuffling the pooled data implicitly assumes the two groups are
// fully exchangeable -- same variance included -- under the null, which is
// exactly the assumption Welch's test (chosen for THIS test specifically
// because it doesn't require equal variance) was meant to avoid pairing
// with. Simulated against 2000 synthetic two-group comparisons per
// scenario with a KNOWN ground-truth mean difference (same non-normal
// shapes and 73.4% non-null rate as this project's own real corpus):
// equal variance/equal n, both methods matched Welch closely (~5% false
// positive rate at the nominal 5% level). Once variance AND sample size
// both differ between groups (plausible here -- two Boxplot columns can
// easily have different counts of valid rows) -- the classic
// Behrens-Fisher setup -- plain permutation's false-positive rate more
// than doubled to 12.4%, while this recentered-bootstrap version stayed
// close to nominal (7.4%) and matched or beat permutation's power in every
// scenario tested, including the equal-variance case. Replaced outright
// rather than kept as a second option, since it was never worse and
// sometimes much better calibrated.
function recenteredBootstrapMeanDiffTest(a, b, { iterations = 3000, seed = RNG_SEED } = {}) {
  const nA = a.length, nB = b.length;
  if (nA < 2 || nB < 2) return null;
  const meanA = a.reduce((s, v) => s + v, 0) / nA;
  const meanB = b.reduce((s, v) => s + v, 0) / nB;
  const observedDiff = meanA - meanB;
  const grandMean = (meanA * nA + meanB * nB) / (nA + nB);
  const aShifted = a.map(v => v - meanA + grandMean);
  const bShifted = b.map(v => v - meanB + grandMean);
  const random = createSeededRandom(seed);
  let atLeastAsExtreme = 0;
  for (let it = 0; it < iterations; it++) {
    let sumA = 0;
    for (let i = 0; i < nA; i++) sumA += aShifted[Math.floor(random() * nA)];
    let sumB = 0;
    for (let i = 0; i < nB; i++) sumB += bShifted[Math.floor(random() * nB)];
    const diff = sumA / nA - sumB / nB;
    if (Math.abs(diff) >= Math.abs(observedDiff) - 1e-12) atLeastAsExtreme++;
  }
  return { meanA, meanB, diff: observedDiff, pValue: atLeastAsExtreme / iterations, iterations };
}

// Grubbs' test: is the single most extreme point in this group too far from
// the mean (in its own SD units) to be explained by chance, if the group
// really were normal at this sample size? G = max|x-mean|/sd, compared
// against the critical value implied by a t-distribution at n-2 df,
// Bonferroni-corrected across the n points that could have been "the"
// extreme one. Returns true if this group looks contaminated by an outlier
// (as opposed to shapiroWilkTest, which flags overall shape problems like
// skew -- Grubbs is specifically about one-or-few points sitting apart from
// an otherwise-fine distribution).
function grubbsOutlierTest(arr) {
  const n = arr.length;
  if (n < 7) return { flagged: false }; // too little power below this
  const mean = arr.reduce((s, v) => s + v, 0) / n;
  const sd = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
  if (sd === 0) return { flagged: false, G: 0 };
  const maxDev = Math.max(...arr.map(v => Math.abs(v - mean)));
  const G = maxDev / sd;
  const alpha = 0.05 / n;
  const df = n - 2;
  const tCrit = studentTInverseCDF(1 - alpha / 2, df);
  const Gcrit = ((n - 1) / Math.sqrt(n)) * Math.sqrt((tCrit * tCrit) / (df + tCrit * tCrit));
  return { flagged: G > Gcrit, G, Gcrit };
}

// Yuen's trimmed-means test: like Welch's t-test, but on each group's
// trimmed mean (drops the top/bottom 20% of values first) with a
// Winsorized variance estimate, instead of the ordinary mean/variance. This
// makes it resistant to the exact thing Grubbs' test flags: a handful of
// genuinely extreme points dragging the plain average around. H0: equal
// trimmed-population means.
function yuenTrimmedMeansTest(a, b, trimFraction = 0.2) {
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);
  const nA = sa.length, nB = sb.length;
  const kA = Math.floor(trimFraction * nA), kB = Math.floor(trimFraction * nB);
  const hA = nA - 2 * kA, hB = nB - 2 * kB; // effective trimmed sample sizes
  if (hA < 2 || hB < 2) return null;
  const trimmedMean = (sorted, k) => {
    const slice = sorted.slice(k, sorted.length - k);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  };
  const winsorizedVar = (sorted, k) => {
    const n = sorted.length;
    const lo = sorted[k], hi = sorted[n - 1 - k];
    const w = sorted.map(v => Math.max(lo, Math.min(hi, v)));
    const mean = w.reduce((s, v) => s + v, 0) / n;
    return w.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  };
  const tMeanA = trimmedMean(sa, kA), tMeanB = trimmedMean(sb, kB);
  const wVarA = winsorizedVar(sa, kA), wVarB = winsorizedVar(sb, kB);
  const dA = wVarA * (nA - 1) / (hA * (hA - 1));
  const dB = wVarB * (nB - 1) / (hB * (hB - 1));
  const se = Math.sqrt(dA + dB);
  const diff = tMeanA - tMeanB;
  if (se === 0) return { tMeanA, tMeanB, diff, se: 0, t: diff === 0 ? 0 : Infinity, df: hA + hB - 2, pValue: diff === 0 ? 1 : 0 };
  const t = diff / se;
  const df = (dA + dB) ** 2 / ((dA ** 2) / (hA - 1) + (dB ** 2) / (hB - 1));
  const pValue = 2 * (1 - studentTCDF(Math.abs(t), df));
  return { tMeanA, tMeanB, diff, se, t, df, pValue };
}

// Runs both: Welch's as the primary result, plus Shapiro-Wilk on each group
// to say whether Welch's normality cushion is actually warranted here. Pays
// for the (slower) permutation test when at least one group's Shapiro-Wilk
// p-value drops below the plain 0.05 cutoff.
//
// Deliberately NOT reusing SW_P_STAR (0.13017) here. That number is this
// project's real calibrated "tipping point" for Shapiro-Wilk -- but it was
// derived from regression RESIDUALS, and two checks confirmed it doesn't
// transfer to raw data columns (what this test actually compares):
// 1. A within-dataset bootstrap (resampling rows inside each of this
//    project's 36 real datasets, 1800 draws) found the ~1700 draws already
//    flagged non-normal under 0.05 contain essentially all the real
//    disagreement between Welch's and the permutation test (3.1% of them
//    disagree); the extra ~70 draws SW_P_STAR alone would have newly
//    flagged showed ZERO disagreements -- raising the cutoff bought nothing.
// 2. Re-running SW_P_STAR's own mixture-model derivation, but on 79 real
//    raw data columns from this corpus instead of residuals, found an
//    estimated 95.6% of them are genuinely non-normal -- so lopsided that
//    the method's weighted-density crossover doesn't exist in a meaningful
//    sense (the "non-null" component already dominates everywhere). Real
//    data columns in this project just don't look Gaussian often enough for
//    a residual-style calibration to carry over; the plain 0.05 cutoff is
//    the honest choice here, not a placeholder waiting to be replaced.
//
// A third layer sits on top of both: Grubbs' test on each group, checking
// specifically for one-or-few extreme points (not overall non-normal shape
// -- that's what Shapiro-Wilk is for). When Grubbs flags either group,
// Yuen's trimmed-means test replaces Welch as the headline result, since
// Welch (and the recentered bootstrap, which still averages the raw
// values) are both still vulnerable to a handful of outliers pulling the
// mean around. Validated with 6000 synthetic pairs per scenario, known
// ground-truth mean differences, real corpus sizes: on clean/skewed data
// (no real outliers) Grubbs stays quiet (~10% trigger rate, matching
// chance) and decision accuracy is within 0.2pp of the current policy
// alone. Once real outlier contamination is present, Grubbs correctly
// fires 92-99% of the time and decision accuracy jumps 4-12 percentage
// points over the current policy (up to +12pp when one group is clean and
// the other has scattered extreme values) -- a close-to-free improvement.
function twoSampleMeanTest(a, b, options = {}) {
  const welch = welchTTest(a, b);
  if (!welch) return null;
  const swA = shapiroWilkTest(a);
  const swB = shapiroWilkTest(b);
  const normalityLooksOff = (swA && swA.pValue < 0.05) || (swB && swB.pValue < 0.05);
  const bootstrap = normalityLooksOff ? recenteredBootstrapMeanDiffTest(a, b, options) : null;
  const grubbsA = grubbsOutlierTest(a);
  const grubbsB = grubbsOutlierTest(b);
  const outliersLookPresent = grubbsA.flagged || grubbsB.flagged;
  const yuen = outliersLookPresent ? yuenTrimmedMeansTest(a, b) : null;
  return { welch, swA, swB, normalityLooksOff, bootstrap, grubbsA, grubbsB, outliersLookPresent, yuen };
}

function formatTwoSampleMeanTest(result, labelA, labelB) {
  if (!result) return 'Not enough data in one or both groups (need at least 2 valid values each).';
  const { welch, swA, swB, normalityLooksOff, bootstrap, grubbsA, grubbsB, outliersLookPresent, yuen } = result;
  const sig = welch.pValue < 0.05 ? ' <-- means look genuinely different' : ' (not distinguishable from "same mean")';
  const lines = [
    `Difference in means: "${labelA}" (n=${welch.nA}, mean=${welch.meanA.toFixed(4)}) vs "${labelB}" (n=${welch.nB}, mean=${welch.meanB.toFixed(4)})`,
    `  Welch's t-test (H0: equal means, unequal variances allowed): t(${welch.df.toFixed(1)})=${welch.t.toFixed(3)}, p=${welch.pValue.toFixed(4)}${sig}`,
  ];
  if (swA) lines.push(`  Shapiro-Wilk on "${labelA}" (H0: normal): W=${swA.W.toFixed(4)}, p=${swA.pValue.toFixed(4)}${swA.pValue < 0.05 ? '  <-- may not be normal' : ''}`);
  if (swB) lines.push(`  Shapiro-Wilk on "${labelB}" (H0: normal): W=${swB.W.toFixed(4)}, p=${swB.pValue.toFixed(4)}${swB.pValue < 0.05 ? '  <-- may not be normal' : ''}`);
  if (normalityLooksOff && bootstrap) {
    lines.push(`  At least one group looks non-normal -- cross-checked with a distribution-free recentered bootstrap (${bootstrap.iterations} resamples, each group resampled separately so unequal variances aren't assumed away):`);
    lines.push(`    p=${bootstrap.pValue.toFixed(4)}${bootstrap.pValue < 0.05 ? '  <-- also looks genuinely different' : '  (also not distinguishable from "same mean")'}`);
  }
  if (grubbsA && grubbsA.flagged) lines.push(`  Grubbs' test on "${labelA}": most extreme point looks like an outlier (G=${grubbsA.G.toFixed(3)} > critical ${grubbsA.Gcrit.toFixed(3)})`);
  if (grubbsB && grubbsB.flagged) lines.push(`  Grubbs' test on "${labelB}": most extreme point looks like an outlier (G=${grubbsB.G.toFixed(3)} > critical ${grubbsB.Gcrit.toFixed(3)})`);
  if (outliersLookPresent && yuen) {
    const yuenSig = yuen.pValue < 0.05 ? ' <-- also looks genuinely different' : ' (also not distinguishable from "same mean")';
    lines.push(`  At least one group has a likely outlier -- cross-checked with Yuen's trimmed-means test (drops the most extreme 20% on each side before comparing):`);
    lines.push(`    t(${yuen.df.toFixed(1)})=${yuen.t.toFixed(3)}, p=${yuen.pValue.toFixed(4)}${yuenSig}`);
  }
  return lines.join('\n');
}

// --- Residual diagnostics: Shapiro-Wilk (normality) and Durbin-Watson
// (autocorrelation) on a fit's own residuals. Both validated in a separate
// analysis session against this tool's own data corpus (see
// shapiro_wilk_test.js / autocorrelation_sensitivity_test.js in this
// folder) before being wired into the live page.

// Shapiro-Wilk: Royston (1995) AS R94 approximation, the same algorithm
// R's shapiro.test() uses. H0: the residuals are normally distributed.
function shapiroWilkTest(residuals) {
  const n = residuals.length;
  if (n < 3) return null;
  const sorted = [...residuals].sort((a, b) => a - b);
  const mean = sorted.reduce((s, v) => s + v, 0) / n;
  const ssq = sorted.reduce((s, v) => s + (v - mean) * (v - mean), 0);
  if (ssq === 0) return { W: 1, pValue: 1 };
  const m = new Array(n);
  for (let i = 1; i <= n; i++) m[i - 1] = studentTInverseNormalApprox((i - 0.375) / (n + 0.25));
  const ssumm2 = m.reduce((s, v) => s + v * v, 0);
  const rsn = 1 / Math.sqrt(n);
  const a = new Array(n).fill(0);
  if (n > 5) {
    const an = -2.706056 * rsn ** 5 + 4.434685 * rsn ** 4 - 2.071190 * rsn ** 3 - 0.147981 * rsn ** 2 + 0.221157 * rsn + m[n - 1] / Math.sqrt(ssumm2);
    const an1 = -3.582633 * rsn ** 5 + 5.682633 * rsn ** 4 - 1.752461 * rsn ** 3 - 0.293762 * rsn ** 2 + 0.042981 * rsn + m[n - 2] / Math.sqrt(ssumm2);
    const phi = (ssumm2 - 2 * m[n - 1] ** 2 - 2 * m[n - 2] ** 2) / (1 - 2 * an * an - 2 * an1 * an1);
    for (let i = 2; i < n - 2; i++) a[i] = m[i] / Math.sqrt(phi);
    a[n - 1] = an; a[0] = -an; a[n - 2] = an1; a[1] = -an1;
  } else if (n >= 4) {
    const an = -2.706056 * rsn ** 5 + 4.434685 * rsn ** 4 - 2.071190 * rsn ** 3 - 0.147981 * rsn ** 2 + 0.221157 * rsn + m[n - 1] / Math.sqrt(ssumm2);
    const phi = (ssumm2 - 2 * m[n - 1] ** 2) / (1 - 2 * an * an);
    for (let i = 1; i < n - 1; i++) a[i] = m[i] / Math.sqrt(phi);
    a[n - 1] = an; a[0] = -an;
  } else { a[0] = -Math.SQRT1_2; a[1] = 0; a[2] = Math.SQRT1_2; }
  let numerator = 0;
  for (let i = 0; i < n; i++) numerator += a[i] * sorted[i];
  const W = Math.min(1, (numerator * numerator) / ssq);
  let pValue;
  if (n <= 11) {
    const gamma = -2.273 + 0.459 * n;
    const w1 = -Math.log(gamma - Math.log(1 - W));
    const mu = 0.5440 - 0.39978 * n + 0.025054 * n * n - 0.0006714 * n * n * n;
    const sigma = Math.exp(1.3822 - 0.77857 * n + 0.062767 * n * n - 0.0020322 * n * n * n);
    pValue = 1 - normalCDFSimple((w1 - mu) / sigma);
  } else {
    const lnn = Math.log(n);
    const y = Math.log(1 - W);
    const mu = -1.5861 - 0.31082 * lnn - 0.083751 * lnn * lnn + 0.0038915 * lnn * lnn * lnn;
    const sigma = Math.exp(-0.4803 - 0.082676 * lnn + 0.0030302 * lnn * lnn);
    pValue = 1 - normalCDFSimple((y - mu) / sigma);
  }
  return { W, pValue };
}

// Standard normal CDF/inverse-CDF — separate, simpler implementations from
// the t-distribution machinery above (erf-based / Acklam's rational
// approximation), only needed by shapiroWilkTest and durbinWatsonTest.
function erfApprox(x) {
  const sign = x >= 0 ? 1 : -1; x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  return sign * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
}
function normalCDFSimple(z) { return 0.5 * (1 + erfApprox(z / Math.SQRT2)); }
function studentTInverseNormalApprox(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= phigh) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// Shapiro-Wilk is correctly calibrated at every sample size (confirmed by
// direct simulation — false-positive rate sits at ~5% from n=15 to
// n=1334), but it's also genuinely MORE POWERFUL with more data, which is
// expected, correct behavior for any working test, not a flaw: a real,
// even small, departure from normality gets easier to detect as n grows.
// Pooled across this project's own 35-dataset corpus, that shows up as a
// real correlation between "looks significant" and "how much data the
// dataset had" (Spearman rho=0.57, p=0.0003) — not because small datasets
// are secretly more normal, but because the test has less power to catch
// the same-sized real effect with less data.
//
// This doesn't need "fixing" (there's no bug — see autocorrelation/
// heteroscedasticity write-ups for what an actual calibration bug looks
// like, e.g. the Anderson-Darling fix elsewhere in this file).
//
// An earlier version of this comment built a SIZE-ADJUSTED secondary
// p-value here: detrend raw p against a curve fit to log(n) (first a
// straight line in logit-space, later a logistic-offset curve -- see git
// history). Both were retired after a later session's null-simulation
// check (size_adjust_null_calibration_check.js) showed the detrend itself
// goes badly uncalibrated past roughly n=150: feeding it TRUE-null
// synthetic data (zero real non-normality, by construction, at every n)
// made the Shapiro-Wilk/Durbin-Watson versions' flag rate collapse to 0%
// and White's inflate to ~21% (vs. the correct ~10%). The root problem:
// that curve was fit on this project's own real, mixed corpus, which
// encodes "what's TYPICAL for a real dataset this size," not "what a truly
// null dataset of this size looks like" -- those are different things, and
// conflating them is exactly the kind of bias this project otherwise
// avoids.
//
// Replaced with a DENSITY-CROSSOVER classification that needs no size
// adjustment at all: z = Phi^-1(1 - rawP) is exactly N(0,1) under the true
// null at EVERY n, by construction (that's what a valid continuous
// p-value's uniformity under H0 guarantees) -- not an approximation this
// project had to calibrate, a mathematical identity. A two-component
// Gaussian mixture fit in z-space (null fixed at N(0,1); non-null component
// N(mu,sigma^2) estimated by EM on this project's own 35-dataset corpus)
// gives each test's real Bayes-optimal classification boundary -- where the
// two components' weighted densities actually cross, not a naive weighted
// average of their means (sw_w_mixture_tipping_point.js has the derivation
// and why the naive version is wrong). Re-run as a FIXED threshold against
// synthetic null data (density_crossover_vs_logistic_null_check.js), this
// stays flat at its own target rate across n=10 to n=1300 -- no drift,
// because the threshold is on an already size-free quantity instead of
// trying to correct a size-confounded one.
//
// RETIRED NUMBER: an earlier session, still using the logistic-offset
// size-adjustment above, estimated the corpus's overall non-null fraction
// at 88.5% via a mixture model built on top of that adjustment. That
// figure inherits the same post-n~150 miscalibration this whole comment
// describes, so it should not be cited or compared against going forward.
// The corpus's non-null fraction, from the density-crossover mixture
// (zmixture_bootstrap_all_tests.js, percentile bootstrap over the 35
// datasets): Shapiro-Wilk 73.4% [95% CI 47.0%-97.1%], White 64.0%
// [32.8%-95.2%], Durbin-Watson 77.2% [48.5%-99.9%]. Wide CIs are real --
// a consequence of only having 35 datasets, same bottleneck noted
// elsewhere in this project -- not a defect in the method.
const SW_P_STAR = 0.13017;
const WHITE_P_STAR = 0.13331;
const DW_P_STAR = 0.11468;
function shapiroWilkLooksNonNull(pValue) { return pValue < SW_P_STAR; }
function whiteLooksNonNull(pValue) { return pValue < WHITE_P_STAR; }
function durbinWatsonLooksNonNull(pValue) { return pValue < DW_P_STAR; }

// Durbin-Watson: checks lag-1 autocorrelation via the standard large-sample
// normal approximation (the implied lag-1 autocorrelation r1 = 1 - DW/2 is
// approximately Normal(0, 1/n) under the null). H0: no autocorrelation.
// Chosen over Ljung-Box for this tool specifically because a real-data
// sensitivity check (this session, autocorrelation_sensitivity_test.js)
// found DW's significance tracks actual held-out forecast failure more
// closely than Ljung-Box's does on this project's own data corpus.
function durbinWatsonTest(residuals) {
  const n = residuals.length;
  if (n < 3) return null;
  let num = 0, denom = 0;
  for (let t = 1; t < n; t++) num += (residuals[t] - residuals[t - 1]) ** 2;
  for (let t = 0; t < n; t++) denom += residuals[t] ** 2;
  if (denom === 0) return { dw: 2, pValue: 1 };
  const dw = num / denom;
  const r1 = 1 - dw / 2;
  const z = r1 * Math.sqrt(n);
  return { dw, pValue: 2 * (1 - normalCDFSimple(Math.abs(z))) };
}

// --- Regularized lower incomplete gamma P(a,x), needed for the chi-square
// CDF below (White's test). Same Numerical-Recipes series/continued-fraction
// split used throughout this project's analysis scripts, reusing logGamma.
function gammaSeriesSimple(a, x) {
  const ITMAX = 200, EPS = 3e-14;
  if (x <= 0) return 0;
  let ap = a, sum = 1 / a, del = sum;
  for (let n = 1; n <= ITMAX; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * EPS) break; }
  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}
function gammaContinuedFractionSimple(a, x) {
  const ITMAX = 200, EPS = 3e-14, FPMIN = 1e-300;
  let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
  for (let i = 1; i <= ITMAX; i++) {
    const an = -i * (i - a); b += 2;
    d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}
function chiSquareCDFSimple(x, k) {
  if (x <= 0) return 0;
  return x < k / 2 + 1 ? gammaSeriesSimple(k / 2, x / 2) : 1 - gammaContinuedFractionSimple(k / 2, x / 2);
}

// White's test for heteroscedasticity: regresses SQUARED residuals on x and
// x² (an auxiliary regression), LM = n*R² ~ chi-square(2). H0: constant
// variance. Chosen as this tool's default heteroscedasticity check over
// Breusch-Pagan and Goldfeld-Quandt after two separate checks against this
// project's own data corpus (this session): (1) White's significance
// correlated most strongly with actual held-out forecast failure
// (-0.183, vs Breusch-Pagan's -0.058 and Goldfeld-Quandt's +0.021 — the
// wrong direction), and (2) White was also the most outlier-resistant of
// the three (47% of its flags evaporate after removing MAD-outliers, vs
// 71% for Breusch-Pagan and 88% for Goldfeld-Quandt — Goldfeld-Quandt in
// particular is fragile because it compares two halves' variance sums
// directly, so one bad point in either half can dominate it).
function whiteTest(xs, residuals) {
  const n = residuals.length;
  if (n < 8) return null;
  const X = xs.map(x => [x, x * x]);
  const y = residuals.map(e => e * e);
  let fit;
  try { fit = fitMultiLinear(X, y); } catch (e) { return null; }
  const LM = n * Math.max(0, fit.r2);
  return { LM, pValue: 1 - chiSquareCDFSimple(LM, 2) };
}

// --- Error-distribution selection for residuals: every BIC/marginal-
// likelihood comparison elsewhere in this project (and any future
// Laplace-approximation CI work) implicitly assumes Gaussian errors. This
// corpus's own Anderson-Darling results show the majority of datasets are
// NOT normal (tail-driven departures), so that assumption is worth
// checking per-fit rather than taken for granted. These three candidate
// error distributions are fit by exact or near-exact maximum likelihood
// and compared by BIC (penalizes the heavier models' extra parameters,
// appropriate for "which distribution really explains this data" rather
// than "which fits best no matter how many knobs it has").
//
// Gaussian: MLE variance = mean(r^2), 1 parameter (sigma).
function gaussianErrorFit(residuals) {
  const n = residuals.length;
  const sigma2 = residuals.reduce((s, r) => s + r * r, 0) / n;
  if (sigma2 <= 0) return null;
  const logLik = residuals.reduce((s, r) => s - 0.5 * Math.log(2 * Math.PI * sigma2) - (r * r) / (2 * sigma2), 0);
  return { name: 'Gaussian', k: 1, logLik, params: { sigma: Math.sqrt(sigma2) } };
}

// Laplace (double-exponential): MLE scale b = mean(|r|), 1 parameter. Fatter
// tails than Gaussian, thinner than a very heavy-tailed Student-t — the
// natural "slightly heavy-tailed" alternative.
function laplaceErrorFit(residuals) {
  const n = residuals.length;
  const b = residuals.reduce((s, r) => s + Math.abs(r), 0) / n;
  if (b <= 0) return null;
  const logLik = residuals.reduce((s, r) => s - Math.log(2 * b) - Math.abs(r) / b, 0);
  return { name: 'Laplace', k: 1, logLik, params: { b } };
}

// Student-t: 2 parameters (scale, degrees of freedom nu — lower nu = heavier
// tails, nu->Infinity recovers Gaussian). No closed-form MLE for either
// parameter jointly, so this profiles over a fixed grid of nu (the values
// practically distinguishable from each other and from Gaussian) and, for
// each candidate nu, finds the MLE scale by a short EM iteration (the
// standard approach for a t-distribution's scale: treat each residual as
// Normal with its own latent precision-weight, alternate estimating the
// weights and re-estimating the scale — converges in a handful of steps).
const STUDENT_T_DF_GRID = [2, 3, 4, 5, 7, 10, 20, 30, 50];
function studentTErrorFit(residuals) {
  const n = residuals.length;
  const initVar = residuals.reduce((s, r) => s + r * r, 0) / n;
  if (initVar <= 0) return null;
  let best = null;
  for (const nu of STUDENT_T_DF_GRID) {
    let sigma2 = initVar;
    for (let iter = 0; iter < 15; iter++) {
      let wSum = 0, wr2Sum = 0;
      for (const r of residuals) {
        const w = (nu + 1) / (nu + (r * r) / sigma2);
        wSum += w; wr2Sum += w * r * r;
      }
      const nextSigma2 = wr2Sum / n;
      if (!isFinite(nextSigma2) || nextSigma2 <= 0) break;
      sigma2 = nextSigma2;
    }
    const logConst = logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(nu * Math.PI * sigma2);
    const logLik = residuals.reduce((s, r) => s + logConst - ((nu + 1) / 2) * Math.log(1 + (r * r) / (nu * sigma2)), 0);
    if (!best || logLik > best.logLik) best = { name: 'Student-t', k: 2, logLik, params: { sigma: Math.sqrt(sigma2), nu } };
  }
  return best;
}

// Cauchy: 1 parameter (scale gamma). No closed-form MLE, but its
// log-likelihood is well-behaved/unimodal in gamma over a sane search
// range, so golden-section search finds the maximum directly. Much
// heavier tails than Student-t even at low nu — the right alternative
// when a handful of residuals are wildly larger than the rest and no
// amount of extra Student-t degrees-of-freedom flexibility captures it.
// Validated via bootstrap against this project's own real-data corpus
// (backward_elimination_bootstrap-style corpus resampling): adding Cauchy
// as a candidate cuts the average distance between the chosen and true
// error distribution roughly in half on truly Cauchy-shaped data, with
// zero measurable cost on Gaussian/Laplace/Student-t data.
function cauchyErrorFit(residuals) {
  const n = residuals.length;
  const scaleGuess = residuals.reduce((s, r) => s + Math.abs(r), 0) / n;
  if (scaleGuess <= 0) return null;
  const ll = gamma => residuals.reduce((acc, r) => acc - Math.log(Math.PI * gamma * (1 + (r / gamma) ** 2)), 0);
  let lo = scaleGuess * 0.05, hi = scaleGuess * 5;
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = hi - gr * (hi - lo), d = lo + gr * (hi - lo);
  for (let iter = 0; iter < 60; iter++) {
    if (ll(c) < ll(d)) { lo = c; c = d; d = lo + gr * (hi - lo); } else { hi = d; d = c; c = hi - gr * (hi - lo); }
  }
  const gamma = (lo + hi) / 2;
  if (!isFinite(gamma) || gamma <= 0) return null;
  return { name: 'Cauchy', k: 1, logLik: ll(gamma), params: { gamma } };
}

// Compares all three by BIC (k*ln(n) - 2*logLik, lower is better) — BIC
// over AIC here because the question is "which distribution is really
// generating this", a model-identification question BIC is designed for,
// not "which predicts best" (AIC's usual framing). A difference of 2-6 is
// "positive" evidence, 6-10 "strong", >10 "very strong" (standard BIC
// evidence-strength bands) for whichever model has the lower score.
function chooseErrorDistribution(residuals) {
  const n = residuals.length;
  if (n < 8) return null;
  const candidates = [gaussianErrorFit(residuals), laplaceErrorFit(residuals), studentTErrorFit(residuals), cauchyErrorFit(residuals)]
    .filter(Boolean)
    .map(c => ({ ...c, bic: c.k * Math.log(n) - 2 * c.logLik }));
  if (!candidates.length) return null;
  candidates.sort((a, b) => a.bic - b.bic);
  const best = candidates[0];
  const gaussian = candidates.find(c => c.name === 'Gaussian');
  return { candidates, best, deltaBICvsGaussian: gaussian ? gaussian.bic - best.bic : 0 };
}

// Standard normal quantile (Acklam's rational approximation) — same
// algorithm as studentTInverseNormalApprox above, duplicated here rather
// than reused across that function's unrelated name, for a general-purpose
// outlier-bound use (see errorDistributionOutlierBounds below).
function standardNormalQuantile(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= phigh) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// Gives the outlier cutoffs (in RESIDUAL units, i.e. relative to whatever
// center the caller subtracted out) implied by whichever error distribution
// chooseErrorDistribution finds best-supported for these residuals --
// Gaussian, Laplace, or Student-t -- instead of a fixed-multiplier rule
// like Tukey's 1.5*IQR. Each one has a real, known tail, so the cutoff is
// just that distribution's own quantile at a chosen tail probability.
//
// tailProb=0.003483 (per side, ~0.6966% combined) is deliberately the
// GAUSSIAN-EQUIVALENT rate of Tukey's classic 1.5*IQR fence (that rule's
// well-known false-flag rate on genuinely normal data) -- so when the
// data really is Gaussian, this reproduces Tukey's own flag rate almost
// exactly, and the two methods only diverge (correctly) when the data's
// real shape isn't Gaussian: a heavier-tailed Student-t or Laplace fit
// automatically widens the cutoff (fewer false flags on data that's
// naturally prone to extreme-looking points), without hand-tuning a
// separate multiplier per shape.
//
// FITS ROBUSTLY, not on the raw residuals directly: a plain MLE fit is
// fooled by the very outliers it's supposed to catch (confirmed directly —
// a single gross outlier widens the fitted Student-t's scale enough to
// swallow itself, matching the classic "masking" failure mode of
// likelihood-based outlier detection). Fix: pre-trim with a preliminary
// Tukey 1.5*IQR fence first, fit the distribution on what's LEFT, then use
// that distribution's own tail quantile as the real cutoff — same
// trim-then-estimate discipline every other robust method in this project
// already follows (Least Trimmed Squares, Huber, Tukey biweight...), not
// a one-off exception for this feature.
function errorDistributionOutlierBounds(residuals, { tailProb = 0.003483 } = {}) {
  const sorted = [...residuals].sort((a, b) => a - b);
  const quantileOfSorted = q => {
    const idx = q * (sorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  };
  const q1 = quantileOfSorted(0.25), q3 = quantileOfSorted(0.75);
  const iqr = q3 - q1;
  const loFence = q1 - 1.5 * iqr, hiFence = q3 + 1.5 * iqr;
  const trimmed = residuals.filter(r => r >= loFence && r <= hiFence);

  const chosen = chooseErrorDistribution(trimmed);
  if (!chosen) return null; // too few points left after trimming (or too few to begin with) to tell distributions apart
  const { best } = chosen;
  let lower, upper;
  if (best.name === 'Gaussian') {
    const { sigma } = best.params;
    lower = sigma * standardNormalQuantile(tailProb);
    upper = sigma * standardNormalQuantile(1 - tailProb);
  } else if (best.name === 'Laplace') {
    // Laplace(0,b) CDF is piecewise-exponential -- closed-form inverse, no
    // special function needed: F(x)=0.5*e^(x/b) for x<0, 1-0.5*e^(-x/b) for x>=0.
    const { b } = best.params;
    lower = b * Math.log(2 * tailProb);
    upper = -b * Math.log(2 * tailProb);
  } else if (best.name === 'Student-t') {
    const { sigma, nu } = best.params;
    lower = sigma * studentTInverseCDF(tailProb, nu);
    upper = sigma * studentTInverseCDF(1 - tailProb, nu);
  } else {
    return null;
  }
  return { lower, upper, distributionName: best.name };
}

// Purely informational: describes the residuals' actual shape. Checked
// twice whether this should feed into anything else in the pipeline
// (error_dist_ci_realdata_check.js: no measurable effect on CI width vs. a
// bootstrap reference; a since-removed spike-and-slab BIC-threading
// experiment: no measurable effect on model-selection either, once a
// formula-artifact confound was controlled for) — both came back negative,
// so this is reported for awareness only, not as a flag that something
// downstream needs correcting.
function formatErrorDistributionReport(result) {
  if (!result) return '';
  const { best, deltaBICvsGaussian } = result;
  if (best.name === 'Gaussian' || deltaBICvsGaussian < 2) {
    return `\n      Error distribution check: Gaussian is still the best (or no other candidate clearly beats it, ΔBIC=${deltaBICvsGaussian.toFixed(1)}).`;
  }
  const strength = deltaBICvsGaussian > 10 ? 'very strong' : deltaBICvsGaussian > 6 ? 'strong' : 'positive';
  const paramDesc = best.name === 'Student-t' ? ` (df=${best.params.nu}, heavier tails than Gaussian)` : '';
  // Worded to stay accurate whether this check is running on the original
  // fit's residuals OR an already-switched robust refit's residuals (this
  // function is called again, unconditionally, on whatever the FINAL
  // displayed fit turns out to be) — never claims a refit "was kept as-is"
  // or "wasn't tried", since by the time this prints, that decision is
  // already made elsewhere and may have gone either way. No specific bar
  // number is quoted here (unlike earlier versions of this message) since
  // this function doesn't know which model family produced these residuals
  // — polynomial, exponential, power, and logarithmic each use their OWN
  // bootstrap-confirmed bar (see ROBUST_REFIT_DELTA_BIC_THRESHOLD and
  // ROBUST_REFIT_LOGLINEAR_THRESHOLDS), and citing the wrong one here
  // would be actively misleading.
  return `\n      Error distribution check: ${best.name}${paramDesc} fits these residuals better than Gaussian` +
         ` (ΔBIC=${deltaBICvsGaussian.toFixed(1)}, ${strength} evidence).`;
}

// Robust refit gate: only engage a reweighted (IRLS) refit instead of plain
// OLS when the error-distribution evidence is at least this strong. The
// number itself — not just the idea of a threshold — comes from a direct
// bootstrap optimization (error_dist_refit_threshold_test.js): a synthetic
// ground-truth test (1500 trials spanning Gaussian, pure Laplace/Student-t,
// and symmetric/skewed outlier contamination at varying severity) found the
// threshold that minimizes parameter error bootstraps to ΔBIC=40 (95% CI
// 40-130), and confirmed that tightening further (up to +30%, ΔBIC~52)
// keeps the benefit intact while triggering less often. 45 sits inside that
// confirmed-safe range. Below this bar, refitting measurably does NOT help
// on average (the same test found near-zero or occasionally negative effect
// for weak evidence, e.g. pure heavy-tailed-but-not-contaminated noise) —
// so this is deliberately a high bar, not "any non-Gaussian flag".
const ROBUST_REFIT_DELTA_BIC_THRESHOLD = 45;

// Same idea, separately bootstrapped for the log-linear family
// (exponential, power, logarithmic — see maybeRobustRefitLogLinear below).
// A first pass POOLED all three log-linear types into one sweep and found
// ΔBIC=100 optimal (120 at +20%) — but splitting the same 2700-trial sweep
// BY FAMILY (error_dist_refit_loglinear_threshold_test.js's per-family
// breakdown) showed the pooled number was hiding real, substantial
// differences between families, not just noise:
//   - exponential: optimal 80 (95% CI 60-80), confirmed positive through
//     +30% (104)
//   - power: optimal 130 (95% CI 130-160), confirmed positive through
//     +30% (169) — needs 62% MORE evidence than exponential before a
//     refit reliably helps, plausibly because its double-log (ln x, ln y)
//     transform compounds residual scale differently than a single ln(y)
//     transform does
//   - logarithmic: optimal 100 (95% CI 40-130, wide), but the confirmed
//     effect actually BREAKS at +30% (130): the CI crosses zero
//     (-10.0 to +0.76) — a narrower safety margin than the other two
// Each threshold below sits between that family's own optimum and its
// confirmed-safe tightened value, same placement logic as the polynomial
// gate's 45 (between optimal 40 and safe-tightened 48-52).
const ROBUST_REFIT_LOGLINEAR_THRESHOLDS = {
  exponential: 95,   // optimal 80, confirmed-safe through 104
  power: 145,        // optimal 130, confirmed-safe through 169
  logarithmic: 110,  // optimal 100, but stays clear of the 130 breakdown point
};

// IRLS (iteratively reweighted least squares) refit of a polynomial fit,
// reusing fitPolynomial's existing `weights` parameter rather than any new
// fitting machinery — each iteration computes residuals from the current
// fit, derives new weights from them (down-weighting whatever the detected
// distribution says should carry less influence), and refits. Only
// validated for simple lines (degree 1) in the synthetic/real-data tests
// this threshold came from; applying it to higher-degree polynomials
// reuses the same mechanism but hasn't been separately re-validated at
// those degrees.
function fitRobustPolynomial(xs, ys, degree, distName, nu, iterations = 20) {
  let fit = fitPolynomial(xs, ys, degree);
  let sigma2 = null;
  if (distName === 'Student-t') {
    const r0 = xs.map((x, i) => ys[i] - fit.predict(x));
    sigma2 = r0.reduce((s, v) => s + v * v, 0) / xs.length;
    if (sigma2 <= 0) return fit;
  }
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    let weights;
    if (distName === 'Laplace') {
      const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / xs.length * 0.05);
      weights = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    } else {
      weights = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    }
    let next;
    try { next = fitPolynomial(xs, ys, degree, weights); } catch (e) { break; }
    fit = next;
    if (distName === 'Student-t') {
      const newResiduals = xs.map((x, i) => ys[i] - fit.predict(x));
      const wSum = weights.reduce((s, v) => s + v, 0);
      const newSigma2 = newResiduals.reduce((s, r, i) => s + weights[i] * r * r, 0) / wSum;
      if (newSigma2 > 0) sigma2 = newSigma2;
    }
  }
  return fit;
}

// The actual gate: runs chooseErrorDistribution on a plain fit's residuals
// and, only past ROBUST_REFIT_DELTA_BIC_THRESHOLD, returns a robust refit
// instead. Returns the ORIGINAL fit unchanged (switched: false) otherwise —
// callers always get something usable back. `degree` must be the plain
// fit's own polynomial degree (fitPolynomial/runRegression already expose
// this as `fit.degree`).
function maybeRobustRefit(fit, xs, ys, degree) {
  if (xs.length < 8) return { fit, switched: false };
  const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
  const errDist = chooseErrorDistribution(residuals);
  if (!errDist || errDist.best.name === 'Gaussian' || errDist.deltaBICvsGaussian < ROBUST_REFIT_DELTA_BIC_THRESHOLD) {
    return { fit, switched: false };
  }
  const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
  const refit = fitRobustPolynomial(xs, ys, degree, errDist.best.name, nu);
  const method = errDist.best.name === 'Laplace' ? 'L1 (least-absolute-deviations) IRLS' : 'Student-t IRLS';
  return {
    fit: refit,
    switched: true,
    reason: `${errDist.best.name} errors detected (ΔBIC=${errDist.deltaBICvsGaussian.toFixed(1)} vs Gaussian, >= the ${ROBUST_REFIT_DELTA_BIC_THRESHOLD}-point bootstrap-confirmed bar) — refit via ${method} instead of OLS`,
  };
}

// Robust refit for the log-linear family (exponential, power, logarithmic)
// — fitExponential/fitPower/fitLogarithmic are all secretly just
// fitPolynomial on a TRANSFORMED (x, y) pair (see each function's own
// comment), so the error-distribution detection and IRLS reweighting both
// need to happen in that SAME transformed space, not the original (x, y)
// scale. Fixing this was the point: an earlier version of this gate (see
// error_dist_refit_other_types_test.js) computed weights from
// ORIGINAL-scale residuals and fed them into fitPower's underlying
// ln(x)/ln(y) fit — a real mismatch, confirmed to actively HURT the power
// fit even on clean Gaussian data (CI 0.078 to 0.109, entirely harmful).
// Detecting and reweighting in the correct transformed space fixes the
// mismatch at its root, rather than patching around it.
//
// Each entry: toX/toY transform the original (x, y) into the space the
// underlying line fit actually runs in; fromLineFit converts that line
// fit's {coeffs} back into this model's own {a, b, predict, type}
// shape — one each, matching exactly how fitExponential/fitPower/
// fitLogarithmic build their own return values.
const LOG_LINEAR_TRANSFORMS = {
  exponential: {
    toX: x => x, toY: y => Math.log(y),
    fromLineFit: lineFit => {
      const a = Math.exp(lineFit.coeffs[0]), b = lineFit.coeffs[1];
      return { type: 'exponential', a, b, predict: x => a * Math.exp(b * x) };
    },
  },
  power: {
    toX: x => Math.log(x), toY: y => Math.log(y),
    fromLineFit: lineFit => {
      const a = Math.exp(lineFit.coeffs[0]), b = lineFit.coeffs[1];
      return { type: 'power', a, b, predict: x => a * Math.pow(x, b) };
    },
  },
  logarithmic: {
    toX: x => Math.log(x), toY: y => y,
    fromLineFit: lineFit => {
      const b = lineFit.coeffs[0], a = lineFit.coeffs[1];
      return { type: 'logarithmic', a, b, predict: x => a * Math.log(x) + b };
    },
  },
};

function maybeRobustRefitLogLinear(fit, xs, ys, kind) {
  if (xs.length < 8) return { fit, switched: false };
  const transform = LOG_LINEAR_TRANSFORMS[kind];
  const threshold = ROBUST_REFIT_LOGLINEAR_THRESHOLDS[kind];
  if (!transform || threshold === undefined) return { fit, switched: false };
  // Transformed-space residual: toY(y) - toY(predict(x)), the space the
  // underlying line fit (and hence any Gaussian/Laplace/Student-t
  // assumption about its errors) actually operates in.
  const transformedResiduals = xs.map((x, i) => transform.toY(ys[i]) - transform.toY(fit.predict(x)));
  const errDist = chooseErrorDistribution(transformedResiduals);
  if (!errDist || errDist.best.name === 'Gaussian' || errDist.deltaBICvsGaussian < threshold) {
    return { fit, switched: false };
  }
  const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
  const txs = xs.map(transform.toX), tys = ys.map(transform.toY);
  const lineFit = fitRobustPolynomial(txs, tys, 1, errDist.best.name, nu);
  const refit = transform.fromLineFit(lineFit);
  refit.r2 = computeR2(xs, ys, refit.predict); // report R^2 on the ORIGINAL scale, same convention as the plain fit
  const method = errDist.best.name === 'Laplace' ? 'L1 (least-absolute-deviations) IRLS' : 'Student-t IRLS';
  return {
    fit: refit,
    switched: true,
    reason: `${errDist.best.name} errors detected in the fit's natural (transformed) scale (ΔBIC=${errDist.deltaBICvsGaussian.toFixed(1)} vs Gaussian, >= this fit type's own ${threshold}-point bar) — refit via ${method} instead of OLS`,
  };
}

// Robust refit for fitMultiExponential — same IRLS-in-the-natural-log-space
// idea as maybeRobustRefitLogLinear, generalized to multiple predictors by
// reusing fitMultiLinear's own `weights` parameter instead of
// fitPolynomial's. Separately bootstrapped (error_dist_refit_multiexp_threshold_test.js,
// 900 synthetic trials, a contamination-severity sweep with 2 predictors):
// optimal ΔBIC=60 (95% CI 60-80), confirmed net-positive through +30%
// (78). 70 sits between the optimum and that confirmed-safe range, same
// placement logic as every other gate's threshold.
const ROBUST_REFIT_MULTIEXP_DELTA_BIC_THRESHOLD = 70;

function fitRobustMultiLinear(X, y, distName, nu, iterations = 20) {
  let fit = fitMultiLinear(X, y);
  let sigma2 = null;
  if (distName === 'Student-t') {
    const r0 = X.map((row, i) => y[i] - fit.predict(row));
    sigma2 = r0.reduce((s, v) => s + v * v, 0) / X.length;
    if (sigma2 <= 0) return fit;
  }
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = X.map((row, i) => y[i] - fit.predict(row));
    let weights;
    if (distName === 'Laplace') {
      const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / X.length * 0.05);
      weights = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    } else {
      weights = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    }
    let next;
    try { next = fitMultiLinear(X, y, weights); } catch (e) { break; }
    fit = next;
    if (distName === 'Student-t') {
      const newResiduals = X.map((row, i) => y[i] - fit.predict(row));
      const wSum = weights.reduce((s, v) => s + v, 0);
      const newSigma2 = newResiduals.reduce((s, r, i) => s + weights[i] * r * r, 0) / wSum;
      if (newSigma2 > 0) sigma2 = newSigma2;
    }
  }
  return fit;
}

function maybeRobustRefitMultiExponential(fit, X, y) {
  if (X.length < 8) return { fit, switched: false };
  const transformedResiduals = X.map((row, i) => Math.log(y[i]) - Math.log(fit.predict(row)));
  const errDist = chooseErrorDistribution(transformedResiduals);
  if (!errDist || errDist.best.name === 'Gaussian' || errDist.deltaBICvsGaussian < ROBUST_REFIT_MULTIEXP_DELTA_BIC_THRESHOLD) {
    return { fit, switched: false };
  }
  const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
  const lineFit = fitRobustMultiLinear(X, y.map(Math.log), errDist.best.name, nu);
  const a = Math.exp(lineFit.intercept);
  const coeffs = lineFit.coeffs;
  const refit = { type: 'multi-exponential', a, coeffs, predict: xRow => a * Math.exp(xRow.reduce((s, v, j) => s + v * coeffs[j], 0)) };
  refit.r2 = computeR2Multi(X, y, refit.predict);
  const method = errDist.best.name === 'Laplace' ? 'L1 (least-absolute-deviations) IRLS' : 'Student-t IRLS';
  return {
    fit: refit,
    switched: true,
    reason: `${errDist.best.name} errors detected in the fit's natural (log) scale (ΔBIC=${errDist.deltaBICvsGaussian.toFixed(1)} vs Gaussian, >= the ${ROBUST_REFIT_MULTIEXP_DELTA_BIC_THRESHOLD}-point bar) — refit via ${method} instead of OLS`,
  };
}

// Robust refit for fitProportional (y = a*x, no intercept) — no log
// transform or multi-predictor generalization needed here, so this is the
// simplest of the gates: IRLS directly reusing fitProportional's own
// `weights` parameter on the original scale. Separately bootstrapped
// (error_dist_refit_proportional_threshold_test.js, 900 synthetic trials):
// optimal ΔBIC=60 (95% CI 26-130, wider than the other gates' but still
// confirmed net-positive through +30%, 78). 70 sits between the optimum
// and that confirmed-safe range, same placement logic as every other gate.
const ROBUST_REFIT_PROPORTIONAL_DELTA_BIC_THRESHOLD = 70;

function fitRobustProportional(xs, ys, distName, nu, iterations = 20) {
  let fit = fitProportional(xs, ys);
  let sigma2 = null;
  if (distName === 'Student-t') {
    const r0 = xs.map((x, i) => ys[i] - fit.predict(x));
    sigma2 = r0.reduce((s, v) => s + v * v, 0) / xs.length;
    if (sigma2 <= 0) return fit;
  }
  for (let iter = 0; iter < iterations; iter++) {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
    let weights;
    if (distName === 'Laplace') {
      const scale = Math.max(1e-6, residuals.reduce((s, r) => s + Math.abs(r), 0) / xs.length * 0.05);
      weights = residuals.map(r => 1 / Math.max(Math.abs(r), scale));
    } else {
      weights = residuals.map(r => (nu + 1) / (nu * sigma2 + r * r));
    }
    let next;
    try { next = fitProportional(xs, ys, weights); } catch (e) { break; }
    fit = next;
    if (distName === 'Student-t') {
      const newResiduals = xs.map((x, i) => ys[i] - fit.predict(x));
      const wSum = weights.reduce((s, v) => s + v, 0);
      const newSigma2 = newResiduals.reduce((s, r, i) => s + weights[i] * r * r, 0) / wSum;
      if (newSigma2 > 0) sigma2 = newSigma2;
    }
  }
  return fit;
}

function maybeRobustRefitProportional(fit, xs, ys) {
  if (xs.length < 8) return { fit, switched: false };
  const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
  const errDist = chooseErrorDistribution(residuals);
  if (!errDist || errDist.best.name === 'Gaussian' || errDist.deltaBICvsGaussian < ROBUST_REFIT_PROPORTIONAL_DELTA_BIC_THRESHOLD) {
    return { fit, switched: false };
  }
  const nu = errDist.best.name === 'Student-t' ? errDist.best.params.nu : null;
  const refit = fitRobustProportional(xs, ys, errDist.best.name, nu);
  const method = errDist.best.name === 'Laplace' ? 'L1 (least-absolute-deviations) IRLS' : 'Student-t IRLS';
  return {
    fit: refit,
    switched: true,
    reason: `${errDist.best.name} errors detected (ΔBIC=${errDist.deltaBICvsGaussian.toFixed(1)} vs Gaussian, >= the ${ROBUST_REFIT_PROPORTIONAL_DELTA_BIC_THRESHOLD}-point bar) — refit via ${method} instead of OLS`,
  };
}

// fitLogisticCurve's robust-refit gate, same spirit as the other
// maybeRobustRefit* functions above but a different mechanism: logistic is
// fit by gradient descent (no closed-form line to swap in an IRLS variant
// for), so instead of the chooseErrorDistribution/BIC approach those use,
// this fits once, drops points whose residual sits more than k=3 robust
// sigmas from the median, and refits on survivors if anything was dropped.
// Compared directly against Huber-weighted, Tukey-biweight, and LTS-style
// alternatives (_scratch_logistic_robust_compare.js, since removed): this
// MAD pre-trim had the best or near-best parameter accuracy at every
// contamination level tested (e.g. 0.267 vs plain's 0.381 at 5%
// contamination, down to 0.320 vs 0.531 at 20%), a negligible clean-data
// cost (0.2444 vs plain's 0.2447 param error at 0% contamination), and was
// the cheapest of the three viable options (~2x plain's runtime — trivial
// for this page's interactive use). It's also the only one of the four
// that produced a concrete real-corpus improvement: "Pressure over time
// (sec)" went from checkR2=-0.597 to +0.360 once an apparent outlier
// stopped dragging the fit. Huber barely moved any real dataset and had
// weaker synthetic gains; LTS was both ~30x slower and occasionally
// destructive on clean real data (one good fit's R2 went from 0.845 to
// -2.225) despite no real outliers being present — rejected as unreliable.
function maybeRobustRefitLogistic(fit, xs, ys) {
  if (xs.length < 8) return { fit, switched: false };
  const residuals = xs.map((x, i) => ys[i] - fit.predict(x));
  const m = median(residuals);
  const sigma = robustSigma(residuals) || 0;
  if (sigma === 0) return { fit, switched: false };
  const k = 3;
  const survivors = { xs: [], ys: [] };
  for (let i = 0; i < xs.length; i++) {
    if (Math.abs(residuals[i] - m) <= k * sigma) { survivors.xs.push(xs[i]); survivors.ys.push(ys[i]); }
  }
  const droppedCount = xs.length - survivors.xs.length;
  if (droppedCount === 0 || survivors.xs.length < 3) return { fit, switched: false };
  let refit;
  try { refit = fitLogisticCurve(survivors.xs, survivors.ys); } catch (e) { return { fit, switched: false }; }
  refit.r2 = computeR2(xs, ys, refit.predict); // report R^2 on the FULL data, same convention as the plain fit
  return {
    fit: refit,
    switched: true,
    reason: `${droppedCount} point(s) sat beyond ${k} robust sigmas from the median residual — refit via MAD pre-trim, excluding ${droppedCount === 1 ? 'it' : 'them'}`,
  };
}

// KNN prediction interval: instead of the usual exact/bootstrap CI (which
// describes uncertainty in the FIT LINE itself), this estimates a spread
// for a single NEW predicted point by looking at how wrong the model was
// on the K training points whose X was closest to the query X, weighted by
// an exponential distance decay. Local residual spread this way can widen
// appropriately in a region the model fits poorly even if the overall fit
// looks fine on average elsewhere. K/rate are NOT the K=4/rate=2.3 values
// used elsewhere in this project's analysis scripts — those were fit for a
// different task (smoothing a corpus of p-values, not a single dataset's
// residuals). These defaults (K=12, rate=3.5) instead come from a
// leave-one-out calibration sweep across every usable dataset in this
// project's own corpus (pooled, each dataset's error measured relative to
// its own std(y) so no single large-scale dataset dominates): the
// narrowest interval (avg width 1.18x std(y)) that still holds the nominal
// 95% coverage (95.3% empirically). Still exposed as adjustable — this is
// a reasonable general default for THIS corpus, not a universal constant.
function knnPredictionInterval(xs, ys, predict, queryX, { K = 12, rate = 3.5, confidence = 0.95 } = {}) {
  const n = xs.length;
  const kUse = Math.min(K, n);
  if (kUse < 2) return null;
  const residuals = xs.map((x, i) => ({ r: ys[i] - predict(x), dist: Math.abs(x - queryX) }));
  residuals.sort((a, b) => a.dist - b.dist);
  const neigh = residuals.slice(0, kUse);
  const maxD = Math.max(...neigh.map(x => x.dist), 1e-12);
  const w = neigh.map(x => Math.exp(-rate * x.dist / maxD));
  const sumW = w.reduce((a, b) => a + b, 0);
  const weightedMean = neigh.reduce((s, x, i) => s + w[i] * x.r, 0) / sumW;
  const weightedVar = neigh.reduce((s, x, i) => s + w[i] * (x.r - weightedMean) ** 2, 0) / sumW;
  const weightedSd = Math.sqrt(weightedVar);
  const df = Math.max(1, kUse - 1);
  const tCrit = studentTInverseCDF(1 - (1 - confidence) / 2, df);
  const pointEstimate = predict(queryX) + weightedMean; // bias-correct using nearby residuals
  return {
    pointEstimate,
    lower: pointEstimate - tCrit * weightedSd,
    upper: pointEstimate + tCrit * weightedSd,
    K: kUse, rate, confidence,
  };
}

const Regression = {
  arrayMin, arrayMax, recommendedHistogramBins, findOptimalHistogramBins,
  bootstrapParameterCI, movingBlockBootstrapParameterCI, wildBootstrapParameterCI, chooseParameterCI, formatParameterCIs,
  genericWeightedRefit, weightedReplicate,
  estimateLag1Autocorrelation, ar1InflateExactCI, widenParameterCIWithFloor,
  ar1CoverageCorrectionFactor, inflateParameterCIAroundCenter,
  runRegression, formatEquation,
  fitProportional, formatProportionalEquation,
  fitMultiLinear, fitMultiLeastTrimmedSquares, fitMultiExponential,
  formatMultiLinearEquation, formatMultiExponentialEquation, computeR2Multi,
  fitLogisticCurve, formatLogisticEquation,
  fitLogisticProbability, logLoss,
  fitLogisticOffset, formatLogisticOffsetEquation,
  fitPiecewiseConstantLinear, formatPiecewiseEquation,
  fitNaturalCubicSpline, formatSplineEquation,
  fitIsotonic, formatIsotonicEquation,
  fitKernelRegression, formatKernelEquation,
  fitExponential, formatExponentialEquation,
  fitTrimmedExponential,
  fitTrimmedLogisticCurve,
  fitExponentialOffset, formatExponentialOffsetEquation,
  fitPower, formatPowerEquation,
  fitPowerOffset, formatPowerOffsetEquation,
  fitLogarithmic, formatLogarithmicEquation,
  fitReciprocal, formatReciprocalEquation,
  fitSinusoidal, formatSinusoidalEquation,
  fitTheilSen, fitRANSAC, fitRANSACFixedFraction, fitHuber, fitLeastMedianSquares, fitLeastTrimmedSquares,
  fitTukeyBiweight, fitAndrewsSine, fitHampel, formatLinearEquation,
  fitPersistence, formatPersistenceEquation,
  fitPersistenceDrift, formatPersistenceDriftEquation,
  fitAR1, formatAR1Equation,
  fitAR2, formatAR2Equation,
  fitARMA11, formatARMA11Equation,
  fitARIMA, formatARIMAEquation, fitARIMAAuto,
  fitSARIMA, formatSARIMAEquation, fitSARIMAAuto, detectSeasonalPeriod,
  fitProphetStyle, formatProphetEquation, fitProphetStyleAuto,
  computeR2, computeR2FullVariance, computeR2Sequential, computeR2FullVarianceSequential, computeR2Weighted, computeR2MultiWeighted,
  shapiroWilkTest, shapiroWilkLooksNonNull, whiteLooksNonNull, durbinWatsonLooksNonNull, SW_P_STAR, WHITE_P_STAR, DW_P_STAR, durbinWatsonTest, whiteTest, knnPredictionInterval,
  welchTTest, recenteredBootstrapMeanDiffTest, grubbsOutlierTest, yuenTrimmedMeansTest, twoSampleMeanTest, formatTwoSampleMeanTest,
  fDistributionCDF, overallFTest, formatFTest,
  zScoreStandardize, normalQuantileStandardize, standardizeForImportance, sampleSkewness,
  fitRidge, fitElasticNet, fitPenalizedAuto, lassoLambdaMax, fitRegularizedPolynomial,
  fitMultinomialLogistic,
  chooseErrorDistribution, formatErrorDistributionReport, errorDistributionOutlierBounds, maybeRobustRefit, maybeRobustRefitLogLinear, fitRobustPolynomial,
  maybeRobustRefitMultiExponential, maybeRobustRefitProportional, maybeRobustRefitLogistic,
};
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Regression;
} else {
  window.Regression = Regression;
}
