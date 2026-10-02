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
    return { estimate: c, se, lower: c - tCritical * se, upper: c + tCritical * se };
  });
  // "coeffs" (not coeffCIs) to match bootstrapParameterCI's result shape —
  // formatParameterCIs renders both the same way regardless of which one
  // produced them.
  return { exact: true, df, tCritical, params: {}, coeffs: coeffs_ };
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

function formatMultiExponentialEquation({ a, coeffs }, varNames) {
  const terms = coeffs.map((c, j) => `${c.toFixed(4)}*${varNames[j]}`);
  return `${a.toFixed(4)} * e^(${terms.join(' + ')})`;
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
  if (xs.some(x => x <= 0)) throw new Error('Logarithmic fit needs every X value to be greater than 0.');
  const w = weights || new Array(xs.length).fill(1);
  // Plain OLS with ln(x) as the input feature — unlike fitExponential/
  // fitPower, Y itself is never transformed, so lineFit's CI needs no
  // exp() back-transform at all, just relabeling (coeffs[0] is b here,
  // coeffs[1] is a — see the assignments right below).
  const lineFit = fitPolynomial(xs.map(Math.log), ys, 1, w);
  const b = lineFit.coeffs[0];
  const a = lineFit.coeffs[1];
  const predict = x => a * Math.log(x) + b;
  const ci = lineFit.ci && {
    exact: true, df: lineFit.ci.df, tCritical: lineFit.ci.tCritical,
    params: {
      b: { lower: lineFit.ci.coeffs[0].lower, upper: lineFit.ci.coeffs[0].upper },
      a: { lower: lineFit.ci.coeffs[1].lower, upper: lineFit.ci.coeffs[1].upper },
    },
  };
  return { type: 'logarithmic', a, b, r2: computeR2Weighted(xs, ys, predict, w), predict, ci };
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
function formatLogarithmicEquation({ a, b }) {
  const sign = b >= 0 ? '+' : '-';
  return `${a.toFixed(4)} * ln(x) ${sign} ${Math.abs(b).toFixed(4)}`;
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

  const predict = x => {
    const steps = Math.max(0, Math.round((x - lastX) / avgStep));
    let y = lastY;
    for (let s = 0; s < steps; s++) y = c + phi * y;
    return y;
  };

  // FIT r2 is measured on the one-step-ahead relationship (y_now vs.
  // predicted from y_lag) — the thing AR(1) actually fits — not on
  // predict(x)'s multi-step recursion, which isn't what least squares
  // minimized.
  const oneStepPredict = ys.map((_, i) => (i === 0 ? null : c + phi * ys[i - 1])).slice(1);
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

  const predict = x => {
    const steps = Math.max(0, Math.round((x - lastX) / avgStep));
    let y2 = prevY, y1 = lastY;
    for (let s = 0; s < steps; s++) {
      const next = c + phi1 * y1 + phi2 * y2;
      y2 = y1; y1 = next;
    }
    return y1;
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

  // Forecasting an MA term past one step uses e=0 (its expected value) —
  // the last known residual only affects the very first predicted step.
  const predict = x => {
    const steps = Math.max(0, Math.round((x - lastX) / avgStep));
    let y = lastY, e = lastE;
    for (let s = 0; s < steps; s++) {
      y = c + phi * y + theta * e;
      e = 0;
    }
    return y;
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
// see that test file for the full swept numbers.
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
    let fit;
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
function wildBootstrapParameterCI(fitFn, xs, ys, predict, { iterations = 150, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 5) return null;
  const residuals = xs.map((x, i) => ys[i] - predict(x));

  const random = createSeededRandom(seed);
  const scalarSamples = {};
  const coeffSamples = [];
  const SKIP_KEYS = new Set(['type', 'r2', 'logLoss', 'inlierCount', 'trimmedCount', 'totalCount', 'coeffs', 'predict', 'degree']);
  let successes = 0;

  for (let i = 0; i < iterations; i++) {
    const rys = new Array(n);
    for (let j = 0; j < n; j++) {
      const w = random() < 0.5 ? -1 : 1;
      rys[j] = predict(xs[j]) + residuals[j] * w;
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
    params[key] = percentileCI(scalarSamples[key]);
  }
  const coeffs = coeffSamples.length
    ? coeffSamples.map(vals => (vals && vals.length >= minNeeded) ? percentileCI(vals) : null)
    : null;

  return { params, coeffs, resamples: successes, requested: iterations, method: 'wild-bootstrap' };
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
function chooseParameterCI(fitFn, xs, ys, predict, exactOrPlainCI, { iterations = 150, seed = RNG_SEED } = {}) {
  const n = xs.length;
  if (n < 8) return exactOrPlainCI; // too little data for DW/White to mean anything
  const residuals = xs.map((x, i) => ys[i] - predict(x));
  const dw = durbinWatsonTest(residuals);
  const wt = whiteTest(xs, residuals);
  if (dw && dw.pValue < 0.1) {
    const ci = movingBlockBootstrapParameterCI(fitFn, xs, ys, { iterations, seed });
    if (ci) return { ...ci, gateReason: 'autocorrelated residuals (Durbin-Watson)' };
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
  for (const key of Object.keys(ci.params || {})) {
    const { lower, upper } = ci.params[key];
    lines.push(`    ${key}: ${lower.toFixed(4)} to ${upper.toFixed(4)}`);
  }
  if (ci.coeffs) {
    ci.coeffs.forEach((c, idx) => {
      if (c) lines.push(`    coeffs[${idx}]: ${c.lower.toFixed(4)} to ${c.upper.toFixed(4)}`);
    });
  }
  return '\n\n' + lines.join('\n');
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
// like, e.g. the Anderson-Darling fix elsewhere in this file). What it DOES
// support is a secondary, SIZE-ADJUSTED reading alongside the raw p-value:
// regress logit(p) on log(n) across the corpus, then remove that fitted
// trend from any new dataset's p-value before interpreting it — telling
// you "is this MORE or LESS normal-looking than a typical dataset of
// THIS SIZE", rather than conflating sample size with normality. Compared
// against a local (KNN/LOESS-style) version of the same idea and a
// straightforward n-tertile stratification (this session): the simple
// global detrend below removed the measured bias (rho 0.57 -> 0.01) at
// least as well as the fancier alternatives, and more reliably (the local
// KNN version frequently collapsed to a degenerate fit on this corpus).
// Calibration constants below come from a one-time OLS fit of this
// project's own 35-dataset corpus (n ranging 15-1334) and are not
// recomputed per-request.
const SW_SIZE_ADJUST = { slope: -4.582411782909428, meanLogN: 4.117442841468619, interceptAtMeanLogN: 10.20358343306811 };
function logit(p) { const v = Math.min(1 - 1e-9, Math.max(1e-9, p)); return Math.log(v / (1 - v)); }
function invLogit(x) { return 1 / (1 + Math.exp(-x)); }
function shapiroWilkSizeAdjustedPValue(rawP, n) {
  const adj = SW_SIZE_ADJUST;
  const adjustedLogit = logit(rawP) - adj.slope * (Math.log(n) - adj.meanLogN);
  return invLogit(adjustedLogit);
}

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

// Compares all three by BIC (k*ln(n) - 2*logLik, lower is better) — BIC
// over AIC here because the question is "which distribution is really
// generating this", a model-identification question BIC is designed for,
// not "which predicts best" (AIC's usual framing). A difference of 2-6 is
// "positive" evidence, 6-10 "strong", >10 "very strong" (standard BIC
// evidence-strength bands) for whichever model has the lower score.
function chooseErrorDistribution(residuals) {
  const n = residuals.length;
  if (n < 8) return null;
  const candidates = [gaussianErrorFit(residuals), laplaceErrorFit(residuals), studentTErrorFit(residuals)]
    .filter(Boolean)
    .map(c => ({ ...c, bic: c.k * Math.log(n) - 2 * c.logLik }));
  if (!candidates.length) return null;
  candidates.sort((a, b) => a.bic - b.bic);
  const best = candidates[0];
  const gaussian = candidates.find(c => c.name === 'Gaussian');
  return { candidates, best, deltaBICvsGaussian: gaussian ? gaussian.bic - best.bic : 0 };
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
    return `\n\n  Error distribution check: Gaussian is still the best (or no other candidate clearly beats it, ΔBIC=${deltaBICvsGaussian.toFixed(1)}).`;
  }
  const strength = deltaBICvsGaussian > 10 ? 'very strong' : deltaBICvsGaussian > 6 ? 'strong' : 'positive';
  const paramDesc = best.name === 'Student-t' ? ` (df=${best.params.nu}, heavier tails than Gaussian)` : '';
  // Worded to stay accurate whether this check is running on the original
  // fit's residuals OR an already-switched robust refit's residuals (this
  // function is called again, unconditionally, on whatever the FINAL
  // displayed fit turns out to be) — never claims a refit "was kept as-is"
  // or "wasn't tried", since by the time this prints, that decision is
  // already made elsewhere and may have gone either way.
  const refitNote = deltaBICvsGaussian >= ROBUST_REFIT_DELTA_BIC_THRESHOLD
    ? ` This meets the ${ROBUST_REFIT_DELTA_BIC_THRESHOLD}-point bar this tool uses for an automatic robust refit (confirmed by synthetic testing to need this strong an evidence level before refitting reliably helps).`
    : ` This is below the ${ROBUST_REFIT_DELTA_BIC_THRESHOLD}-point bar this tool uses for an automatic robust refit (synthetic testing found weaker evidence than this doesn't reliably justify one).`;
  return `\n\n  Error distribution check: ${best.name}${paramDesc} fits these residuals better than Gaussian` +
         ` (ΔBIC=${deltaBICvsGaussian.toFixed(1)}, ${strength} evidence).${refitNote}`;
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
  if (!transform) return { fit, switched: false };
  // Transformed-space residual: toY(y) - toY(predict(x)), the space the
  // underlying line fit (and hence any Gaussian/Laplace/Student-t
  // assumption about its errors) actually operates in.
  const transformedResiduals = xs.map((x, i) => transform.toY(ys[i]) - transform.toY(fit.predict(x)));
  const errDist = chooseErrorDistribution(transformedResiduals);
  if (!errDist || errDist.best.name === 'Gaussian' || errDist.deltaBICvsGaussian < ROBUST_REFIT_DELTA_BIC_THRESHOLD) {
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
    reason: `${errDist.best.name} errors detected in the fit's natural (transformed) scale (ΔBIC=${errDist.deltaBICvsGaussian.toFixed(1)} vs Gaussian, >= the ${ROBUST_REFIT_DELTA_BIC_THRESHOLD}-point bar) — refit via ${method} instead of OLS`,
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
  arrayMin, arrayMax,
  bootstrapParameterCI, movingBlockBootstrapParameterCI, wildBootstrapParameterCI, chooseParameterCI, formatParameterCIs,
  runRegression, formatEquation,
  fitProportional, formatProportionalEquation,
  fitMultiLinear, fitMultiLeastTrimmedSquares, fitMultiExponential,
  formatMultiLinearEquation, formatMultiExponentialEquation, computeR2Multi,
  fitLogisticCurve, formatLogisticEquation,
  fitLogisticProbability, logLoss,
  fitLogisticOffset, formatLogisticOffsetEquation,
  fitPiecewiseConstantLinear, formatPiecewiseEquation,
  fitExponential, formatExponentialEquation,
  fitTrimmedExponential,
  fitTrimmedLogisticCurve,
  fitExponentialOffset, formatExponentialOffsetEquation,
  fitPower, formatPowerEquation,
  fitPowerOffset, formatPowerOffsetEquation,
  fitLogarithmic, formatLogarithmicEquation,
  fitReciprocal, formatReciprocalEquation,
  fitSinusoidal, formatSinusoidalEquation,
  fitTheilSen, fitRANSAC, fitHuber, fitLeastMedianSquares, fitLeastTrimmedSquares,
  fitTukeyBiweight, fitAndrewsSine, formatLinearEquation,
  fitPersistence, formatPersistenceEquation,
  fitPersistenceDrift, formatPersistenceDriftEquation,
  fitAR1, formatAR1Equation,
  fitAR2, formatAR2Equation,
  fitARMA11, formatARMA11Equation,
  computeR2, computeR2FullVariance, computeR2Weighted, computeR2MultiWeighted,
  shapiroWilkTest, shapiroWilkSizeAdjustedPValue, durbinWatsonTest, whiteTest, knnPredictionInterval,
  chooseErrorDistribution, formatErrorDistributionReport, maybeRobustRefit, maybeRobustRefitLogLinear,
};
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Regression;
} else {
  window.Regression = Regression;
}
