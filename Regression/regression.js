// Fits a polynomial regression (degree 1 = straight line, 2 = curve with one
// bend, 3 = curve with two bends, etc.) to X/Y data using least squares.

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

// Fits y = c0 + c1*x + c2*x^2 + ... + cDegree*x^degree
function fitPolynomial(xs, ys, degree) {
  const n = xs.length;
  const terms = degree + 1;

  const A = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const b = new Array(terms).fill(0);

  for (let i = 0; i < n; i++) {
    const powers = [1];
    for (let p = 1; p <= degree * 2; p++) powers.push(powers[p - 1] * xs[i]);
    for (let r = 0; r < terms; r++) {
      for (let c = 0; c < terms; c++) A[r][c] += powers[r + c];
      b[r] += powers[r] * ys[i];
    }
  }

  const coeffs = solveLinearSystem(A, b);

  const predict = x => coeffs.reduce((sum, c, p) => sum + c * Math.pow(x, p), 0);

  const yMean = ys.reduce((a, v) => a + v, 0) / n;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - predict(xs[i])) ** 2;
    ssTot += (ys[i] - yMean) ** 2;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { degree, coeffs, r2, predict };
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
function runRegression(xs, ys, { degree = null, maxDegree = 3 } = {}) {
  if (xs.length !== ys.length) throw new Error('X and Y must have the same number of values.');
  if (xs.length < 2) throw new Error('Need at least 2 data points.');

  if (degree !== null) {
    if (xs.length < degree + 1) throw new Error(`Need at least ${degree + 1} points to fit a degree-${degree} curve.`);
    return fitPolynomial(xs, ys, degree);
  }

  let best = null;
  for (let d = 1; d <= Math.min(maxDegree, xs.length - 1); d++) {
    const fit = fitPolynomial(xs, ys, d);
    if (!best || fit.r2 > best.r2 + 0.001) best = fit; // require a real improvement, not noise
  }
  return best;
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

  const yScale = Math.max(...ys.map(Math.abs)) || 1;
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

  if (linXs.length < 2) throw new Error(`Need at least 2 points that aren't equal to the constant (${constantValue}) to fit a line through.`);

  const lineFit = fitPolynomial(linXs, linYs, 1);
  const activeMin = Math.min(...linXs);
  const activeMax = Math.max(...linXs);

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

// Fits y = a * e^(b*x) — steady growth/decay by a constant percentage each
// step. Linearized as ln(y) = ln(a) + b*x, so it reuses the straight-line
// fit above; only works when every Y is positive (e^anything is never <= 0).
function fitExponential(xs, ys) {
  if (ys.some(y => y <= 0)) throw new Error('Exponential fit needs every Y value to be greater than 0.');
  const lineFit = fitPolynomial(xs, ys.map(Math.log), 1);
  const a = Math.exp(lineFit.coeffs[0]);
  const b = lineFit.coeffs[1];
  const predict = x => a * Math.exp(b * x);
  return { type: 'exponential', a, b, r2: computeR2(xs, ys, predict), predict };
}

// Fits y = a * x^b — a constant-percentage relationship between X and Y
// (common for scaling laws). Linearized as ln(y) = ln(a) + b*ln(x); needs
// every X and Y to be positive.
function fitPower(xs, ys) {
  if (xs.some(x => x <= 0)) throw new Error('Power fit needs every X value to be greater than 0.');
  if (ys.some(y => y <= 0)) throw new Error('Power fit needs every Y value to be greater than 0.');
  const lineFit = fitPolynomial(xs.map(Math.log), ys.map(Math.log), 1);
  const a = Math.exp(lineFit.coeffs[0]);
  const b = lineFit.coeffs[1];
  const predict = x => a * Math.pow(x, b);
  return { type: 'power', a, b, r2: computeR2(xs, ys, predict), predict };
}

// Fits y = a * ln(x) + b — fast initial change that levels off, without
// ever fully plateauing the way the S-curve does. Needs every X positive.
function fitLogarithmic(xs, ys) {
  if (xs.some(x => x <= 0)) throw new Error('Logarithmic fit needs every X value to be greater than 0.');
  const lineFit = fitPolynomial(xs.map(Math.log), ys, 1);
  const b = lineFit.coeffs[0];
  const a = lineFit.coeffs[1];
  const predict = x => a * Math.log(x) + b;
  return { type: 'logarithmic', a, b, r2: computeR2(xs, ys, predict), predict };
}

function formatExponentialEquation({ a, b }) {
  return `${a.toFixed(4)} * e^(${b.toFixed(4)}*x)`;
}
function formatPowerEquation({ a, b }) {
  return `${a.toFixed(4)} * x^${b.toFixed(4)}`;
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
  const xnSpan = Math.max(...xn) - Math.min(...xn) || 1;

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

const Regression = {
  runRegression, formatEquation,
  fitLogisticCurve, formatLogisticEquation,
  fitPiecewiseConstantLinear, formatPiecewiseEquation,
  fitExponential, formatExponentialEquation,
  fitPower, formatPowerEquation,
  fitLogarithmic, formatLogarithmicEquation,
  fitReciprocal, formatReciprocalEquation,
  fitSinusoidal, formatSinusoidalEquation,
  computeR2,
};
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Regression;
} else {
  window.Regression = Regression;
}
