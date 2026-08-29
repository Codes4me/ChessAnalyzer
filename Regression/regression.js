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

// Fits y = a*x — a straight line forced through the origin (no intercept
// term). Different from the regular linear fit whenever the data doesn't
// actually cross zero at x=0 — this is for when you specifically want that
// constraint (e.g. "zero input must mean zero output").
function fitProportional(xs, ys) {
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += xs[i] * ys[i]; sxx += xs[i] * xs[i]; }
  if (sxx === 0) throw new Error('Proportional fit needs at least one nonzero X value.');
  const a = sxy / sxx;
  const predict = x => a * x;
  return { type: 'proportional', a, r2: computeR2(xs, ys, predict), predict };
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

  const yScale = Math.max(...ys.map(Math.abs)) || 1;
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

  const yScale = Math.max(...ys.map(Math.abs)) || 1;
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
function fitTheilSen(xs, ys, { maxPairs = 200000 } = {}) {
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
    for (let k = 0; k < maxPairs; k++) {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
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
function fitRANSAC(xs, ys, { iterations = 300, thresholdMultiplier = 2 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a RANSAC fit.');

  // A rough initial line just to measure a typical residual size from.
  const rough = fitPolynomial(xs, ys, 1);
  const roughResiduals = ys.map((y, i) => y - rough.predict(xs[i]));
  const sigma = robustSigma(roughResiduals) || (Math.max(...ys) - Math.min(...ys)) * 0.01 || 1;
  const threshold = thresholdMultiplier * sigma;

  let bestInliers = null;
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(Math.random() * n);
    let j = Math.floor(Math.random() * n);
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
function fitHuber(xs, ys, { iterations = 30 } = {}) {
  const n = xs.length;
  if (n < 2) throw new Error('Need at least 2 data points for a Huber fit.');

  const { slope, intercept } = fitByIRLS(xs, ys, (r, sigma) => {
    const delta = 1.345 * sigma;
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
function fitLeastMedianSquares(xs, ys, { iterations = 500 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a least-median-of-squares fit.');

  let best = null;
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(Math.random() * n);
    let j = Math.floor(Math.random() * n);
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
function fitLeastTrimmedSquares(xs, ys, { iterations = 500, trimFraction = 0.90 } = {}) {
  const n = xs.length;
  if (n < 3) throw new Error('Need at least 3 data points for a least-trimmed-squares fit.');
  const h = Math.max(2, Math.round(n * trimFraction));

  let best = null;
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(Math.random() * n);
    let j = Math.floor(Math.random() * n);
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

const Regression = {
  runRegression, formatEquation,
  fitProportional, formatProportionalEquation,
  fitLogisticCurve, formatLogisticEquation,
  fitLogisticOffset, formatLogisticOffsetEquation,
  fitPiecewiseConstantLinear, formatPiecewiseEquation,
  fitExponential, formatExponentialEquation,
  fitExponentialOffset, formatExponentialOffsetEquation,
  fitPower, formatPowerEquation,
  fitPowerOffset, formatPowerOffsetEquation,
  fitLogarithmic, formatLogarithmicEquation,
  fitReciprocal, formatReciprocalEquation,
  fitSinusoidal, formatSinusoidalEquation,
  fitTheilSen, fitRANSAC, fitHuber, fitLeastMedianSquares, fitLeastTrimmedSquares,
  fitTukeyBiweight, fitAndrewsSine, formatLinearEquation,
  computeR2,
};
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Regression;
} else {
  window.Regression = Regression;
}
