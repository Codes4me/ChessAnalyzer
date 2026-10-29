// C&RT (classification and regression trees) feasibility check. This
// project already has fitPiecewiseConstantLinear (flat-then-linear-then-
// flat, 2 fixed breakpoints) -- CART generalizes that idea to arbitrary
// recursive binary splits, each one picked to minimize SSE, same greedy
// algorithm Breiman et al.'s original CART and every textbook
// implementation (rpart, sklearn's DecisionTreeRegressor) use.
//
// Regression tree, single predictor (x): at each node, try every
// candidate split point, pick the one that minimizes total SSE of the two
// child means; recurse. Stopping rule (depth + minimum leaf size) chosen
// via a FIT/CHECK split carved out of TRAIN only, same convention
// selectByExtrapolation already uses for every other candidate -- never
// touching TEST during that choice.
//
// Two-part validation, same as every other new method this session:
// (1) synthetic data -- does it recover a KNOWN true step function
// cleanly, and does it avoid "finding" spurious splits on pure noise
// (overfitting check, train vs. held-out R² on null data); (2) real
// corpus backtest against the existing candidates already in
// model_candidates.js.

const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');
const { selectByExtrapolation } = require('./model_candidates');

function mulberry32(seed) { let state = seed >>> 0; return function () { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randNormal(r) { const u1 = Math.max(1e-12, r()), u2 = r(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function sse(ys) { if (!ys.length) return 0; const m = mean(ys); return ys.reduce((s, v) => s + (v - m) ** 2, 0); }
function medianOf(a) { const s = [...a].sort((x, y) => x - y); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

// Build a regression tree node. Points come in as {x, y} pairs already
// sorted by x. Returns a tree: leaf = {leaf:true, value}, internal =
// {leaf:false, splitX, left, right}.
function buildTree(points, { maxDepth, minLeafSize }, depth = 0) {
  const ys = points.map(p => p.y);
  if (depth >= maxDepth || points.length < 2 * minLeafSize) {
    return { leaf: true, value: mean(ys), n: points.length };
  }
  // Candidate splits: midpoints between consecutive distinct x values.
  let best = null;
  for (let i = minLeafSize; i <= points.length - minLeafSize; i++) {
    if (points[i].x === points[i - 1].x) continue; // don't split inside a tie
    const left = points.slice(0, i), right = points.slice(i);
    const cost = sse(left.map(p => p.y)) + sse(right.map(p => p.y));
    if (!best || cost < best.cost) best = { cost, i, splitX: (points[i - 1].x + points[i].x) / 2 };
  }
  if (!best) return { leaf: true, value: mean(ys), n: points.length };
  const parentCost = sse(ys);
  if (parentCost - best.cost < 1e-12) return { leaf: true, value: mean(ys), n: points.length }; // split gains nothing -- stop
  const left = points.slice(0, best.i), right = points.slice(best.i);
  return {
    leaf: false, splitX: best.splitX,
    left: buildTree(left, { maxDepth, minLeafSize }, depth + 1),
    right: buildTree(right, { maxDepth, minLeafSize }, depth + 1),
  };
}
function predictTree(tree, x) {
  let node = tree;
  while (!node.leaf) node = x < node.splitX ? node.left : node.right;
  return node.value;
}

// Fit function matching this project's existing candidate shape (fit(xs,ys)
// -> {predict, r2, ...}). Hyperparameters (maxDepth, minLeafSize) are
// chosen by trying a small grid and picking whichever does best on a
// FIT/CHECK split carved out of the training data itself -- same
// discipline as selectByExtrapolation, so the tree can't just memorize
// training noise and call it a result.
const DEPTH_GRID = [1, 2, 3, 4, 5];
const MINLEAF_GRID = [2, 3, 5, 8];
function fitCART(xs, ys, { fitFrac = 0.7 } = {}) {
  const n = xs.length;
  if (n < 10) throw new Error('Need at least 10 data points for a CART fit.');
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const sortedPoints = order.map(i => ({ x: xs[i], y: ys[i] }));

  const fitLen = Math.max(6, Math.floor(n * fitFrac));
  const fitPoints = sortedPoints.slice(0, fitLen); // NOTE: this is a random-order split by original index, not x-order, done below properly
  // Use original (row) order for the fit/check split (consistent with the
  // rest of this project's walk-forward-style convention), then re-sort
  // each half by x for tree-building (CART needs x-sorted points to find
  // split candidates; the fit/check boundary is still in ROW order/time).
  const half1Idx = Array.from({ length: fitLen }, (_, i) => i);
  const rowFitPoints = half1Idx.map(i => ({ x: xs[i], y: ys[i] })).sort((a, b) => a.x - b.x);
  const rowCheckXs = xs.slice(fitLen), rowCheckYs = ys.slice(fitLen);
  if (rowCheckXs.length < 3) {
    // Not enough left for a proper check split -- just pick a safe, small default.
    const tree = buildTree(sortedPoints, { maxDepth: 2, minLeafSize: 5 });
    return makeCartFit(tree, xs, ys, 2, 5);
  }

  let best = null;
  for (const maxDepth of DEPTH_GRID) {
    for (const minLeafSize of MINLEAF_GRID) {
      if (rowFitPoints.length < 2 * minLeafSize) continue;
      const tree = buildTree(rowFitPoints, { maxDepth, minLeafSize });
      const r2 = Regression.computeR2(rowCheckXs, rowCheckYs, x => predictTree(tree, x));
      if (!best || r2 > best.r2) best = { r2, maxDepth, minLeafSize };
    }
  }
  if (!best) { const tree = buildTree(sortedPoints, { maxDepth: 2, minLeafSize: 5 }); return makeCartFit(tree, xs, ys, 2, 5); }

  const finalTree = buildTree(sortedPoints, { maxDepth: best.maxDepth, minLeafSize: best.minLeafSize });
  return makeCartFit(finalTree, xs, ys, best.maxDepth, best.minLeafSize);
}
function makeCartFit(tree, xs, ys, maxDepth, minLeafSize) {
  const predict = x => predictTree(tree, x);
  return { type: `CART(depth<=${maxDepth}, minLeaf=${minLeafSize})`, tree, r2: Regression.computeR2(xs, ys, predict), predict };
}

// =====================================================================
// PART 1a: recovers a known true step function cleanly?
// =====================================================================
console.log('=== Part 1a: does CART recover a known true step function? ===\n');
console.log('True: y = 0 for x<30, y=10 for 30<=x<70, y=5 for x>=70, plus noise.\n');
{
  const n = 150;
  const rand = mulberry32(55555);
  const xs = Array.from({ length: n }, (_, i) => i * (100 / n));
  const trueF = x => x < 30 ? 0 : x < 70 ? 10 : 5;
  const ys = xs.map(x => trueF(x) + randNormal(rand) * 1.5);
  const fit = fitCART(xs, ys);
  console.log(`  Selected: ${fit.type}, in-sample R²=${fit.r2.toFixed(4)}`);
  console.log(`  predict(10)=${fit.predict(10).toFixed(2)} (true 0), predict(50)=${fit.predict(50).toFixed(2)} (true 10), predict(90)=${fit.predict(90).toFixed(2)} (true 5)`);
}

// =====================================================================
// PART 1b: overfitting check on PURE NULL (noise) data -- does the
// fit/check hyperparameter selection correctly keep the tree shallow
// instead of carving noise into spurious leaves?
// =====================================================================
console.log('\n=== Part 1b: overfitting check on pure noise (no real x-y relationship) ===\n');
{
  const TRIALS = 200;
  const trainR2s = [], heldOutR2s = [], depths = [];
  for (let t = 0; t < TRIALS; t++) {
    const rand = mulberry32(600000 + t);
    const n = 60;
    const xs = Array.from({ length: n }, (_, i) => i);
    const ys = xs.map(() => randNormal(rand) * 2); // y has NO real relationship to x
    // Genuinely held-out split: fitCART only ever sees TRAIN (its own
    // internal fit/check split happens INSIDE that, carved further from
    // train only) -- TEST stays untouched until scoring, unlike the first
    // version of this check, which passed the WHOLE dataset to fitCART
    // and then wrongly treated a slice of that same data as "held out".
    const trainLen = Math.floor(n * 0.7);
    const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);
    const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
    let fit; try { fit = fitCART(trainXs, trainYs); } catch (e) { continue; }
    const testR2 = Regression.computeR2(testXs, testYs, fit.predict);
    trainR2s.push(fit.r2); heldOutR2s.push(testR2);
    depths.push(parseInt(fit.type.match(/depth<=(\d+)/)[1], 10));
  }
  console.log(`  mean in-sample R² on pure noise: ${mean(trainR2s).toFixed(4)}  (higher = more overfitting to noise)`);
  console.log(`  mean held-out R² on pure noise:  ${mean(heldOutR2s).toFixed(4)}  (should be near/below 0 -- there's nothing real to predict)`);
  console.log(`  mean selected max depth: ${mean(depths).toFixed(2)}  (low = the fit/check selection correctly resisted finding spurious splits)`);
}

// =====================================================================
// PART 2: real corpus backtest
// =====================================================================
console.log('\n=== Part 2: real corpus -- does CART beat the existing candidates already in model_candidates.js? ===\n');
const { groups } = DataParser.parseGroups(fs.readFileSync(path.resolve('data.txt'), 'utf8'));
function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const n = xs.length;
  if (n < 20) return null;
  const trainLen = Math.floor(n * 0.7);
  const trainXs = xs.slice(0, trainLen), trainYs = ys.slice(0, trainLen);
  const testXs = xs.slice(trainLen), testYs = ys.slice(trainLen);
  if (testXs.length < 3) return null;

  let bestExisting = null;
  try {
    const sel = selectByExtrapolation(trainXs, trainYs, { fitFrac: 0.5, checkFrac: 0.2 });
    const existingFit = sel.candidate.fit(trainXs, trainYs);
    bestExisting = { name: sel.candidate.name, r2: Regression.computeR2(testXs, testYs, existingFit.predict) };
  } catch (e) { return null; }

  let cartFit; try { cartFit = fitCART(trainXs, trainYs); } catch (e) { return null; }
  const cartR2 = Regression.computeR2(testXs, testYs, cartFit.predict);
  return { title: group.title, n, existingName: bestExisting.name, existingR2: bestExisting.r2, cartName: cartFit.type, cartR2 };
}
const results = groups.map(evaluateDataset).filter(Boolean);
console.log(`${results.length} of ${groups.length} datasets evaluated.\n`);
let helped = 0, hurt = 0;
results.slice().sort((a, b) => (b.cartR2 - b.existingR2) - (a.cartR2 - a.existingR2)).forEach(r => {
  const delta = r.cartR2 - r.existingR2;
  if (delta > 0.01) helped++; else if (delta < -0.01) hurt++;
  console.log(`  ${r.title}  n=${r.n}  existing best=${r.existingName} (R²=${r.existingR2.toFixed(3)})  CART=${r.cartName} (R²=${r.cartR2.toFixed(3)})  delta=${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`);
});
console.log(`\nCART beat the existing best candidate (>0.01 R²) on ${helped}/${results.length}, lost on ${hurt}/${results.length}.`);
