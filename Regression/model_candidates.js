// The list of model types the "Auto (best by extrapolation check)" option
// tries, plus the selection logic itself — shared between backtest_models.js
// (the offline CLI script) and the /regression page, so both pick models
// exactly the same way.

// Named RegressionLib, not Regression — classic (non-module) <script src>
// tags share one global lexical scope in a browser, so a top-level `const
// Regression` here would collide with regression.js's own top-level `const
// Regression` and throw a duplicate-declaration SyntaxError on the page.
const RegressionLib = (typeof module !== 'undefined' && module.exports) ? require('./regression') : window.Regression;

// `minFit` is the fewest FIT-half points a candidate needs to even be
// attempted. `params` is roughly how many fitted coefficients the model
// has — used below to break near-ties in favor of the simpler shape.
// (A data-driven complexity measure, based on how much each model's fit R²
// actually outruns its check R² across a whole corpus of datasets, was
// tried as a replacement for this — it measurably hurt the aggregate
// median held-out R², so `params` stays the default; see selectByExtrapolation's
// `complexityOf` option below if you want to swap it back in for testing.
// Re-checked after the corpus grew from 37 to 64 datasets: still hurts.
// Median test R² across the corpus: -0.7367 with `params` (the default)
// vs. -1.1044 using each candidate's median fit-check R² gap, or -0.8823
// using the mean gap — both worse than just counting coefficients. The
// gap measure is also noisy in practice: R² is unbounded below, so a
// handful of badly-extrapolating datasets blow the *mean* gap up by many
// orders of magnitude for some candidates, and even the more robust
// *median* gap ranks plain `constant` as more "complex" than cubic —
// check R² swinging very negative on a volatile dataset looks identical
// to genuine overfitting under this measure, so it can't reliably tell
// the two apart.)
// Quartic/quintic are included deliberately rather than banned outright —
// earlier experiments showed adjusted R² selection alone lets them win on
// training data and then blow up on unseen data; the fit/check
// extrapolation test below is what actually keeps them honest.
// Model types are periodically re-checked with an empirical
// backward-elimination pass: starting from ALL candidate types (including
// ones previously removed), repeatedly find whichever single candidate's
// REMOVAL improves the aggregate median held-out test R² the most, drop
// it, and repeat until no further removal helps. Re-run after the corpus
// grew to 64 datasets and the checkLen >= params gate (below) was added:
// took the median from -0.9937 (all 24 types) to -0.2701 by dropping
// S-curve with offset, proportional (no intercept), exponential with
// offset, reciprocal, power, and cubic (degree 3) — each was more often
// winning ties on fit/check noise than actually extrapolating well to
// genuinely new data. Notably this run's result differs from the previous
// pass (which had dropped quadratic and kept cubic/power/exponential with
// offset instead) — the winning pool isn't fixed, it depends on the
// current corpus and selection rules, so re-run this periodically as the
// corpus grows rather than treating either result as final. All removed
// types stay available as manual dropdown options on the /regression page
// — this only affects the Auto-selection pool.
//
// Also tried adding a new "trimmed exponential" candidate (an exponential
// fit that drops the worst ~10% of points, same idea as Least Trimmed
// Squares) to this pool: it looked great in isolation (median test R²
// -2.59 -> -1.3ish when every eligible dataset was forced through
// exponential vs. trimmed exponential only), but inside the full pool it
// was the FIRST thing backward elimination removed — it was winning ties
// against other model types by chance more than by actually extrapolating
// better, which only shows up once it's competing against everything
// else, not in an isolated A/B test. Left out of CANDIDATES; still
// available as a manual dropdown option on the /regression page.
//
// That same re-run also caught a latent bug: backtest_models.js's final
// refit step wasn't wrapped in try/catch, so a dataset whose FIT half
// (first 50%) is all-positive but whose CHECK half (next 20%) dips to
// zero or negative could pass selection then crash once folded into the
// full 70% training pool a candidate like `exponential` gets refit on
// (only the FIT half is ever run through .fit() during selection — the
// CHECK half only goes through .predict(), which never validates
// anything). Fixed by treating a final-refit failure as skipped, same as
// a selection failure. That fix alone — before even considering trimmed
// exponential — changed which datasets `exponential` could legally win,
// which cascaded into a further pool improvement: dropping Least Trimmed
// Squares, Least Median of Squares, and linear (degree 1) took the median
// from -0.2701 to -0.2284.
// A flat "persistence (last value)" candidate (RegressionLib.fitPersistence)
// was tried here too — same idea as trimmed exponential above: it looked
// like a safety net against catastrophic extrapolation (median test R² held
// steadier, average improved a lot), but it got selected on 19 of 56
// datasets and several of those picks had bad held-out test R² (down to
// -157.9), because a 1-parameter flat line can win a tiny CHECK window (as
// few as 2-3 points) by noise alone, not real fit. Net effect: aggregate
// median test R² went from -0.2619 to -0.3663 — worse, for the same reason
// trimmed exponential was kept out. Left out of CANDIDATES; still available
// directly via Regression.fitPersistence for manual/baseline use.
//
// Re-run again after three unrelated pipeline changes landed together: the
// RNG bug fix (RANSAC/Theil-Sen/etc. were using unseeded Math.random(),
// making backward-elimination results themselves noisy — now deterministic
// via a seeded PRNG), the tolerance constant moving to C=3, and the new
// z_check gate (rejects a candidate whose CHECK-window residuals look like
// statistical outliers relative to its own FIT-residual spread — see
// selectByExtrapolation's zCheckThreshold). Also switched the scoring
// metric itself to computeR2FullVariance (denominator from the whole
// dataset's variance, not just the held-out slice's own — the old metric
// let a near-zero-variance test window turn a small absolute miss into an
// astronomically bad score). Starting from all 24 candidate types under
// this new setup: median test R² (full-variance) was 0.8166. Backward
// elimination dropped logarithmic (0.8166 -> 0.8511 — the single biggest
// swing of this whole re-run), S-curve with offset (-> 0.8792), and
// piecewise (constant+linear) (-> 0.8827), then stopped. The resulting
// 21-candidate pool beats the previous 15-candidate CANDIDATES list
// measured on this exact same new methodology: 0.8827 vs 0.8211.
//
// The interesting part: this pool ADDS BACK several types the *previous*
// backward-elimination pass had removed (proportional, linear (degree 1),
// cubic, exponential with offset, power, reciprocal, Least Median of
// Squares, Least Trimmed Squares) — they used to lose because they'd win
// ties on fit/check noise and then extrapolate badly with nothing to catch
// it. The z_check gate now catches exactly that failure mode directly, so
// these types can contribute real wins again instead of being excluded
// wholesale. This is a good illustration of why this pool isn't fixed —
// it depends on the whole selection pipeline around it, not just the
// corpus, so re-run this after any future change to the selection
// mechanism itself, not only after the corpus grows.
const CANDIDATES = [
  { name: 'constant', params: 1, minFit: 2, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 0 }) },
  { name: 'proportional (no intercept)', params: 1, minFit: 2, fit: (xs, ys) => RegressionLib.fitProportional(xs, ys) },
  { name: 'linear (degree 1)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 1 }) },
  { name: 'quadratic (degree 2)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.runRegression(xs, ys, { degree: 2 }) },
  { name: 'S-curve (logistic)', params: 3, minFit: 3, fit: (xs, ys) => RegressionLib.fitLogisticCurve(xs, ys) },
  { name: 'exponential', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitExponential(xs, ys) },
  { name: 'exponential with offset', params: 3, minFit: 4, fit: (xs, ys) => RegressionLib.fitExponentialOffset(xs, ys) },
  { name: 'power', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitPower(xs, ys) },
  { name: 'power with offset', params: 3, minFit: 4, fit: (xs, ys) => RegressionLib.fitPowerOffset(xs, ys) },
  { name: 'reciprocal', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitReciprocal(xs, ys) },
  // cubic, quartic, quintic (degree 3/4/5 polynomials) and sinusoidal were
  // removed after a resample-based bootstrap check (3000 bootstrap draws of
  // this corpus, "least useful" candidates re-ranked fresh inside EACH
  // resample rather than fixed once from the full corpus, to avoid
  // selecting winners/losers on the same data used to score them). At
  // K=4 excluded, these four are excluded in 95-100% of resamples each and
  // the measured effect on average held-out test R² is EXACTLY zero in
  // every single resample (mean 0.00000, 95% CI [0.000, 0.000], 3000/3000
  // resamples) — they never actually win against the rest of the pool, so
  // removing them changes nothing. At K=5 the same procedure starts
  // sweeping in genuinely useful models by chance (quintic/RANSAC/Theil-Sen
  // volatility, mean effect ~0 but far noisier, CI straddling zero), and at
  // K=8 it's measurably net-negative (mean -0.021) — so K=4 is the largest
  // cut that's unambiguously free, and this pool deliberately stops there
  // rather than extending further.
  { name: 'Theil-Sen (robust line)', params: 2, minFit: 2, fit: (xs, ys) => RegressionLib.fitTheilSen(xs, ys) },
  { name: 'RANSAC (robust line)', params: 2, minFit: 3, fit: (xs, ys) => RegressionLib.fitRANSAC(xs, ys) },
  { name: 'Least Median of Squares (robust line)', params: 2, minFit: 3, fit: (xs, ys) => RegressionLib.fitLeastMedianSquares(xs, ys) },
  // Huber, Least Trimmed Squares, Tukey biweight and Andrews' sine (all
  // robust-line methods) were removed from this pool after a redundancy
  // pass: their per-dataset CHECK R² is >=0.95 correlated with at least one
  // other candidate already in the pool (Huber/Tukey/Andrews all >=0.99
  // correlated with each other and with plain `linear`; LTS >=0.99
  // correlated with `linear` and Theil-Sen) across the 55 datasets with a
  // usable score for both — i.e. they're not adding a genuinely different
  // shape, just another near-copy of a line fit. Checked each removal
  // individually against this file's real backtest before cutting: Huber,
  // Tukey biweight and Andrews' sine never actually win a single dataset in
  // this corpus at all (removing any of them, alone or together, leaves
  // both median and average test R² completely unchanged), so they're pure
  // dead weight. Least Trimmed Squares does occasionally win but removing
  // it is a wash (average test R² 0.45553 -> 0.45554, noise-level).
  // Two other candidates flagged by the same correlation pass — `linear
  // (degree 1)` (>=0.95 correlated with LTS, Huber, Theil-Sen and Tukey)
  // and RANSAC (>=0.95 correlated with Theil-Sen) — were deliberately NOT
  // removed despite scoring "worse" by a same-shape comparison: removing
  // either one alone measurably hurts this corpus's average test R² (linear
  // -> 0.4555 to 0.4267; RANSAC -> 0.4555 to 0.4482), so whatever real
  // signal they contribute isn't actually redundant here even though their
  // CHECK R² correlates with a similar-shaped neighbor. High correlation
  // between two models' scores is necessary but not sufficient for one of
  // them to be safely cuttable — always confirm on the real backtest before
  // removing, not just on the correlation number.
  // Added after finding strong positive autocorrelation (Durbin-Watson well
  // below 1) in the residuals of this corpus's worst-extrapolating
  // datasets — none of the shapes above model serial structure directly,
  // they're all plain y=f(x) curves. AR(1) targets that gap. Its raw OLS phi
  // estimate is dangerously unstable on the small FIT windows selection uses
  // (predict() recurses, so a poorly-estimated phi near 1 diverges
  // explosively), so fitAR1 shrinks phi toward 0 by default (see its own
  // comment in regression.js) — validated with that shrinkage in place: adds
  // this file's best median test R² found so far (0.9259, vs 0.8827 without
  // it), including a genuine fix on "Average high and low temperature by
  // day" (AR(1) now wins there, R²=0.237, vs the previous `power` pick's
  // -0.282).
  { name: 'AR(1)', params: 2, minFit: 3, fit: (xs, ys) => RegressionLib.fitAR1(xs, ys) },
  // AR(2) (fitAR2 in regression.js) was tried here and rejected: on this
  // corpus it never wins a single dataset at the default shrinkage (=1.5,
  // matching AR(1)'s), and sweeping shrinkage from 2 to 30 never helps
  // either — it stays exactly at AR(1)'s numbers until shrinkage=8, where it
  // starts winning one dataset it shouldn't (median full-variance test R²
  // drops from 0.9259 to 0.9207). The two remaining worst performers
  // ("Bachelor degree men salary", "traffic flow over time") were checked
  // directly: AR(2)'s CHECK-window R² there is close to AR(1)'s (both bad),
  // never close to the actual winner (S-curve/linear-cluster and sinusoidal
  // respectively) — their problem isn't missing lag structure, so adding a
  // second AR lag doesn't reach them. Kept as a library function in case a
  // future corpus addition has genuinely richer serial structure, but left
  // out of the pool.
  //
  // ARMA(1,1) (fitARMA11 in regression.js, fit via the Hannan-Rissanen
  // two-step OLS method) was tried too and also rejected: at AR(1)'s own
  // shrinkage (1.5) it does win 2 datasets in the pool, but it drags the
  // aggregate average test R² down (0.4555 -> 0.4462) for no median gain,
  // and at shrinkage >= 5 it stops winning anything at all — never a net
  // improvement at any setting tried. On the two hardest datasets directly,
  // its CHECK-window R² is worse than plain AR(1)'s, not better. Also kept
  // as a library function, also left out of the pool.
  //
  // Alternative priors for AR(1)'s existing shrinkage were swept too
  // (shrink-toward-a-nonzero-target, a hard |phi| clip instead of
  // shrinkage, shrink-then-clip) — none beat the current shrink-toward-0,
  // k=1.5 setting on this corpus's median test R² (0.9259). One variant
  // (k=2.0) trades that median down to 0.9040 for a better average
  // (0.5094 vs 0.4555) — a real tradeoff, not a strict win, and the same
  // "narrow peak" already documented on fitAR1's own shrinkage comment in
  // regression.js. Not adopted; current default stands.
];

// Splits (xs, ys) into a FIT slice (the first `fitFrac` of the data) and a
// CHECK slice right after it (the next `checkFrac`), fits every candidate on
// FIT alone, and scores each one on CHECK — data just past what it was fit
// on. That score is itself the extrapolation penalty: a model that overfits
// FIT predicts CHECK badly and simply loses, no separate complexity formula
// needed. Returns the winning candidate plus its fit/check R² — NOT a
// ready-to-use model, since the caller decides what to re-fit it on
// (backtest_models.js re-fits on the fit+check data only, to leave a true
// final test set untouched; the /regression page re-fits on all the data,
// since there's no held-out set to protect there — just the best equation
// to hand back).
// Standard deviation, used only by the z_check gate below.
function stddev(nums) {
  const m = nums.reduce((s, v) => s + v, 0) / nums.length;
  return Math.sqrt(nums.reduce((s, v) => s + (v - m) ** 2, 0) / nums.length);
}

function selectByExtrapolation(xs, ys, { fitFrac = 0.5, checkFrac = 0.2, tolerance, toleranceConstant = 3, maxTolerance = 0.48, gateMultiplier = 1, zCheckThreshold = 10, derivativeThreshold = 5, derivativeNoiseK = 1, derivativeNoiseFloor = 0.05, complexityOf = (candidate) => candidate.params } = {}) {
  const n = xs.length;
  const fitLen = Math.floor(n * fitFrac);
  const checkEnd = Math.floor(n * (fitFrac + checkFrac));
  const checkLen = checkEnd - fitLen;

  if (fitLen < 2 || checkLen < 2) {
    throw new Error(`Need more data for a fit/check split — got ${n} rows.`);
  }

  const fitXs = xs.slice(0, fitLen), fitYs = ys.slice(0, fitLen);
  const checkXs = xs.slice(fitLen, checkEnd), checkYs = ys.slice(fitLen, checkEnd);

  // For the derivative gate below: how fast is each candidate moving right
  // at the boundary where extrapolation begins (the end of CHECK, right
  // before TEST)? The evaluation POINT stays at the end of CHECK — that's
  // genuinely where extrapolation into unseen data begins, regardless of
  // config. But the normalization SCALE (xRange/yRange) is computed from
  // FIT alone, not FIT+CHECK — using FIT+CHECK for the scale was a real bug:
  // the "/regression" page's "use full dataset" option calls this with
  // fitFrac=0.7, checkFrac=0.3 (vs. the 0.5/0.2 default), and since CHECK is
  // much bigger there, the combined FIT+CHECK range is bigger too — which
  // inflates the normalized derivative (it scales with xRange^2 for the
  // second derivative, xRange^3 for the third) enough that a genuinely
  // excellent quadratic fit (checkR2=0.998 on "Total Balance") got rejected
  // by this gate alone, leaving only `constant` (R²=0.0000) standing.
  // Keying the scale to FIT alone (which is unaffected by checkFrac) makes
  // the gate mean the same thing regardless of what fitFrac/checkFrac the
  // caller chooses, while leaving the well-tuned default-mode behavior
  // (fitFrac=0.5, checkFrac=0.2) essentially undisturbed, since the scale
  // was already dominated by FIT there.
  const checkPoolXs = xs.slice(0, checkEnd), checkPoolYs = ys.slice(0, checkEnd);
  const xRange = Math.max(...fitXs) - Math.min(...fitXs) || 1;
  const yRange = Math.max(...fitYs) - Math.min(...fitYs) || 1;
  const boundaryX = checkPoolXs[checkPoolXs.length - 1];
  const derivStep = xRange * 0.01 || 1e-6;

  // An autocorrelation-triggered tightening of the derivative gate was
  // tried here first (detect lag-1 autocorrelation in a linear-detrended
  // FIT residual, cap the derivative threshold at 3 instead of 5 when it's
  // high) and REVERTED. It looked like a clean win on this corpus's default
  // fitFrac=0.5/checkFrac=0.2 backtest, but broke "Total Balance" under the
  // /regression page's "use full dataset" mode (fitFrac=0.7, checkFrac=0.3)
  // — a genuinely well-fit quadratic (R²=0.998) got excluded because its
  // real-world noise happens to be autocorrelated, which autocorrelation
  // alone can't distinguish from a genuinely wrong shape chasing noise.
  //
  // Noise-adaptive tightening (this version) uses a better-targeted signal
  // instead: `noiseFrac` = 1 - R² of a plain low-order polynomial fit
  // (auto-degree up to 3) on the FIT window — how much of the FIT window's
  // own variance ISN'T explained by a simple trend, regardless of what that
  // trend's shape is. A clean dataset scores near 0 (threshold stays ~5,
  // unchanged); a noisy one scores higher and the cap tightens toward
  // `derivativeThreshold * derivativeNoiseFloor`. This targets the actual
  // problem (does this specific dataset have enough real noise that a
  // steep-looking extrapolation is more likely overfitting than genuine
  // curvature?) rather than a proxy (autocorrelation) that both well-fit
  // and badly-fit curves can share.
  //
  // k=1, floor=0.05 were chosen after a real sweep across 0/5/10/15/20%
  // synthetic Gaussian noise (added to every corpus dataset) plus a
  // resample-based bootstrap: k=1 keeps the real corpus's median test R²
  // within -0.004 to +0.004 at every noise level tested except 15% (-0.037,
  // still the smallest cost of any k tried there) while still capturing
  // most of the average-R² benefit at each level (e.g. avg 0.14 vs the
  // fixed threshold's 0.05 at 20% noise). Higher k values (2.4-8) looked
  // better on synthetic-sweep AVERAGE alone, including in an unconstrained
  // per-resample bootstrap (which favored k=3, +96.5% chance of beating no
  // adaptivity at all) — but average is dominated by rare catastrophic
  // outliers on this R²-is-unbounded-below metric (confirmed directly: one
  // tiny dataset hit a -49.5 BILLION test R² under 10% noise + a loose
  // cap), and checking those higher-k settings against real per-level
  // MEDIANS showed they cost real accuracy at 4 of 5 noise levels to win
  // big only at the noisiest one. k=1 was the one setting that stayed safe
  // (near-zero or positive median change) across every level tested, which
  // is why it was chosen over what the raw average/bootstrap alone favored.
  //
  // Fixes this session's worst dataset directly: "Bachelor degree men
  // salary" (real, natural noiseFrac ≈ 0.59, no synthetic noise involved)
  // goes from test R² -12.47 (exponential with offset, wrongly admitted at
  // the flat cap of 5) to -2.91 (quadratic, correctly favored once the cap
  // tightens) — a real improvement, though a stronger k (2.4) pushed this
  // one dataset even further (to -0.20 via Theil-Sen) at the cost of real
  // median accuracy elsewhere on the corpus; k=1 was chosen for that
  // corpus-wide safety, not because it maxes out this one dataset.
  let noiseFrac = 0;
  try {
    const simpleFit = RegressionLib.runRegression(fitXs, fitYs);
    noiseFrac = Math.max(0, Math.min(1, 1 - simpleFit.r2));
  } catch (e) { /* fall back to noiseFrac=0 (no tightening) if this fails */ }
  const noiseMultiplier = Math.max(derivativeNoiseFloor, Math.min(1, 1 - derivativeNoiseK * noiseFrac));
  const effectiveDerivativeThreshold = derivativeThreshold * noiseMultiplier;

  const allResults = [];
  for (const candidate of CANDIDATES) {
    if (fitLen < candidate.minFit) continue;
    // A check window smaller than the candidate's own parameter count can't
    // meaningfully validate it — a flexible model can pass close to a
    // handful of check points by coincidence (not because it actually
    // extrapolates), then diverge badly past them. Confirmed empirically:
    // requiring checkLen >= params took this file's aggregate median
    // held-out test R² from -0.7367 to -0.3926 (stricter multiples, 1.5x
    // and 2x params, made it worse — 1x is the sweet spot).
    if (checkLen < candidate.params * gateMultiplier) continue;
    try {
      const fit = candidate.fit(fitXs, fitYs);
      if (!Number.isFinite(fit.r2)) continue;
      // Plain computeR2(checkXs, checkYs, ...) re-centers on the CHECK
      // window's own mean, so a CHECK window that happens to have near-zero
      // (or exactly zero) variance — e.g. "Chess bot elo vs Result" where
      // every CHECK-window Y was exactly 1.0 — makes EVERY candidate's
      // checkR2 trivially 1.0 regardless of prediction quality, collapsing
      // selection to "pick the simplest candidate" with zero real signal.
      // Switching to a full-variance-style denominator for EVERY dataset
      // was tried and rejected — it changes checkR2's scale broadly enough
      // to break the toleranceConstant/z_check/derivative tuning done
      // against the old metric (corpus median collapsed 0.8827 -> 0.4453).
      // So this only substitutes the stable (train-pool-variance)
      // denominator in the specific degenerate case where the CHECK
      // window's own variance is too small to be a meaningful yardstick —
      // normal cases keep using the exact metric everything else was tuned
      // against.
      const checkYMean = checkYs.reduce((s, v) => s + v, 0) / checkYs.length;
      const checkSsTot = checkYs.reduce((s, v) => s + (v - checkYMean) ** 2, 0);
      const checkPoolYMean = checkPoolYs.reduce((s, v) => s + v, 0) / checkPoolYs.length;
      const checkPoolSsTot = checkPoolYs.reduce((s, v) => s + (v - checkPoolYMean) ** 2, 0);
      const checkR2 = checkSsTot < 1e-9 * (checkPoolSsTot || 1)
        ? RegressionLib.computeR2FullVariance(checkPoolYs, checkXs, checkYs, fit.predict)
        : RegressionLib.computeR2(checkXs, checkYs, fit.predict);
      if (!Number.isFinite(checkR2)) continue;

      // z_check: standardize each CHECK residual against how tightly this
      // candidate fit the FIT window in the first place. A candidate whose
      // CHECK predictions already look like statistical outliers relative
      // to its own FIT-residual spread is showing an early-warning sign of
      // poor generalization — this is a genuinely different signal from
      // checkR2 itself (confirmed empirically: a 2-point CHECK window can
      // give a deceptively decent checkR2 while still containing a residual
      // that's wildly inconsistent with the FIT fit's own noise level — see
      // the "Python Triangles" exponential-blowup case this was built to
      // catch). Used as a GATE (exclude, don't just penalize) — tested at
      // several thresholds; tight ones (z>2 through z>7) actively hurt by
      // removing genuinely-correct candidates over one noisy residual, but
      // a permissive z>10 through z>20 band gave a real, reproducible
      // median test R² improvement (0.8114 -> 0.8225 on this file's corpus,
      // pooled across 10 seeds). z>10 was chosen as the least aggressive
      // setting inside that winning band. Confirmed this only helps when
      // paired with the toleranceConstant tie-break above, not as a
      // replacement for it — removing that tie-break (toleranceConstant=0)
      // drops performance regardless of whether this gate is applied.
      // Floored at 5% of the FIT window's own Y-range: with very few FIT
      // points, the raw residual std is itself a noisy estimate and can come
      // out deceptively tiny just by chance (confirmed: fitting y=log(x)
      // over x=1..10000, only 5 FIT points, gave a raw std of 0.18 purely
      // from a lucky near-perfect fit) — which then makes z_check flag any
      // ordinary, non-catastrophic CHECK deviation as a huge violation
      // (15+ standard deviations) and reject a genuinely excellent model
      // (power-with-offset, R²=0.997 on the full curve) in favor of a much
      // worse one. The floor stops an unusually-tight small-sample fit from
      // creating an artificially hair-trigger denominator.
      const fitResidualStd = Math.max(stddev(fitXs.map((x, i) => fitYs[i] - fit.predict(x))), yRange * 0.05) || 1e-9;
      const zCheck = Math.max(...checkXs.map((x, i) => Math.abs(checkYs[i] - fit.predict(x)) / fitResidualStd));

      // Derivative gate: z_check can only see what happens inside the
      // narrow CHECK window — it has no way to catch a candidate that looks
      // perfectly reasonable there but accelerates away from reality once
      // TEST begins (classic polynomial extrapolation runaway, "Runge's
      // phenomenon" — confirmed directly on "effective tax rate in us by
      // year", where a quintic tracked the data closely right at the CHECK
      // boundary, passed z_check, then predicted a NEGATIVE tax rate a
      // decade later). The first/second/third numerical derivatives of
      // predict() right at that boundary, normalized by the train pool's
      // own X/Y range, measure exactly that risk directly — how fast is
      // this candidate moving/curving/accelerating at the exact point
      // extrapolation begins. Using the max of all three (not just one)
      // catches more real cases than any single order alone.
      const f = fit.predict;
      const d1 = (f(boundaryX + derivStep) - f(boundaryX - derivStep)) / (2 * derivStep);
      const d2 = (f(boundaryX + derivStep) - 2 * f(boundaryX) + f(boundaryX - derivStep)) / (derivStep * derivStep);
      const d3 = (f(boundaryX + 2 * derivStep) - 2 * f(boundaryX + derivStep) + 2 * f(boundaryX - derivStep) - f(boundaryX - 2 * derivStep)) / (2 * derivStep * derivStep * derivStep);
      const d1n = Number.isFinite(d1) ? Math.abs(d1) * xRange / yRange : Infinity;
      const d2n = Number.isFinite(d2) ? Math.abs(d2) * xRange * xRange / yRange : Infinity;
      const d3n = Number.isFinite(d3) ? Math.abs(d3) * xRange * xRange * xRange / yRange : Infinity;
      const derivative = Math.max(d1n, d2n, d3n);

      allResults.push({ candidate, fitR2: fit.r2, checkR2, zCheck, derivative });
    } catch (e) { /* this model type doesn't apply to this data — skip it */ }
  }

  if (!allResults.length) throw new Error('No candidate model could be fit to this data.');

  // Apply the z_check gate — but fall back to the unfiltered list if every
  // candidate would be excluded, rather than failing the dataset outright.
  const zCheckPassed = allResults.filter(r => r.zCheck <= zCheckThreshold);
  const zCheckFallbackTriggered = zCheckPassed.length === 0;
  let results = zCheckFallbackTriggered ? allResults : zCheckPassed;

  // Apply the derivative gate on top — same fallback philosophy. Threshold
  // of 5 was swept empirically: tied the no-gate median exactly (0.8827)
  // while lifting the aggregate average from 0.0501 to 0.5704 — a strictly
  // better tradeoff than z_extrap (a similar gate using the known test-X
  // points directly), which only reached ~0.62 average by giving up real
  // median performance (0.8498). Also confirmed this makes removing
  // cubic/quartic/quintic from CANDIDATES entirely unnecessary — with this
  // gate active, keeping vs. dropping them changes the average by <0.0001.
  let derivFiltered = results.filter(r => r.derivative <= effectiveDerivativeThreshold);
  if (derivFiltered.length) results = derivFiltered;

  // When z_check's OWN fallback just triggered (every candidate failed it),
  // it has zero real signal left to contribute, and the derivative gate
  // ends up as the sole decision-maker in a role it was never validated
  // for alone — confirmed on "delay vs value of reward": with only 4 FIT
  // points, z_check fails for literally everyone (z-scores of 30-98
  // against a threshold of 10), and the derivative gate alone then picked
  // `linear` (checkR2=-2.31) over the genuinely correct exponential decay
  // shape (checkR2=+0.10, boundary derivative 59 against a cap of 5)
  // purely for being flat — a hyperbolic discounting curve is SUPPOSED to
  // be steep right where the data ends.
  //
  // First attempt (skip the derivative gate entirely whenever z_check's
  // fallback triggers) was too blunt and got reverted: it fixed that case
  // but broke "Plies vs Perft Log scale", where the derivative gate is
  // doing real work even in a fallback dataset — it correctly excludes an
  // `exponential` fit that's a genuine Runge's-phenomenon blowup (boundary
  // derivative ~104 MILLION, not a borderline case), which then only lost
  // to `AR(1)` because the gate was there to remove it; without the gate,
  // the tolerance tie-break let it win anyway (candidate declaration
  // order breaks same-complexity ties, and `exponential` sits earlier in
  // CANDIDATES than `AR(1)`), taking that dataset's test R² from 0.50 to
  // 0.34.
  //
  // Narrower fix: only guarantee the single BEST raw-CHECK candidate
  // (across every fit, gates or not) survives into contention, rather than
  // reopening the gate for everyone. On "delay vs value of reward" that's
  // `exponential` (checkR2=0.10, best of anyone) — it gets added back even
  // though the derivative gate excluded it. On "Plies vs Perft" that's
  // already `AR(1)` (checkR2=0.61, best of anyone AND already gate-safe on
  // its own merits) — nothing changes, the dangerous `exponential` outlier
  // there is never the single best scorer, so it's never force-added.
  // Validated against the full corpus: fixes "delay vs value of reward"
  // (test R², full-variance: -0.18 -> -0.07 — still a small held-out
  // window, only 3 TEST points, but now the right shape instead of the
  // wrong one) while leaving every other dataset's pick untouched — full
  // corpus median stays at 0.9259, average moves 0.4566 -> 0.4586.
  if (zCheckFallbackTriggered) {
    const bestOverall = allResults.reduce((best, r) => (r.checkR2 > best.checkR2 ? r : best));
    if (!results.includes(bestOverall)) results = results.concat([bestOverall]);
  }

  // Pick the model with the best extrapolation score — but if a more
  // complex model only edges out a simpler one by less than `tolerance`,
  // prefer the simpler one instead. A win that small is more likely noise
  // than a real feature the extra parameters are capturing (this is what
  // stopped a quartic curve — 0.9997 R² vs quadratic's 0.9995 on one
  // dataset — from beating out the obviously-correct parabola shape).
  // `complexityOf` decides what counts as "simpler" for the tie-break —
  // defaults to each candidate's fixed coefficient count, but can be swapped
  // for a data-driven measure (e.g. how much a model's fit R² typically
  // outruns its check R² across a whole corpus of datasets).
  //
  // The tolerance itself scales with how few points are IN the check
  // window (unless the caller passes an explicit one) — an R² measured on
  // just 3 points is far noisier than one measured on 30, so a "win" needs
  // to be much bigger to actually mean something on a tiny check window.
  // Without this, a genuinely tied model (e.g. piecewise vs. plain linear,
  // both ~0.52 R² on the full data) could look like a clear winner purely
  // from a lucky/unlucky 3-point sample, and a needlessly complex model
  // would win on noise alone.
  //
  // Plain C/checkLen, not C/sqrt(checkLen) and not a constant + C/checkLen
  // — all three were tried against this file's real datasets (median
  // held-out test R² across everything, plus whether ONI half rodriguez
  // specifically stopped picking piecewise over an equally-good linear
  // fit). Pure C/checkLen with C=4 won outright against those two: -0.27
  // median vs -0.40 for the "+base" version and inconsistent results for
  // 1/sqrt(checkLen).
  //
  // C was later swept more finely (0 through 24) and split by whether the
  // dataset is a genuine time series or not, since RANSAC/Theil-Sen's
  // unseeded randomness made earlier single-run comparisons unreliable —
  // averaging 10 runs per dataset that picks a random-based model, then
  // repeating that across several independent process runs, showed: time
  // series datasets are flat across C=0-6 (median identical to 4 decimal
  // places, C=4 included), so they don't prefer any particular value in
  // that range; non-time-series datasets have a real, reproducible best at
  // C=3 (median test R² -0.2078, vs -0.2701 at C=4), stable across 7
  // independent runs, while C=2 looked good only as an artifact of
  // comparing rounds that shared one unseeded RNG stream within a single
  // process rather than truly independent runs. Net effect: C=3 is at
  // least as good as C=4 for time series and clearly better for
  // non-time-series, so it's the new default.
  // Capped at maxTolerance: C/checkLen alone was only ever tuned against the
  // default fitFrac=0.5/checkFrac=0.2 split. The /regression page's "use
  // full dataset" option calls this with checkFrac=0.3 instead, and on a
  // small dataset (e.g. n=10 -> checkLen=3) that makes C/checkLen = 1.0 — so
  // generous that a candidate scoring checkR2=0.437 ("living wage over
  // time"'s RANSAC pick) and one scoring -0.007 (constant) count as "tied,"
  // handing the win to constant on simplicity despite a real, large gap.
  // Capping the tolerance keeps the same "small checkLen needs a bigger
  // margin to trust" idea from blowing up into "almost anything counts as
  // tied" once checkLen gets small enough. Swept: 0.4 fixed the bug but
  // measurably hurt the default split's median (0.8827 -> 0.8631, some
  // default-split datasets also have small enough checkLen to hit the cap);
  // 0.48 fixes the same bug and leaves the default split fully intact
  // (median 0.8827, average even ticked up slightly to 0.4135).
  const effectiveTolerance = tolerance !== undefined ? tolerance : Math.min(toleranceConstant / checkLen, maxTolerance);

  const bestCheckR2 = Math.max(...results.map(r => r.checkR2));
  const contenders = results.filter(r => r.checkR2 >= bestCheckR2 - effectiveTolerance);
  const best = contenders.reduce((simplest, r) => (complexityOf(r.candidate) < complexityOf(simplest.candidate) ? r : simplest));

  return { name: best.candidate.name, candidate: best.candidate, fitR2: best.fitR2, checkR2: best.checkR2, fitLen, checkLen };
}

// Walk-forward (rolling-origin) validation — the alternative to the single
// fit/check split above. Instead of judging extrapolation from one cutoff
// point, it builds several: train on the first 40%, test on the next 10%;
// train on the first 50%, test on the next 10%; and so on, each cutoff
// moving forward by `stepFrac`. A candidate's score is its AVERAGE test R²
// across every cutoff, not just one — so a candidate that only looks good
// because of where one particular split happened to land can't win here.
// The true final segment (the last `finalHoldoutFrac` of the data) is never
// touched by any cutoff — it's reserved for the caller to do a genuine,
// single, untouched validation check after a model type has been chosen.
function selectByWalkForward(xs, ys, {
  initialTrainFrac = 0.4, stepFrac = 0.1, testFrac = 0.1, finalHoldoutFrac = 0.3,
  tolerance, toleranceConstant = 3, gateMultiplier = 1,
  complexityOf = (candidate) => candidate.params,
} = {}) {
  const n = xs.length;
  const reservedStart = Math.floor(n * (1 - finalHoldoutFrac));

  // A test window with zero variance in y (either a single point, or several
  // points that all happen to be equal — e.g. two attempts that both scored
  // 100%) makes R² meaningless by construction: computeR2's ssTot===0
  // branch returns a trivial 1 no matter how wrong the prediction is. A
  // cutoff like that is rejected rather than letting every candidate "pass"
  // it for free.
  const isUsableWindow = (trainEnd, testEnd) => {
    if (testEnd - trainEnd < 2) return false;
    const testYs = ys.slice(trainEnd, testEnd);
    return !testYs.every(y => y === testYs[0]);
  };

  const cutoffs = [];
  for (let trainFrac = initialTrainFrac; trainFrac + testFrac <= 1 - finalHoldoutFrac + 1e-9; trainFrac += stepFrac) {
    const trainEnd = Math.floor(n * trainFrac);
    const testEnd = Math.floor(n * (trainFrac + testFrac));
    if (testEnd > reservedStart) break;
    if (!isUsableWindow(trainEnd, testEnd)) continue;
    cutoffs.push({ trainEnd, testStart: trainEnd, testEnd });
  }

  // Too little data for multiple walk-forward cutoffs — fall back to a
  // single train/test pair instead of failing outright. Uses the whole
  // remaining pre-holdout stretch as the one test window (not just
  // `testFrac`) so that single pair has the best chance of being large
  // enough to mean anything, while the last `finalHoldoutFrac` still stays
  // untouched exactly as it would with multiple cutoffs.
  if (!cutoffs.length) {
    const trainEnd = Math.floor(n * initialTrainFrac);
    if (isUsableWindow(trainEnd, reservedStart)) {
      cutoffs.push({ trainEnd, testStart: trainEnd, testEnd: reservedStart });
    }
  }

  if (!cutoffs.length) {
    throw new Error(`Need more data for walk-forward validation — got ${n} rows.`);
  }

  const totalTestPoints = cutoffs.reduce((s, c) => s + (c.testEnd - c.testStart), 0);
  const avgTestLen = totalTestPoints / cutoffs.length;

  const results = [];
  for (const candidate of CANDIDATES) {
    if (cutoffs[0].trainEnd < candidate.minFit) continue;
    // Same reasoning as the single-split gate: a test window smaller than
    // the candidate's own parameter count can't meaningfully validate it.
    if (avgTestLen < candidate.params * gateMultiplier) continue;

    const foldScores = [];
    let failed = false;
    for (const cutoff of cutoffs) {
      const trainXs = xs.slice(0, cutoff.trainEnd), trainYs = ys.slice(0, cutoff.trainEnd);
      const testXs = xs.slice(cutoff.testStart, cutoff.testEnd), testYs = ys.slice(cutoff.testStart, cutoff.testEnd);
      try {
        const fit = candidate.fit(trainXs, trainYs);
        if (!Number.isFinite(fit.r2)) { failed = true; break; }
        const foldR2 = RegressionLib.computeR2(testXs, testYs, fit.predict);
        if (!Number.isFinite(foldR2)) { failed = true; break; }
        foldScores.push(foldR2);
      } catch (e) { failed = true; break; }
    }
    // Require every cutoff to succeed, same strictness as the single-split
    // path — no partial credit for a candidate that only works on some
    // cutoffs, since that's its own form of picking a lucky window.
    if (failed) continue;

    const avgScore = foldScores.reduce((s, v) => s + v, 0) / foldScores.length;
    results.push({ candidate, avgScore, foldScores });
  }

  if (!results.length) throw new Error('No candidate model could be fit across the walk-forward cutoffs.');

  // Tolerance uses the TOTAL points seen across every cutoff's test window,
  // not just one — more cutoffs (or bigger ones) means more evidence, so a
  // "win" needs to be smaller to mean something, same logic as the single-
  // split path but with a much larger effective sample most of the time.
  const effectiveTolerance = tolerance !== undefined ? tolerance : toleranceConstant / totalTestPoints;

  const bestScore = Math.max(...results.map(r => r.avgScore));
  const contenders = results.filter(r => r.avgScore >= bestScore - effectiveTolerance);
  const best = contenders.reduce((simplest, r) => (complexityOf(r.candidate) < complexityOf(simplest.candidate) ? r : simplest));

  return {
    name: best.candidate.name, candidate: best.candidate,
    avgScore: best.avgScore, foldScores: best.foldScores,
    cutoffCount: cutoffs.length, totalTestPoints, reservedStart,
  };
}

const ModelCandidates = { CANDIDATES, selectByExtrapolation, selectByWalkForward };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ModelCandidates;
} else {
  window.ModelCandidates = ModelCandidates;
}
