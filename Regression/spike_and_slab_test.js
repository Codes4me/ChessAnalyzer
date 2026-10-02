// Spike-and-slab term selection over a small polynomial basis {x, x^2, x^3,
// x^4} — a standalone analysis script (not wired into regression.js/html
// yet; see the discussion this came out of). Unlike the project's existing
// "try degree 1..4, keep whichever wins by extrapolation R²" approach
// (discrete model selection — one winner, no sense of how CONFIDENT that
// pick is), this enumerates all 2^4=16 possible subsets of the basis
// (always keeping the intercept) and computes each one's exact marginal
// likelihood under a standard Bayesian linear-model prior, then reports a
// posterior INCLUSION PROBABILITY per term: "given everything, how much of
// the posterior mass wants x^3 in the model at all" — a continuous number,
// not a discrete yes/no, and the thing needed to actually settle the
// cubic/quartic/quintic question in this project's commit history (06c0780:
// "Remove cubic/quartic/quintic/sinusoidal: bootstrap-confirmed dead
// weight") with a real posterior number instead of an empirical approximate
// finding.
//
// Marginal likelihood: Fernandez/Ley/Steel (2001) closed-form Bayes factor
// for a Gaussian linear model under Zellner's g-prior, g=n (the "unit
// information prior" — a common, mild default with no free knob to tune):
//
//   BF(model vs null) = (1+g)^((n-1-p)/2) / (1 + g*(1-R^2))^((n-1)/2)
//
// where p = number of non-intercept regressors in `model`, R^2 is that
// model's ordinary R^2 against the (intercept-included) null model, and
// `null` is the intercept-only model (BF=1 by definition). This has a
// genuine closed form — no MCMC, no approximation — which is exactly what
// "exact 2^k marginal-likelihood comparison" means for a basis this small.
// Posterior model probability (uniform prior over all 16 models) is then
// just each model's BF normalized to sum to 1 across all candidates.
const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression');

const BASIS = ['x', 'x^2', 'x^3', 'x^4'];

function allSubsets(items) {
  const subsets = [[]];
  for (const item of items) {
    const n = subsets.length;
    for (let i = 0; i < n; i++) subsets.push([...subsets[i], item]);
  }
  return subsets; // 2^items.length subsets, [0] is the empty set
}

// R^2 of a given subset of basis terms against y, via the project's own
// fitMultiLinear (always includes an intercept). Returns the fit itself too
// (not just r2) so the robust pass below can get residuals out of it
// without refitting.
function subsetFit(xs, ys, subset) {
  if (subset.length === 0) {
    const mean = ys.reduce((s, y) => s + y, 0) / ys.length;
    return { r2: 0, predict: () => mean };
  }
  const X = xs.map(x => subset.map(term => {
    if (term === 'x') return x;
    if (term === 'x^2') return x * x;
    if (term === 'x^3') return x * x * x;
    if (term === 'x^4') return x * x * x * x;
  }));
  const fit = Regression.fitMultiLinear(X, ys);
  return { ...fit, r2: Math.max(0, fit.r2) };
}

function evaluateDataset(xs, ys) {
  const n = xs.length;
  const g = n; // unit-information prior
  const subsets = allSubsets(BASIS);
  const fits = subsets.map(subset => ({ subset, fit: subsetFit(xs, ys, subset) }));

  // --- Original pass: Fernandez-Ley-Steel g-prior Bayes factor. Exact
  // under a Gaussian-error assumption — this is the "every BIC/marginal-
  // likelihood comparison implicitly assumes Gaussian errors" gap.
  const models = fits.map(({ subset, fit }) => {
    const p = subset.length;
    const logBF = p === 0 ? 0
      : ((n - 1 - p) / 2) * Math.log(1 + g) - ((n - 1) / 2) * Math.log(1 + g * (1 - fit.r2));
    return { subset, r2: fit.r2, logBF };
  });

  // --- Robust pass: same 16 models, but scored by robustModelBIC, which
  // picks whichever of Gaussian/Laplace/Student-t actually fits THIS
  // subset's own residuals best (regression.js's chooseErrorDistribution),
  // instead of assuming Gaussian for all of them. Posterior model
  // probability from a BIC score is the standard large-n approximation
  // P(model) ~ exp(-BIC/2), the same approximation the g-prior formula
  // itself asymptotes to — so this is a like-for-like alternative, not a
  // different kind of number.
  const robustModels = fits.map(({ subset, fit }) => {
    const residuals = xs.map((x, i) => ys[i] - fit.predict(subset.length ? subset.map(term => {
      if (term === 'x') return x;
      if (term === 'x^2') return x * x;
      if (term === 'x^3') return x * x * x;
      return x * x * x * x;
    }) : x));
    const robust = Regression.robustModelBIC(residuals, subset.length + 1); // +1 for intercept
    return { subset, robust, residuals };
  });

  // --- Control: a plain Gaussian BIC using the SAME generic formula as the
  // robust pass (NOT the FLS exact Bayes factor), just without ever
  // swapping away from Gaussian. This isolates whether a Gaussian-vs-robust
  // difference is really about non-Gaussian errors, or just about which
  // formula (exact FLS Bayes factor vs. generic BIC approximation) is used
  // — those are two independent axes, and conflating them would have been
  // a real mistake here (confirmed: on this corpus, almost the entire gap
  // between the ORIGINAL (FLS) and ROBUST numbers turned out to be the
  // formula switch, not the error-distribution choice — see this script's
  // printed GaussianBIC-control row).
  const gaussianBICModels = robustModels.map(({ subset, residuals }) => {
    const n2 = residuals.length;
    const sigma2 = residuals.reduce((s, r) => s + r * r, 0) / n2;
    if (sigma2 <= 0) return { subset, bic: null };
    const logLik = residuals.reduce((s, r) => s - 0.5 * Math.log(2 * Math.PI * sigma2) - (r * r) / (2 * sigma2), 0);
    const bic = (subset.length + 1 + 1) * Math.log(n2) - 2 * logLik; // +1 intercept, +1 sigma
    return { subset, bic };
  });

  function posteriorsFromLog(logScores) {
    const maxLog = Math.max(...logScores);
    const weights = logScores.map(l => Math.exp(l - maxLog));
    const Z = weights.reduce((s, w) => s + w, 0);
    return weights.map(w => w / Z);
  }

  const posteriors = posteriorsFromLog(models.map(m => m.logBF));
  const robustPosteriors = robustModels.every(m => m.robust)
    ? posteriorsFromLog(robustModels.map(m => -0.5 * m.robust.bic))
    : null;
  const gaussianBICPosteriors = gaussianBICModels.every(m => m.bic !== null)
    ? posteriorsFromLog(gaussianBICModels.map(m => -0.5 * m.bic))
    : null;

  const inclusion = {};
  const robustInclusion = {};
  const gaussianBICInclusion = {};
  for (const term of BASIS) {
    inclusion[term] = models.reduce((s, m, i) => s + (m.subset.includes(term) ? posteriors[i] : 0), 0);
    robustInclusion[term] = robustPosteriors
      ? models.reduce((s, m, i) => s + (m.subset.includes(term) ? robustPosteriors[i] : 0), 0)
      : null;
    gaussianBICInclusion[term] = gaussianBICPosteriors
      ? models.reduce((s, m, i) => s + (m.subset.includes(term) ? gaussianBICPosteriors[i] : 0), 0)
      : null;
  }
  const best = models[posteriors.indexOf(Math.max(...posteriors))];
  const robustBest = robustPosteriors ? models[robustPosteriors.indexOf(Math.max(...robustPosteriors))] : null;
  const errDistUsed = robustModels.map(m => m.robust && m.robust.errorDistribution);
  return { n, inclusion, robustInclusion, gaussianBICInclusion, best, robustBest, models, posteriors, robustPosteriors, errDistUsed };
}

// Excluded: flagged in an earlier session as "structurally sorted by rank,
// not a real sequence" — Movie Gross's x axis is a rank/sort order, so its
// smooth, near-deterministic curvature makes every basis term look like it
// has near-1.0 posterior inclusion probability regardless of whether
// there's a real quartic relationship. Including it would make the
// aggregate numbers below mostly describe one artifact-laden dataset,
// especially at n=1334 (by far the largest in the corpus, so it would also
// dominate any n-weighted reading even if it weren't excluded outright).
const EXCLUDED_TITLES = ['Movie Gross'];

// Same seeded-PRNG convention as regression.js's createSeededRandom, kept
// separate here since this script doesn't load regression.js's internals.
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

// Dataset-level bootstrap: resamples the per-dataset results themselves
// (with replacement), not the raw data points — the question here is "is
// the AGGREGATE inclusion-probability finding being driven by a handful of
// datasets", the same kind of robustness check this project has run on
// every other aggregate finding (e.g. the autocorrelation/heteroscedasticity
// outlier-resistance audits). Each resample recomputes the mean inclusion
// probability per term; the spread across resamples is the uncertainty on
// that aggregate, not on any individual dataset's own fit.
function bootstrapAggregate(results, field, { iterations = 2000, seed = 0x5eed1234 } = {}) {
  const random = mulberry32(seed);
  const n = results.length;
  const samples = {};
  for (const term of BASIS) samples[term] = [];
  for (let i = 0; i < iterations; i++) {
    const sums = {}; for (const term of BASIS) sums[term] = 0;
    for (let j = 0; j < n; j++) {
      const r = results[Math.floor(random() * n)];
      for (const term of BASIS) sums[term] += r[field][term];
    }
    for (const term of BASIS) samples[term].push(sums[term] / n);
  }
  const out = {};
  for (const term of BASIS) {
    const sorted = [...samples[term]].sort((a, b) => a - b);
    const lo = sorted[Math.floor(0.025 * (sorted.length - 1))];
    const hi = sorted[Math.ceil(0.975 * (sorted.length - 1))];
    out[term] = { lower: lo, upper: hi };
  }
  return out;
}

function main() {
  const inputPath = path.resolve(process.argv[2] || 'data.txt');
  const rawText = fs.readFileSync(inputPath, 'utf8');
  const { groups } = DataParser.parseGroups(rawText);

  const results = [];
  for (const group of groups) {
    const title = group.title || '(untitled)';
    if (EXCLUDED_TITLES.some(ex => title.includes(ex))) continue;
    const xs = group.columns[0], ys = group.columns[1];
    const valid = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
    const vxs = xs.filter((_, i) => valid[i]), vys = ys.filter((_, i) => valid[i]);
    if (vxs.length < 10 || new Set(vxs).size < 6) continue; // need enough distinct x to support a quartic at all
    try {
      const r = evaluateDataset(vxs, vys);
      results.push({ title, ...r });
    } catch (e) { /* skip datasets fitMultiLinear can't handle (e.g. singular design) */ }
  }

  console.log(`Spike-and-slab posterior inclusion probabilities, ${results.length} datasets (excluded: ${EXCLUDED_TITLES.join(', ')}), basis = {${BASIS.join(', ')}}\n`);
  console.log('(Gaussian = original Fernandez-Ley-Steel g-prior; Robust = same 16 models re-scored by whichever error distribution — Gaussian/Laplace/Student-t — actually fits THAT subset\'s own residuals best)\n');
  results.forEach(r => {
    const incLine = BASIS.map(t => `${t}=${r.inclusion[t].toFixed(2)}`).join('  ');
    const robustIncLine = BASIS.map(t => `${t}=${r.robustInclusion[t]?.toFixed(2) ?? 'n/a'}`).join('  ');
    const bestLabel = r.best.subset.length ? r.best.subset.join('+') : '(intercept only)';
    const nonGaussianCount = r.errDistUsed.filter(d => d && d !== 'Gaussian').length;
    console.log(`  ${r.title.slice(0, 40).padEnd(40)} n=${String(r.n).padStart(4)}  best subset: ${bestLabel}`);
    console.log(`      Gaussian: ${incLine}`);
    console.log(`      Robust:   ${robustIncLine}  (${nonGaussianCount}/16 subsets preferred a non-Gaussian error distribution)`);
  });

  const bootCI = bootstrapAggregate(results, 'inclusion');
  const robustResults = results.filter(r => BASIS.every(t => r.robustInclusion[t] !== null));
  const robustBootCI = robustResults.length ? bootstrapAggregate(robustResults, 'robustInclusion') : null;
  const gbicResults = results.filter(r => BASIS.every(t => r.gaussianBICInclusion[t] !== null));
  const gbicBootCI = gbicResults.length ? bootstrapAggregate(gbicResults, 'gaussianBICInclusion') : null;

  console.log('\nAggregate (mean posterior inclusion probability across all datasets, with a dataset-level bootstrap 95% CI — 2000 resamples):');
  console.log('(Original = exact FLS Bayes factor, Gaussian-BIC-control = generic BIC formula forced to stay Gaussian, Robust = generic BIC formula allowed to pick Laplace/Student-t.');
  console.log(' Comparing Original vs. Gaussian-BIC-control isolates the FORMULA effect; comparing Gaussian-BIC-control vs. Robust isolates the actual non-Gaussian-likelihood effect.)');
  for (const term of BASIS) {
    const mean = results.reduce((s, r) => s + r.inclusion[term], 0) / results.length;
    const { lower, upper } = bootCI[term];
    console.log(`  Original (FLS)         ${term}: ${mean.toFixed(3)}  (95% CI ${lower.toFixed(3)} to ${upper.toFixed(3)})`);
  }
  if (gbicBootCI) {
    for (const term of BASIS) {
      const mean = gbicResults.reduce((s, r) => s + r.gaussianBICInclusion[term], 0) / gbicResults.length;
      const { lower, upper } = gbicBootCI[term];
      console.log(`  Gaussian-BIC-control   ${term}: ${mean.toFixed(3)}  (95% CI ${lower.toFixed(3)} to ${upper.toFixed(3)})`);
    }
  }
  if (robustBootCI) {
    for (const term of BASIS) {
      const mean = robustResults.reduce((s, r) => s + r.robustInclusion[term], 0) / robustResults.length;
      const { lower, upper } = robustBootCI[term];
      console.log(`  Robust (non-Gaussian)  ${term}: ${mean.toFixed(3)}  (95% CI ${lower.toFixed(3)} to ${upper.toFixed(3)})`);
    }
  }

  console.log('\nHow many datasets give each term >=0.5 posterior inclusion probability (i.e. "the data genuinely wants this term"):');
  for (const term of BASIS) {
    const count = results.filter(r => r.inclusion[term] >= 0.5).length;
    const gbicCount = gbicResults.filter(r => r.gaussianBICInclusion[term] >= 0.5).length;
    const robustCount = robustResults.filter(r => r.robustInclusion[term] >= 0.5).length;
    console.log(`  ${term}: Original ${count}/${results.length}   Gaussian-BIC-control ${gbicCount}/${gbicResults.length}   Robust ${robustCount}/${robustResults.length}`);
  }
}

main();
