// Outlier-resistance check for the orphaned diagnostic battery
// (moretests.js: Box-Pierce, Runs test, Glejser, Spearman-rank
// heteroscedasticity), same methodology already used in this project to
// compare White's/Breusch-Pagan/Goldfeld-Quandt (see the comment on
// whiteTest in regression.js: "47% of its flags evaporate after removing
// MAD-outliers, vs 71% for Breusch-Pagan and 88% for Goldfeld-Quandt").
//
// Outlier rule: a point whose residual is more than 1.5 * robustSigma away
// from the median residual, where robustSigma = 1.4826 * MAD — the exact
// "1.5 robust standard deviations" convention already used elsewhere in
// this codebase (regression.js's own robustSigma, backtest_models.js's
// trimming fence). For each dataset where a test is significant (p<0.05)
// on the full residuals, drop the outlier points and re-run the SAME test
// on what's left — if it stops being significant, that flag was resting on
// a handful of extreme points rather than a real, broad pattern.
const fs = require('fs');
const path = require('path');
const DataParser = require('./data_parser');
const Regression = require('./regression.js');
const { selectByExtrapolation } = require('./model_candidates');
const { boxPierce, runsTest, glejserTest, spearmanHetero } = require('./moretests.js');

function median(arr) { const s = [...arr].sort((a, b) => a - b); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function robustSigma(values) {
  const m = median(values);
  const mad = median(values.map(v => Math.abs(v - m)));
  return 1.4826 * mad;
}

function dropOutliers(xs, residuals) {
  const m = median(residuals);
  const sigma = robustSigma(residuals);
  if (sigma === 0) return { xs, residuals, dropped: 0 };
  const fence = 1.5 * sigma;
  const keepXs = [], keepRes = [];
  let dropped = 0;
  for (let i = 0; i < residuals.length; i++) {
    if (Math.abs(residuals[i] - m) > fence) { dropped++; continue; }
    keepXs.push(xs[i]); keepRes.push(residuals[i]);
  }
  return { xs: keepXs, residuals: keepRes, dropped };
}

function evaluateDataset(group) {
  const xs = group.columns[0], ys = group.columns[1];
  const valid = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
  const vxs = xs.filter((_, i) => valid[i]), vys = ys.filter((_, i) => valid[i]);
  if (vxs.length < 15) return null;
  let selection, fit;
  try { selection = selectByExtrapolation(vxs, vys, { fitFrac: 0.5, checkFrac: 0.2 }); fit = selection.candidate.fit(vxs, vys); } catch (e) { return null; }
  const residuals = vxs.map((x, i) => vys[i] - fit.predict(x));
  const h = Math.max(1, Math.min(10, Math.floor(vxs.length / 5)));
  return { title: group.title, n: vxs.length, xs: vxs, residuals, h };
}

function main() {
  const inputPath = path.resolve(process.argv[2] || 'data.txt');
  const { groups } = DataParser.parseGroups(fs.readFileSync(inputPath, 'utf8'));
  const datasets = groups.map(evaluateDataset).filter(Boolean);
  console.log(`Outlier-resistance check, ${datasets.length} datasets (1.5*robustSigma fence)\n`);

  const tests = {
    'Box-Pierce': d => boxPierce(d.residuals, d.h).pValue,
    'Runs test': d => runsTest(d.residuals).pValue,
    'Glejser': d => glejserTest(d.xs, d.residuals).pValue,
    'Spearman-rank hetero': d => spearmanHetero(d.xs, d.residuals).pValue,
    // The two SHIPPED tests, run through this exact same script/corpus/
    // fence for an apples-to-apples comparison — regression.js's own
    // comments cite "47%" for White's test from a different session's
    // script; this re-derives it under identical conditions to the four
    // above instead of comparing numbers from two different methodologies.
    'Durbin-Watson (shipped)': d => Regression.durbinWatsonTest(d.residuals).pValue,
    "White's test (shipped)": d => Regression.whiteTest(d.xs, d.residuals).pValue,
  };

  for (const [name, pValueFn] of Object.entries(tests)) {
    let flagged = 0, evaporated = 0;
    const detail = [];
    for (const d of datasets) {
      let pBefore;
      try { pBefore = pValueFn(d); } catch (e) { continue; }
      if (!(pBefore < 0.05)) continue;
      flagged++;
      const { xs: txs, residuals: tres, dropped } = dropOutliers(d.xs, d.residuals);
      if (tres.length < 8) { detail.push({ title: d.title, pBefore, pAfter: null, dropped, note: 'too few left' }); continue; }
      const trimmedD = { xs: txs, residuals: tres, h: Math.max(1, Math.min(10, Math.floor(tres.length / 5))) };
      let pAfter;
      try { pAfter = pValueFn(trimmedD); } catch (e) { continue; }
      const evap = !(pAfter < 0.05);
      if (evap) evaporated++;
      detail.push({ title: d.title, pBefore, pAfter, dropped, evap });
    }
    const pct = flagged ? (100 * evaporated / flagged).toFixed(0) : 'n/a';
    console.log(`${name}: ${evaporated}/${flagged} flags evaporate after dropping outliers (${pct}%)`);
    detail.forEach(d => {
      const line = d.pAfter === null
        ? `    ${d.title.slice(0, 40).padEnd(40)} p=${d.pBefore.toFixed(4)} -> (${d.note}, dropped ${d.dropped})`
        : `    ${d.title.slice(0, 40).padEnd(40)} p=${d.pBefore.toFixed(4)} -> ${d.pAfter.toFixed(4)}  dropped=${d.dropped}  ${d.evap ? 'EVAPORATED' : 'still significant'}`;
      console.log(line);
    });
    console.log('');
  }
}

main();
