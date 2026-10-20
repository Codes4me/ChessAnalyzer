// The 10 RNG seeds used across the fit:check ratio sweeps (see
// seed_ratio_sweep.js-style experiments) — saved here so different sweep
// runs stay comparable instead of each picking its own arbitrary seeds.
// Passed to RegressionLib.fitRANSAC/fitTheilSen's `seed` option.
const SWEEP_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SWEEP_SEEDS;
} else {
  window.SweepSeeds = SWEEP_SEEDS;
}
