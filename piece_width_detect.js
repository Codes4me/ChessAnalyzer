// Second, alternative piece-occupancy detector: instead of measuring
// interior gradient VARIANCE (the existing detectPieceCells heuristic),
// look for a pair of edges separated by roughly a piece's expected width.
// A piece photographed at an angle stretches long in one direction, but
// its SHORTEST visible cross-section stays close to its true width
// regardless of tilt -- so we scan several angles through each cell and
// take the smallest valid edge-pair separation found, then check whether
// that width falls in the range a real piece would occupy (relative to
// the cell's own size, which already reflects local perspective scale
// since it comes from the same box/pitch used to build the cell).
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const { buildGrad } = require('./quad_segment_score.js');
const { correctRotateDeg } = require('./orientation_fix.js');
const { sliceImageToSquares, STARTING_FEN_ROWS } = require('./test_match.js');

const CONTRAST_THRESHOLD = 20;
const WIDTH_FRAC_MIN = 0.3, WIDTH_FRAC_MAX = 0.9; // fraction of cell pitch
const ANGLES_DEG = [0, 30, 60, 90, 120, 150]; // sampled scan directions through each cell

function buildGxGy(gray, W, H) {
  const gx = new Float64Array(W*H), gy = new Float64Array(W*H);
  for (let y=1;y<H-1;y++) for (let x=1;x<W-1;x++) {
    const i=y*W+x;
    gx[i]=gray[i-W+1]+2*gray[i+1]+gray[i+W+1]-gray[i-W-1]-2*gray[i-1]-gray[i+W-1];
    gy[i]=gray[i-W-1]+2*gray[i-W]+gray[i-W+1]-gray[i+W-1]-2*gray[i+W]-gray[i+W+1];
  }
  return { gx, gy };
}

// Scan a line through (cx,cy) at the given angle, spanning +-halfLen,
// sample gradient magnitude, find local-maxima "edge points", and return
// the smallest separation between any two edge points (or null if <2 found).
function scanMinEdgeSeparation(gx, gy, W, H, cx, cy, angleRad, halfLen, step) {
  const dirx = Math.cos(angleRad), diry = Math.sin(angleRad);
  const samples = [];
  for (let t=-halfLen; t<=halfLen; t+=step) {
    const x = cx+dirx*t, y = cy+diry*t;
    const xi = Math.round(x), yi = Math.round(y);
    if (xi<1||xi>=W-1||yi<1||yi>=H-1) { samples.push(0); continue; }
    const i = yi*W+xi;
    samples.push(Math.hypot(gx[i], gy[i]));
  }
  const edgeTs = [];
  for (let i=1;i<samples.length-1;i++) {
    if (samples[i] > CONTRAST_THRESHOLD && samples[i] >= samples[i-1] && samples[i] >= samples[i+1]) {
      edgeTs.push(-halfLen + i*step);
    }
  }
  if (edgeTs.length < 2) return null;
  // Full span (outermost edge to outermost edge) along this scan direction
  // -- NOT the smallest gap between any two points, which just picks up
  // internal piece-carving texture noise instead of the true silhouette
  // width. The "shortest direction" is the smallest such SPAN across angles.
  return Math.max(...edgeTs) - Math.min(...edgeTs);
}

// Returns an 8x8 boolean grid.
function detectPieceCellsByWidth(gray, W, H, box) {
  const { gx, gy } = buildGxGy(gray, W, H);
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const pitch = (pitchX+pitchY)/2;
  const result = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const cx = box.x0 + (c+0.5)*pitchX, cy = box.y0 + (r+0.5)*pitchY;
      let smallestWidth = Infinity;
      for (const deg of ANGLES_DEG) {
        const w = scanMinEdgeSeparation(gx, gy, W, H, cx, cy, deg*Math.PI/180, pitch*0.5, 1);
        if (w != null && w < smallestWidth) smallestWidth = w;
      }
      const frac = smallestWidth / pitch;
      row.push(Number.isFinite(smallestWidth) && frac >= WIDTH_FRAC_MIN && frac <= WIDTH_FRAC_MAX);
    }
    result.push(row);
  }
  return result;
}

// --- Reuse the existing gradient-variance detector for comparison ---
// Returns the raw 8x8 average-gradient grid (continuous), plus a helper
// to threshold it into booleans the way the original heuristic did.
function computeCellGrad(grad, W, H, box) {
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const cellGrad = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (let c = 0; c < 8; c++) {
      const cx0 = box.x0+c*pitchX, cx1 = box.x0+(c+1)*pitchX, cy0 = box.y0+r*pitchY, cy1 = box.y0+(r+1)*pitchY;
      const innerX0 = cx0+pitchX*0.2, innerX1 = cx1-pitchX*0.2, innerY0 = cy0+pitchY*0.2, innerY1 = cy1-pitchY*0.2;
      let s=0,n=0;
      for (let y=Math.round(innerY0); y<Math.round(innerY1); y++) {
        if (y<0||y>=H) continue;
        for (let x=Math.round(innerX0); x<Math.round(innerX1); x++) {
          if (x<0||x>=W) continue;
          s += grad[y*W+x]; n++;
        }
      }
      row.push(n?s/n:0);
    }
    cellGrad.push(row);
  }
  return cellGrad;
}
function detectPieceCellsByVariance(grad, W, H, box) {
  const cellGrad = computeCellGrad(grad, W, H, box);
  const flat = cellGrad.flat();
  const mean = flat.reduce((a,b)=>a+b,0)/flat.length;
  return cellGrad.map(row => row.map(v => v > mean*1.3));
}
// Continuous "how piece-like" score for the variance method: the cell's
// gradient relative to the photo's own mean (same normalization the
// boolean version thresholds at 1.3), so it is comparable across photos.
function varianceScoreGrid(grad, W, H, box) {
  const cellGrad = computeCellGrad(grad, W, H, box);
  const flat = cellGrad.flat();
  const mean = flat.reduce((a,b)=>a+b,0)/flat.length || 1;
  return cellGrad.map(row => row.map(v => v/mean));
}
// Continuous "how piece-like" score for the width method: peaks at the
// center of the valid width-fraction range and falls off toward its edges
// (and is 0 when no width could be measured at all), instead of a hard
// in-range/out-of-range boolean.
function widthScoreGrid(gray, W, H, box) {
  const { gx, gy } = buildGxGy(gray, W, H);
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const pitch = (pitchX+pitchY)/2;
  const center = (WIDTH_FRAC_MIN+WIDTH_FRAC_MAX)/2, spread = (WIDTH_FRAC_MAX-WIDTH_FRAC_MIN)/2;
  const result = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const cx = box.x0 + (c+0.5)*pitchX, cy = box.y0 + (r+0.5)*pitchY;
      let smallestWidth = Infinity;
      for (const deg of ANGLES_DEG) {
        const w = scanMinEdgeSeparation(gx, gy, W, H, cx, cy, deg*Math.PI/180, pitch*0.5, 1);
        if (w != null && w < smallestWidth) smallestWidth = w;
      }
      if (!Number.isFinite(smallestWidth)) { row.push(0); continue; }
      const frac = smallestWidth / pitch;
      const score = Math.max(0, 1 - Math.abs(frac-center)/spread);
      row.push(score);
    }
    result.push(row);
  }
  return result;
}

// --- Ground truth (reused from train_piece_classifier.js's known mapping) ---
function expandFenRow(row) { let out=''; for (const ch of row) out += /[1-8]/.test(ch)?'-'.repeat(parseInt(ch,10)):ch; return out; }
function fenToRows(fen) { return fen.split(' ')[0].split('/').map(expandFenRow); }
const STARTING_ROWS = fenToRows('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
const QUEEN_SORTIE_ROWS = fenToRows('rnbqkbnr/pppppppp/8/8/2Q2q2/8/PPPPPPPP/RNBQKBNR');
const ELITEST_ROWS = fenToRows('8/8/8/8/3K4/8/3kr3/8');
const PHOTOS = {
  'start1.jpeg':       { rows: STARTING_ROWS, orientation: 'rot90cw' },
  'elistart.jpeg':      { rows: STARTING_ROWS, orientation: 'rowflip' },
  'image0.jpeg':        { rows: STARTING_ROWS, orientation: 'default' },
  'image1.jpeg':        { rows: STARTING_ROWS, orientation: 'default' },
  'image3.jpeg':        { rows: STARTING_ROWS, orientation: 'default' },
  'image4.jpeg':        { rows: STARTING_ROWS, orientation: 'rowflip' },
  'image5.jpeg':        { rows: STARTING_ROWS, orientation: 'rot90cw' },
  'image6.jpeg':        { rows: STARTING_ROWS, orientation: 'default' },
  'image7.jpeg':        { rows: STARTING_ROWS, orientation: 'default' },
  'bright_light.jpeg':  { rows: QUEEN_SORTIE_ROWS, orientation: 'default' },
  'low_light.jpeg':     { rows: QUEEN_SORTIE_ROWS, orientation: 'default' },
  'medium_light.jpeg':  { rows: QUEEN_SORTIE_ROWS, orientation: 'default' },
  'oblique_no.jpeg':    { rows: QUEEN_SORTIE_ROWS, orientation: 'default' },
  'oblique_yes.jpeg':   { rows: QUEEN_SORTIE_ROWS, orientation: 'default' },
  'elitest.jpg':        { rows: ELITEST_ROWS, orientation: 'default' },
};
function labelAt(rows, orientation, row, col) {
  if (orientation === 'default') return rows[row][col];
  if (orientation === 'rowflip') return rows[7-row][col];
  if (orientation === 'rot90cw') return rows[7-col][row];
}

// Same holdout set used for the piece-classifier NN work: one of each kind
// (custom-FEN photo, rowflip photo, rot90cw photo) -- kept OUT of tuning.
const HOLDOUT = ['oblique_yes.jpeg', 'image4.jpeg', 'start1.jpeg'];

async function main() {
  const gt = JSON.parse(fs.readFileSync('ground_truth.json', 'utf8'));
  let total=0;
  const samples = []; // { varRaw, widthScore, truePiece, isHoldout } for the weighted-combo sweep

  console.log('Holdout photos (tuning never sees these):', HOLDOUT.join(', '), '\n');

  for (const [name, info] of Object.entries(PHOTOS)) {
    const entry = gt[name];
    if (!entry) continue;
    const imgPath = path.join(__dirname, 'Data', name);
    const rotateDeg = await correctRotateDeg(imgPath);
    const { grad, W, H } = await buildGrad(imgPath, rotateDeg);
    const { data: gray } = await sharp(imgPath).rotate(rotateDeg).resize(W, H, { fit: 'fill' }).grayscale().raw().toBuffer({ resolveWithObject: true });

    const meta = (await sharp(imgPath).rotate(rotateDeg).toBuffer({ resolveWithObject: true })).info;
    const sx = W/meta.width, sy = H/meta.height;
    const pts = entry.corners.map(([x,y]) => [x*sx, y*sy]);
    const x0 = Math.min(...pts.map(p=>p[0])), x1 = Math.max(...pts.map(p=>p[0]));
    const y0 = Math.min(...pts.map(p=>p[1])), y1 = Math.max(...pts.map(p=>p[1]));
    const box = { x0, y0, x1, y1 };

    const varGrid = detectPieceCellsByVariance(grad, W, H, box);
    const widthGrid = detectPieceCellsByWidth(gray, W, H, box);
    const varScores = varianceScoreGrid(grad, W, H, box);
    const widthScores = widthScoreGrid(gray, W, H, box);
    const isHoldout = HOLDOUT.includes(name);

    for (let r=0;r<8;r++) for (let c=0;c<8;c++) {
      const label = labelAt(info.rows, info.orientation, r, c);
      const truePiece = label !== '-';
      total++;
      const varPred = varGrid[r][c], widthPred = widthGrid[r][c];
      samples.push({
        varRaw: varScores[r][c], widthScore: widthScores[r][c],
        varPred, widthPred, truePiece, isHoldout,
      });
    }
    console.log(`${name}: done`);
  }

  function prf(tp,fp,fn) { const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0; return { p, r, f1: 2*p*r/(p+r)||0 }; }
  function statsFor(preds, subset) {
    let tp=0,fp=0,fn=0,correct=0;
    for (const s of subset) {
      const pred = preds(s);
      if (pred === s.truePiece) correct++;
      if (s.truePiece && pred) tp++; if (!s.truePiece && pred) fp++; if (s.truePiece && !pred) fn++;
    }
    return { ...prf(tp,fp,fn), acc: correct/subset.length };
  }
  function report(label, subset) {
    const v = statsFor(s=>s.varPred, subset);
    const w = statsFor(s=>s.widthPred, subset);
    const or_ = statsFor(s=>s.varPred||s.widthPred, subset);
    const and_ = statsFor(s=>s.varPred&&s.widthPred, subset);
    console.log(`\n[${label}] n=${subset.length}`);
    console.log(`  Variance:  accuracy=${(v.acc*100).toFixed(1)}%  precision=${(v.p*100).toFixed(1)}%  recall=${(v.r*100).toFixed(1)}%  F1=${(v.f1*100).toFixed(1)}%`);
    console.log(`  Width:     accuracy=${(w.acc*100).toFixed(1)}%  precision=${(w.p*100).toFixed(1)}%  recall=${(w.r*100).toFixed(1)}%  F1=${(w.f1*100).toFixed(1)}%`);
    console.log(`  OR:        accuracy=${(or_.acc*100).toFixed(1)}%  precision=${(or_.p*100).toFixed(1)}%  recall=${(or_.r*100).toFixed(1)}%  F1=${(or_.f1*100).toFixed(1)}%`);
    console.log(`  AND:       accuracy=${(and_.acc*100).toFixed(1)}%  precision=${(and_.p*100).toFixed(1)}%  recall=${(and_.r*100).toFixed(1)}%  F1=${(and_.f1*100).toFixed(1)}%`);
  }

  const trainSamples = samples.filter(s => !s.isHoldout);
  const holdSamples = samples.filter(s => s.isHoldout);
  report('TRAIN (tuning set)', trainSamples);
  report('HOLDOUT (never tuned on)', holdSamples);

  // --- Weighted combination: blend the two methods' confidence SCORES and
  // threshold the blend, instead of hard AND/OR on booleans. Normalize the
  // variance score using ONLY the training set's min/max (so the holdout
  // set can't leak into how the scores get scaled), then apply that same
  // scaling to holdout samples for the final validation.
  const trainVarVals = trainSamples.map(s => s.varRaw);
  const varMin = Math.min(...trainVarVals), varMax = Math.max(...trainVarVals);
  const normVar = raw => Math.max(0, Math.min(1, (raw - varMin) / (varMax - varMin || 1)));
  for (const s of samples) s.varNorm = normVar(s.varRaw);
  // widthScore is already in [0,1] by construction.

  // Tune weight + threshold on TRAIN ONLY.
  let best = null;
  for (let wI = 0; wI <= 20; wI++) {
    const weight = wI / 20; // 0 = width only, 1 = variance only
    const combined = trainSamples.map(s => weight*s.varNorm + (1-weight)*s.widthScore);
    const thresholds = [...new Set(combined.map(v => Math.round(v*200)/200))].sort((a,b)=>a-b);
    for (const th of thresholds) {
      let tp=0, fp=0, fn=0, correct=0;
      for (let i=0;i<trainSamples.length;i++) {
        const pred = combined[i] > th;
        if (pred === trainSamples[i].truePiece) correct++;
        if (trainSamples[i].truePiece && pred) tp++;
        if (!trainSamples[i].truePiece && pred) fp++;
        if (trainSamples[i].truePiece && !pred) fn++;
      }
      const p = tp/(tp+fp)||0, r = tp/(tp+fn)||0, f1 = 2*p*r/(p+r)||0;
      const acc = correct/trainSamples.length;
      if (!best || f1 > best.f1) best = { weight, th, p, r, f1, acc };
    }
  }
  console.log(`\nWeighted combo tuned on TRAIN ONLY:`);
  console.log(`  weight=${best.weight.toFixed(2)} (0=width-only,1=variance-only)  threshold=${best.th.toFixed(3)}`);
  console.log(`  train accuracy=${(best.acc*100).toFixed(1)}%  precision=${(best.p*100).toFixed(1)}%  recall=${(best.r*100).toFixed(1)}%  F1=${(best.f1*100).toFixed(1)}%`);

  // Now VALIDATE that fixed weight+threshold on the holdout set it never saw.
  const holdCombined = holdSamples.map(s => best.weight*s.varNorm + (1-best.weight)*s.widthScore);
  let htp=0, hfp=0, hfn=0, hcorrect=0;
  for (let i=0;i<holdSamples.length;i++) {
    const pred = holdCombined[i] > best.th;
    if (pred === holdSamples[i].truePiece) hcorrect++;
    if (holdSamples[i].truePiece && pred) htp++;
    if (!holdSamples[i].truePiece && pred) hfp++;
    if (holdSamples[i].truePiece && !pred) hfn++;
  }
  const hStats = prf(htp,hfp,hfn);
  console.log(`\nWeighted combo VALIDATED on HOLDOUT (same weight/threshold, unseen photos):`);
  console.log(`  accuracy=${(hcorrect/holdSamples.length*100).toFixed(1)}%  precision=${(hStats.p*100).toFixed(1)}%  recall=${(hStats.r*100).toFixed(1)}%  F1=${(hStats.f1*100).toFixed(1)}%`);
}
main().catch(e => { console.error(e); process.exit(1); });
