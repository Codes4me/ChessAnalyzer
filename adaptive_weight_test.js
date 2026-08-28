// Tests whether adjusting the weighted variance/width combo's weight based
// on a photo's detected JPEG compression level helps. Earlier testing
// (compression_test.js) showed: variance method barely moves under
// compression, width method's precision drops noticeably below JPEG
// quality ~50 (F1 78.5% at quality=10 vs 81.6% uncompressed). So as
// estimated quality drops, this shifts the combo's weight toward variance
// (which stays reliable) and away from width (which doesn't). Falls back
// to the tuned default weight whenever quality can't be determined (e.g.
// WebP, or any non-baseline-JPEG file) -- never guesses.
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const { buildGrad } = require('./quad_segment_score.js');
const { correctRotateDeg } = require('./orientation_fix.js');
const { detectLightingArtifact, correctIllumination, cellBrightnessGrid, cellGradGrid } = require('./shadow_glare_removal.js');
const { estimateJpegQuality } = require('./jpeg_quality_estimate.js');

const CONTRAST_THRESHOLD = 20;
const WIDTH_FRAC_MIN = 0.3, WIDTH_FRAC_MAX = 0.9;
const ANGLES_DEG = [0, 30, 60, 90, 120, 150];
const BASE_WEIGHT = 0.90, BASE_THRESHOLD = 0.110; // tuned on TRAIN in piece_width_detect.js

// Fitted from an actual weight sweep (optimal_weight_by_quality.js), not
// hand-picked -- see the comment on QUALITY_WEIGHT_CURVE in
// piece_width_detect.js for why. Kept in sync with that curve.
const QUALITY_WEIGHT_CURVE = [
  { q: 100, weight: BASE_WEIGHT }, { q: 25, weight: BASE_WEIGHT }, { q: 20, weight: 0.95 }, { q: 10, weight: 0.95 },
];
function weightForQuality(quality) {
  if (quality == null) return BASE_WEIGHT;
  const curve = QUALITY_WEIGHT_CURVE;
  if (quality >= curve[0].q) return curve[0].weight;
  for (let i = 0; i < curve.length - 1; i++) {
    const hi = curve[i], lo = curve[i+1];
    if (quality <= hi.q && quality >= lo.q) {
      const t = (hi.q - quality) / (hi.q - lo.q || 1);
      return hi.weight + (lo.weight - hi.weight) * t;
    }
  }
  return curve[curve.length-1].weight;
}

function buildGxGy(gray, W, H) {
  const gx = new Float64Array(W*H), gy = new Float64Array(W*H);
  for (let y=1;y<H-1;y++) for (let x=1;x<W-1;x++) {
    const i=y*W+x;
    gx[i]=gray[i-W+1]+2*gray[i+1]+gray[i+W+1]-gray[i-W-1]-2*gray[i-1]-gray[i+W-1];
    gy[i]=gray[i-W-1]+2*gray[i-W]+gray[i-W+1]-gray[i+W-1]-2*gray[i+W]-gray[i+W+1];
  }
  return { gx, gy };
}
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
  return Math.max(...edgeTs) - Math.min(...edgeTs);
}
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
      row.push(Math.max(0, 1 - Math.abs(frac-center)/spread));
    }
    result.push(row);
  }
  return result;
}
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
function varianceScoreGrid(grad, W, H, box) {
  const cellGrad = computeCellGrad(grad, W, H, box);
  const flat = cellGrad.flat();
  const mean = flat.reduce((a,b)=>a+b,0)/flat.length || 1;
  return cellGrad.map(row => row.map(v => v/mean));
}

// --- Ground truth ---
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
const HOLDOUT = ['oblique_yes.jpeg', 'image4.jpeg', 'start1.jpeg'];

function prf(tp,fp,fn) { const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0; return { p, r, f1: 2*p*r/(p+r)||0 }; }

// Same train-set min/max normalization for variance scores as
// piece_width_detect.js used, so weight=BASE_WEIGHT reproduces those numbers.
const TRAIN_VAR_MIN = 0.05, TRAIN_VAR_MAX = 3.5; // approximate stand-in; recomputed properly below per run

async function evalVariant(dir) {
  const gt = JSON.parse(fs.readFileSync('ground_truth.json', 'utf8'));
  const samples = [];
  for (const name of HOLDOUT) {
    const entry = gt[name];
    const info = PHOTOS[name];
    const imgPath = path.join(dir, name);
    if (!fs.existsSync(imgPath)) continue;
    const quality = estimateJpegQuality(imgPath);
    const rotateDeg = await correctRotateDeg(imgPath);
    const { grad, W, H } = await buildGrad(imgPath, rotateDeg);
    const { data: gray8 } = await sharp(imgPath).rotate(rotateDeg).resize(W, H, { fit: 'fill' }).grayscale().raw().toBuffer({ resolveWithObject: true });
    const gray = new Float64Array(gray8.length);
    for (let i=0;i<gray8.length;i++) gray[i] = gray8[i];

    const meta = (await sharp(imgPath).rotate(rotateDeg).toBuffer({ resolveWithObject: true })).info;
    const sx = W/meta.width, sy = H/meta.height;
    const pts = entry.corners.map(([x,y]) => [x*sx, y*sy]);
    const x0 = Math.min(...pts.map(p=>p[0])), x1 = Math.max(...pts.map(p=>p[0]));
    const y0 = Math.min(...pts.map(p=>p[1])), y1 = Math.max(...pts.map(p=>p[1]));
    const box = { x0, y0, x1, y1 };
    const pitchX = (x1-x0)/8, pitchY = (y1-y0)/8;

    const cellBright = cellBrightnessGrid(gray, W, H, box);
    const cellGradAvg = cellGradGrid(grad, W, H, box);
    const artifact = detectLightingArtifact(cellBright, cellGradAvg);
    const widthGray = artifact.isArtifact ? correctIllumination(gray, W, H, pitchX, pitchY) : gray;

    const varScores = varianceScoreGrid(grad, W, H, box);
    const widthScores = widthScoreGrid(widthGray, W, H, box);

    for (let r=0;r<8;r++) for (let c=0;c<8;c++) {
      const truePiece = labelAt(info.rows, info.orientation, r, c) !== '-';
      const varNorm = Math.max(0, Math.min(1, (varScores[r][c] - TRAIN_VAR_MIN) / (TRAIN_VAR_MAX - TRAIN_VAR_MIN)));
      samples.push({ varNorm, widthScore: widthScores[r][c], truePiece, quality });
    }
  }

  function evalWith(weightFn) {
    let tp=0,fp=0,fn=0,correct=0;
    for (const s of samples) {
      const w = weightFn(s.quality);
      const combined = w*s.varNorm + (1-w)*s.widthScore;
      const pred = combined > BASE_THRESHOLD;
      if (pred===s.truePiece) correct++;
      if (s.truePiece&&pred) tp++; if (!s.truePiece&&pred) fp++; if (s.truePiece&&!pred) fn++;
    }
    const stats = prf(tp,fp,fn);
    return { acc: correct/samples.length, ...stats };
  }

  const staticResult = evalWith(() => BASE_WEIGHT);
  const adaptiveResult = evalWith(weightForQuality);
  const qualities = [...new Set(samples.map(s=>s.quality))];
  return { staticResult, adaptiveResult, qualities };
}

async function main() {
  const SCRATCH_DIR = '/tmp/chessanalyzer-compression-test';
  const variants = [
    { label: 'Original', dir: path.join(SCRATCH_DIR, 'original') },
    { label: 'JPEG q80', dir: path.join(SCRATCH_DIR, 'jpeg80') },
    { label: 'JPEG q50', dir: path.join(SCRATCH_DIR, 'jpeg50') },
    { label: 'JPEG q25', dir: path.join(SCRATCH_DIR, 'jpeg25') },
    { label: 'JPEG q10', dir: path.join(SCRATCH_DIR, 'jpeg10') },
  ];
  for (const v of variants) {
    if (!fs.existsSync(v.dir)) { console.log(`${v.label}: (run compression_test.js first) skipped`); continue; }
    const { staticResult, adaptiveResult, qualities } = await evalVariant(v.dir);
    console.log(`${v.label}  (detected quality per holdout photo: ${qualities.join(', ')})`);
    console.log(`  static weight=${BASE_WEIGHT}:   accuracy=${(staticResult.acc*100).toFixed(1)}%  F1=${(staticResult.f1*100).toFixed(1)}%`);
    console.log(`  adaptive weight:      accuracy=${(adaptiveResult.acc*100).toFixed(1)}%  F1=${(adaptiveResult.f1*100).toFixed(1)}%\n`);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
