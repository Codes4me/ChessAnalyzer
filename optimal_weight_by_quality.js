// Finds the ACTUALLY-best combo weight at each JPEG compression level, by
// sweeping weight against compressed copies of the TRAINING photos only
// (holdout stays untouched for a fair final check). The previous
// weightForQuality() in piece_width_detect.js was a hand-picked linear
// ramp, not tuned against data -- this replaces the guess with a real
// per-quality optimum, then validates the fitted curve against the
// holdout's own compressed copies to see if it actually generalizes better.
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { buildGrad } = require('./quad_segment_score.js');
const { correctRotateDeg } = require('./orientation_fix.js');
const { detectLightingArtifact, correctIllumination, cellBrightnessGrid, cellGradGrid } = require('./shadow_glare_removal.js');
const { estimateJpegQuality } = require('./jpeg_quality_estimate.js');

const CONTRAST_THRESHOLD = 20;
const WIDTH_FRAC_MIN = 0.3, WIDTH_FRAC_MAX = 0.9;
const ANGLES_DEG = [0, 30, 60, 90, 120, 150];
const BASE_THRESHOLD = 0.110; // tuned threshold from piece_width_detect.js, held fixed here

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
const TRAIN_NAMES = Object.keys(PHOTOS).filter(n => !HOLDOUT.includes(n));

function prf(tp,fp,fn) { const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0; return { p, r, f1: 2*p*r/(p+r)||0 }; }

async function makeCompressed(imgPath, quality, outPath) {
  await sharp(imgPath).withMetadata().jpeg({ quality }).toFile(outPath);
}

// Builds raw {varRaw, widthScore, truePiece} samples for a set of photos
// living in `dir` (already-compressed copies, or originals).
async function buildSamples(dir, names, gt) {
  const samples = [];
  for (const name of names) {
    const entry = gt[name];
    const info = PHOTOS[name];
    const imgPath = path.join(dir, name);
    if (!fs.existsSync(imgPath)) continue;
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
      samples.push({ varRaw: varScores[r][c], widthScore: widthScores[r][c], truePiece });
    }
  }
  return samples;
}

function bestWeightFor(samples, varMin, varMax) {
  let best = null;
  for (let wI = 0; wI <= 20; wI++) {
    const weight = wI / 20;
    let tp=0,fp=0,fn=0,correct=0;
    for (const s of samples) {
      const varNorm = Math.max(0, Math.min(1, (s.varRaw - varMin) / (varMax - varMin || 1)));
      const combined = weight*varNorm + (1-weight)*s.widthScore;
      const pred = combined > BASE_THRESHOLD;
      if (pred===s.truePiece) correct++;
      if (s.truePiece&&pred) tp++; if (!s.truePiece&&pred) fp++; if (s.truePiece&&!pred) fn++;
    }
    const stats = prf(tp,fp,fn);
    const acc = correct/samples.length;
    if (!best || stats.f1 > best.f1) best = { weight, acc, ...stats };
  }
  return best;
}

async function main() {
  const gt = JSON.parse(fs.readFileSync('ground_truth.json', 'utf8'));
  const SCRATCH_DIR = path.join(os.tmpdir(), 'chessanalyzer-optimal-weight');
  fs.mkdirSync(SCRATCH_DIR, { recursive: true });

  // Fixed variance normalization range, computed from ORIGINAL (uncompressed)
  // train photos -- matches piece_width_detect.js's approach: this scaling
  // must be decided once, independent of a fresh photo's unknown quality.
  const origSamples = await buildSamples(path.join(__dirname, 'Data'), TRAIN_NAMES, gt);
  const origVarVals = origSamples.map(s => s.varRaw);
  const varMin = Math.min(...origVarVals), varMax = Math.max(...origVarVals);

  const QUALITIES = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10];
  const curve = [];
  console.log('Optimal weight by JPEG quality (found by sweeping on TRAIN photos only):\n');
  for (const q of QUALITIES) {
    let samples;
    if (q === 100) {
      samples = origSamples;
    } else {
      const qDir = path.join(SCRATCH_DIR, `q${q}`);
      fs.mkdirSync(qDir, { recursive: true });
      for (const name of TRAIN_NAMES) {
        const src = path.join(__dirname, 'Data', name);
        if (fs.existsSync(src)) await makeCompressed(src, q, path.join(qDir, name));
      }
      samples = await buildSamples(qDir, TRAIN_NAMES, gt);
    }
    const best = bestWeightFor(samples, varMin, varMax);
    curve.push({ q, weight: best.weight });
    console.log(`  quality=${q}: best weight=${best.weight.toFixed(2)}  (F1=${(best.f1*100).toFixed(1)}%, accuracy=${(best.acc*100).toFixed(1)}%)`);
  }

  // Fit a simple piecewise-linear lookup from the discovered curve, then
  // validate it against the HOLDOUT photos' own compressed copies (never
  // touched during the sweep above) alongside the original hand-picked ramp.
  function fittedWeight(quality) {
    if (quality == null) return curve.find(c => c.q === 100).weight;
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
  function handPickedWeight(quality) {
    const BASE = 0.90, CEIL = 50, FLOOR = 10;
    if (quality == null || quality >= CEIL) return BASE;
    const t = Math.max(0, Math.min(1, (CEIL - quality) / (CEIL - FLOOR)));
    return BASE + (1 - BASE) * t;
  }

  console.log('\nValidating fitted curve vs the original hand-picked ramp on HOLDOUT photos (never used for fitting):\n');
  const HOLDOUT_SCRATCH = path.join(SCRATCH_DIR, 'holdout');
  for (const q of [100, 50, 25, 10]) {
    let samples, quality;
    if (q === 100) {
      samples = await buildSamples(path.join(__dirname, 'Data'), HOLDOUT, gt);
      quality = 100;
    } else {
      const qDir = path.join(HOLDOUT_SCRATCH, `q${q}`);
      fs.mkdirSync(qDir, { recursive: true });
      for (const name of HOLDOUT) {
        const src = path.join(__dirname, 'Data', name);
        if (fs.existsSync(src)) await makeCompressed(src, q, path.join(qDir, name));
      }
      samples = await buildSamples(qDir, HOLDOUT, gt);
      quality = estimateJpegQuality(path.join(qDir, HOLDOUT[0]));
    }
    function evalAt(weight) {
      let tp=0,fp=0,fn=0,correct=0;
      for (const s of samples) {
        const varNorm = Math.max(0, Math.min(1, (s.varRaw - varMin) / (varMax - varMin || 1)));
        const combined = weight*varNorm + (1-weight)*s.widthScore;
        const pred = combined > BASE_THRESHOLD;
        if (pred===s.truePiece) correct++;
        if (s.truePiece&&pred) tp++; if (!s.truePiece&&pred) fp++; if (s.truePiece&&!pred) fn++;
      }
      const stats = prf(tp,fp,fn);
      return { acc: correct/samples.length, f1: stats.f1 };
    }
    const handW = handPickedWeight(quality), fitW = fittedWeight(quality);
    const handR = evalAt(handW), fitR = evalAt(fitW);
    console.log(`  quality~${quality}: hand-picked weight=${handW.toFixed(2)} -> F1=${(handR.f1*100).toFixed(1)}%  |  fitted weight=${fitW.toFixed(2)} -> F1=${(fitR.f1*100).toFixed(1)}%`);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
