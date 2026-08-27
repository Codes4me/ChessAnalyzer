// Shadow/glare removal, tested SEPARATELY from the piece-width detector
// (per instruction: "add this separately, as a different test").
//
// Idea: a shadow or glare patch is diffuse lighting, not a piece, so it
// tends to spread across MULTIPLE squares in both the board's length and
// width directions (unlike a piece, which sits inside one square). We:
//   1. Build an 8x8 cell-brightness grid.
//   2. Flag cells that are much brighter/darker than the board's own
//      median brightness.
//   3. If the flagged cells' bounding box spans >=2 rows AND >=2 cols,
//      call it a lighting artifact (not pieces) and estimate its shape.
//   4. Model the artifact as a blurred blob and SUBTRACT it from the
//      image (flat-field correction), stretched elliptically to match
//      the board's own row/col pitch -- which already encodes the
//      camera's viewing angle, since a tilted photo compresses one
//      axis of the board more than the other.
//   5. Compare occupancy-detection accuracy (the existing gradient-
//      variance method) before vs after correction, on real photos.
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const { buildGrad } = require('./quad_segment_score.js');
const { correctRotateDeg } = require('./orientation_fix.js');

// --- Ground truth mapping (same convention as piece_width_detect.js) ---
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

// --- Cell brightness grid (whole cell, not just the "inner" crop used for
// the piece-variance detector, since we want the raw lighting field) ---
function cellBrightnessGrid(gray, W, H, box) {
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const grid = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const x0=box.x0+c*pitchX, x1=box.x0+(c+1)*pitchX, y0=box.y0+r*pitchY, y1=box.y0+(r+1)*pitchY;
      let s=0,n=0;
      for (let y=Math.round(y0); y<Math.round(y1); y++) {
        if (y<0||y>=H) continue;
        for (let x=Math.round(x0); x<Math.round(x1); x++) {
          if (x<0||x>=W) continue;
          s+=gray[y*W+x]; n++;
        }
      }
      row.push(n?s/n:0);
    }
    grid.push(row);
  }
  return grid;
}

function median(arr) { const s=[...arr].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; }

// Average gradient-magnitude within a cell bounding box -- a real piece
// row has crisp edges (high gradient), while diffuse shadow/glare does not,
// even though both can show up as a multi-row/col brightness deviation.
function cellGradGrid(grad, W, H, box) {
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const grid = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const x0=box.x0+c*pitchX, x1=box.x0+(c+1)*pitchX, y0=box.y0+r*pitchY, y1=box.y0+(r+1)*pitchY;
      let s=0,n=0;
      for (let y=Math.round(y0); y<Math.round(y1); y++) {
        if (y<0||y>=H) continue;
        for (let x=Math.round(x0); x<Math.round(x1); x++) {
          if (x<0||x>=W) continue;
          s+=grad[y*W+x]; n++;
        }
      }
      row.push(n?s/n:0);
    }
    grid.push(row);
  }
  return grid;
}

// Detects whether an out-of-range brightness blob spans >=2 rows AND >=2
// cols of cells -- the signature of diffuse lighting rather than a piece --
// AND is smooth (low gradient) rather than crisp-edged like real pieces.
// Without the smoothness check, a plain starting position (a whole row of
// light pieces next to a whole row of dark ones) trips this just as
// easily as an actual shadow, since both look like a "wide brightness
// band" from cell-average brightness alone.
function detectLightingArtifact(cellBright, cellGradAvg) {
  const flat = cellBright.flat();
  const med = median(flat);
  const spread = Math.sqrt(flat.reduce((s,v)=>s+(v-med)**2,0)/flat.length) || 1;
  const gradFlat = cellGradAvg.flat();
  const gradMean = gradFlat.reduce((a,b)=>a+b,0)/gradFlat.length || 1;

  const THRESH = 1.5; // std-devs from median counts as "extreme" brightness
  const SMOOTH_MAX = 0.6; // flagged cells' own gradient must be well below the board's average
  let minR=8,maxR=-1,minC=8,maxC=-1,count=0,sign=0;
  for (let r=0;r<8;r++) for (let c=0;c<8;c++) {
    const z = (cellBright[r][c]-med)/spread;
    const smooth = cellGradAvg[r][c] < gradMean*SMOOTH_MAX;
    if (Math.abs(z) > THRESH && smooth) {
      minR=Math.min(minR,r); maxR=Math.max(maxR,r);
      minC=Math.min(minC,c); maxC=Math.max(maxC,c);
      count++; sign += Math.sign(z);
    }
  }
  const rowSpan = count ? maxR-minR+1 : 0, colSpan = count ? maxC-minC+1 : 0;
  const isArtifact = count >= 2 && rowSpan >= 2 && colSpan >= 2;
  return {
    isArtifact, med, spread,
    kind: sign >= 0 ? 'bright (glare)' : 'dark (shadow)',
    bbox: isArtifact ? { minR, maxR, minC, maxC } : null,
  };
}

// Elliptical box blur -- radius in x/y independently scaled by the board's
// own pitchX/pitchY, so a tilted photo (where one axis is compressed by
// perspective) gets a correspondingly squashed "circle" (i.e. an ellipse),
// rather than a naive symmetric blur.
function ellipticalBoxBlur(gray, W, H, rx, ry) {
  rx = Math.max(1, Math.round(rx)); ry = Math.max(1, Math.round(ry));
  // Horizontal pass then vertical pass (separable box blur).
  const tmp = new Float64Array(W*H);
  for (let y=0;y<H;y++) {
    let sum=0, cnt=0;
    for (let x=-rx; x<=rx; x++) { const xi=Math.min(W-1,Math.max(0,x)); sum+=gray[y*W+xi]; cnt++; }
    for (let x=0;x<W;x++) {
      tmp[y*W+x] = sum/cnt;
      const addX = Math.min(W-1, x+rx+1), remX = Math.max(0, x-rx);
      sum += gray[y*W+addX] - gray[y*W+remX];
    }
  }
  const out = new Float64Array(W*H);
  for (let x=0;x<W;x++) {
    let sum=0, cnt=0;
    for (let y=-ry; y<=ry; y++) { const yi=Math.min(H-1,Math.max(0,y)); sum+=tmp[yi*W+x]; cnt++; }
    for (let y=0;y<H;y++) {
      out[y*W+x] = sum/cnt;
      const addY = Math.min(H-1, y+ry+1), remY = Math.max(0, y-ry);
      sum += tmp[addY*W+x] - tmp[remY*W+x];
    }
  }
  return out;
}

// Flat-field correction: subtract the large-scale blurred "lighting field"
// and re-add the global mean, so overall brightness is preserved but the
// diffuse bright/dark blob is flattened out.
function correctIllumination(gray, W, H, pitchX, pitchY) {
  // Blur radius: a couple of cells wide/tall, so piece-scale detail
  // survives but board-scale lighting gradients don't.
  const rx = pitchX*2, ry = pitchY*2;
  const blurred = ellipticalBoxBlur(gray, W, H, rx, ry);
  let mean=0; for (let i=0;i<gray.length;i++) mean+=gray[i]; mean/=gray.length;
  const out = new Float64Array(W*H);
  for (let i=0;i<gray.length;i++) out[i] = Math.max(0, Math.min(255, gray[i]-blurred[i]+mean));
  return out;
}

// --- Reuse the gradient-variance occupancy detector, but on a supplied
// grayscale buffer (so we can run it on both original and corrected). ---
function detectPieceCellsByVariance(grad, W, H, box) {
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const cellGrad = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const cx0=box.x0+c*pitchX, cx1=box.x0+(c+1)*pitchX, cy0=box.y0+r*pitchY, cy1=box.y0+(r+1)*pitchY;
      const innerX0=cx0+pitchX*0.2, innerX1=cx1-pitchX*0.2, innerY0=cy0+pitchY*0.2, innerY1=cy1-pitchY*0.2;
      let s=0,n=0;
      for (let y=Math.round(innerY0); y<Math.round(innerY1); y++) {
        if (y<0||y>=H) continue;
        for (let x=Math.round(innerX0); x<Math.round(innerX1); x++) {
          if (x<0||x>=W) continue;
          s+=grad[y*W+x]; n++;
        }
      }
      row.push(n?s/n:0);
    }
    cellGrad.push(row);
  }
  const flat = cellGrad.flat();
  const mean = flat.reduce((a,b)=>a+b,0)/flat.length;
  return cellGrad.map(row => row.map(v => v > mean*1.3));
}

// Recompute a Sobel gradient-magnitude map from a plain grayscale buffer
// (buildGrad in quad_segment_score.js reads straight from a file, but here
// we need it on our OWN corrected pixel buffer).
function sobelGradFromGray(gray, W, H) {
  const grad = new Float64Array(W*H);
  for (let y=1;y<H-1;y++) for (let x=1;x<W-1;x++) {
    const i=y*W+x;
    const gx=gray[i-W+1]+2*gray[i+1]+gray[i+W+1]-gray[i-W-1]-2*gray[i-1]-gray[i+W-1];
    const gy=gray[i-W-1]+2*gray[i-W]+gray[i-W+1]-gray[i+W-1]-2*gray[i+W]-gray[i+W+1];
    grad[i]=Math.hypot(gx,gy);
  }
  return grad;
}

// --- Width detector (ported from piece_width_detect.js), run against a
// plain grayscale buffer -- this is the one worth testing against
// illumination correction, since it thresholds gradient magnitude against a
// fixed ABSOLUTE value (CONTRAST_THRESHOLD), unlike the variance method
// which already compares each cell to the photo's own mean and so is fairly
// insensitive to slow brightness gradients. A shadow/glare region can push
// real edges below (or fake edges above) that fixed threshold.
const WIDTH_CONTRAST_THRESHOLD = 20;
const WIDTH_FRAC_MIN = 0.3, WIDTH_FRAC_MAX = 0.9;
const WIDTH_ANGLES_DEG = [0, 30, 60, 90, 120, 150];

function buildGxGyFromGray(gray, W, H) {
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
    if (samples[i] > WIDTH_CONTRAST_THRESHOLD && samples[i] >= samples[i-1] && samples[i] >= samples[i+1]) {
      edgeTs.push(-halfLen + i*step);
    }
  }
  if (edgeTs.length < 2) return null;
  return Math.max(...edgeTs) - Math.min(...edgeTs);
}
function detectPieceCellsByWidth(gray, W, H, box) {
  const { gx, gy } = buildGxGyFromGray(gray, W, H);
  const pitchX = (box.x1-box.x0)/8, pitchY = (box.y1-box.y0)/8;
  const pitch = (pitchX+pitchY)/2;
  const result = [];
  for (let r=0;r<8;r++) {
    const row = [];
    for (let c=0;c<8;c++) {
      const cx = box.x0 + (c+0.5)*pitchX, cy = box.y0 + (r+0.5)*pitchY;
      let smallestWidth = Infinity;
      for (const deg of WIDTH_ANGLES_DEG) {
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

function prf(tp,fp,fn) { const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0; return { p, r, f1: 2*p*r/(p+r)||0 }; }

async function main() {
  const gt = JSON.parse(fs.readFileSync('ground_truth.json', 'utf8'));
  let totalCells=0;
  let flaggedPhotos = 0;
  // Variance-method (already scale-normalized per photo) and width-method
  // (fixed absolute contrast threshold) tallies, before/after correction.
  const tally = {
    varBefore: {c:0,tp:0,fp:0,fn:0}, varAfter: {c:0,tp:0,fp:0,fn:0},
    widthBefore: {c:0,tp:0,fp:0,fn:0}, widthAfter: {c:0,tp:0,fp:0,fn:0},
  };
  function score(bucket, pred, truePiece) {
    if (pred===truePiece) bucket.c++;
    if (truePiece&&pred) bucket.tp++; if (!truePiece&&pred) bucket.fp++; if (truePiece&&!pred) bucket.fn++;
  }

  for (const [name, info] of Object.entries(PHOTOS)) {
    const entry = gt[name];
    if (!entry) continue;
    const imgPath = path.join(__dirname, 'Data', name);
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

    const varBeforeGrid = detectPieceCellsByVariance(grad, W, H, box);
    const widthBeforeGrid = detectPieceCellsByWidth(gray, W, H, box);

    let corrGray = gray, corrGrad = grad;
    if (artifact.isArtifact) {
      flaggedPhotos++;
      corrGray = correctIllumination(gray, W, H, pitchX, pitchY);
      corrGrad = sobelGradFromGray(corrGray, W, H);
    }
    const varAfterGrid = detectPieceCellsByVariance(corrGrad, W, H, box);
    const widthAfterGrid = detectPieceCellsByWidth(corrGray, W, H, box);

    let vB=0, vA=0, wB=0, wA=0;
    for (let r=0;r<8;r++) for (let c=0;c<8;c++) {
      const truePiece = labelAt(info.rows, info.orientation, r, c) !== '-';
      score(tally.varBefore, varBeforeGrid[r][c], truePiece); if (varBeforeGrid[r][c]===truePiece) vB++;
      score(tally.varAfter, varAfterGrid[r][c], truePiece); if (varAfterGrid[r][c]===truePiece) vA++;
      score(tally.widthBefore, widthBeforeGrid[r][c], truePiece); if (widthBeforeGrid[r][c]===truePiece) wB++;
      score(tally.widthAfter, widthAfterGrid[r][c], truePiece); if (widthAfterGrid[r][c]===truePiece) wA++;
      totalCells++;
    }

    const tag = artifact.isArtifact ? `ARTIFACT (${artifact.kind}, rows ${artifact.bbox.minR}-${artifact.bbox.maxR} cols ${artifact.bbox.minC}-${artifact.bbox.maxC})` : 'no artifact';
    console.log(`${name}: ${tag}  |  variance ${(vB/64*100).toFixed(1)}%->${(vA/64*100).toFixed(1)}%  |  width ${(wB/64*100).toFixed(1)}%->${(wA/64*100).toFixed(1)}%`);
  }

  console.log(`\n${flaggedPhotos}/${Object.keys(PHOTOS).length} photos flagged with a shadow/glare artifact.\n`);
  function report(label, before, after) {
    const bs = prf(before.tp,before.fp,before.fn), as = prf(after.tp,after.fp,after.fn);
    console.log(`${label}:`);
    console.log(`  before: accuracy=${(before.c/totalCells*100).toFixed(1)}%  precision=${(bs.p*100).toFixed(1)}%  recall=${(bs.r*100).toFixed(1)}%  F1=${(bs.f1*100).toFixed(1)}%`);
    console.log(`  after:  accuracy=${(after.c/totalCells*100).toFixed(1)}%  precision=${(as.p*100).toFixed(1)}%  recall=${(as.r*100).toFixed(1)}%  F1=${(as.f1*100).toFixed(1)}%`);
  }
  report('Variance method', tally.varBefore, tally.varAfter);
  report('Width method', tally.widthBefore, tally.widthAfter);
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { detectLightingArtifact, correctIllumination, cellBrightnessGrid, cellGradGrid, ellipticalBoxBlur, detectPieceCellsByWidth };
