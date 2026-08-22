// Standalone Node test harness that replicates the browser matching algorithm
// in public/index.html, so it can be run against real files without going
// through the UI. Not part of the app itself — a debugging tool.
const sharp = require('sharp');

const SQ_SIZE = 24;
const CELL_RENDER_SIZE = 32;
const CELL_MARGIN = (CELL_RENDER_SIZE - SQ_SIZE) / 2;

const STARTING_FEN_ROWS = [
  'rnbqkbnr',
  'pppppppp',
  '--------',
  '--------',
  '--------',
  '--------',
  'PPPPPPPP',
  'RNBQKBNR'
];

const PIECE_LABELS = ['K','Q','R','B','N','P','k','q','r','b','n','p'];
const PIECE_UNICODE = { K:'K',Q:'Q',R:'R',B:'B',N:'N',P:'P',k:'k',q:'q',r:'r',b:'b',n:'n',p:'p' };

function refKey(piece, squareColor) {
  return (piece || 'empty') + '_' + squareColor;
}

function squareColorAt(row, col) {
  return (row + col) % 2 === 0 ? 'light' : 'dark';
}

function toGrayArray(rgbaData, channels) {
  const gray = new Array(SQ_SIZE * SQ_SIZE);
  for (let i = 0; i < gray.length; i++) {
    const r = rgbaData[i * channels], g = rgbaData[i * channels + 1], b = rgbaData[i * channels + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return gray;
}

// Combines edge-density and checkerboard-alternation signals (see
// public/index.html for the full rationale — each has an opposite blind
// spot alone; combining cancels both out). Kept in sync with the app.
async function autoDetectCropRect(imgPath, rotateDeg) {
  const WORK_SIZE = 240;
  let pipeline = sharp(imgPath);
  if (rotateDeg) pipeline = pipeline.rotate(rotateDeg);
  const { data } = await pipeline.resize(WORK_SIZE, WORK_SIZE, { fit: 'fill' }).grayscale().raw().toBuffer({ resolveWithObject: true });
  const W = WORK_SIZE, H = WORK_SIZE;

  const grad = new Float64Array(W * H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = data[i-W+1]+2*data[i+1]+data[i+W+1]-data[i-W-1]-2*data[i-1]-data[i+W-1];
      const gy = data[i-W-1]+2*data[i-W]+data[i-W+1]-data[i+W-1]-2*data[i+W]-data[i+W+1];
      grad[i] = Math.sqrt(gx*gx+gy*gy);
    }
  }

  function altScoreForLine(getPixel, start, span, p) {
    const bins = new Array(8).fill(0);
    const binW = span / 8;
    for (let i = 0; i < 8; i++) {
      const c0 = Math.round(start + i * binW), c1 = Math.round(start + (i + 1) * binW);
      let s = 0, n = 0;
      for (let c = c0; c < c1; c++) { s += getPixel(c, p); n++; }
      bins[i] = n ? s / n : 0;
    }
    let alt = 0;
    for (let i = 0; i < 8; i++) alt += bins[i] * (i % 2 === 0 ? 1 : -1);
    const range = Math.max(...bins) - Math.min(...bins) + 1;
    return Math.abs(alt) / range;
  }
  function edgeScoreForLine(getGrad, start, span, p) {
    let s = 0, n = 0;
    for (let c = Math.round(start); c < Math.round(start + span); c++) { s += getGrad(c, p); n++; }
    return n ? s / n : 0;
  }

  const getColPixel = (x, y) => data[y * W + x];
  const getColGrad = (x, y) => grad[y * W + x];
  const getRowPixel = (y, x) => data[y * W + x];
  const getRowGrad = (y, x) => grad[y * W + x];

  const rowSamples = []; for (let y = 10; y < H - 10; y += 4) rowSamples.push(y);
  const colSamples = []; for (let x = 10; x < W - 10; x += 4) colSamples.push(x);

  function search(dim, getPixel, getGrad, perpSamples) {
    const step = Math.max(2, Math.round(dim * 0.025));
    const candidates = [];
    for (let start = 0; start < dim * 0.4; start += step) {
      for (let end = dim * 0.6; end < dim; end += step) {
        const span = end - start;
        if (span < dim * 0.3) continue;
        candidates.push({ start, end, span });
      }
    }
    let maxAlt = 0, maxEdge = 0;
    for (const cand of candidates) {
      let altSum = 0, edgeSum = 0, n = 0;
      for (const p of perpSamples) {
        altSum += altScoreForLine(getPixel, cand.start, cand.span, p);
        edgeSum += edgeScoreForLine(getGrad, cand.start, cand.span, p);
        n++;
      }
      cand.alt = altSum / n; cand.edge = edgeSum / n;
      if (cand.alt > maxAlt) maxAlt = cand.alt;
      if (cand.edge > maxEdge) maxEdge = cand.edge;
    }
    let best = null, bestScore = -1;
    for (const cand of candidates) {
      const normAlt = maxAlt > 0 ? cand.alt / maxAlt : 0;
      const normEdge = maxEdge > 0 ? cand.edge / maxEdge : 0;
      const combined = normAlt + normEdge;
      if (combined > bestScore) { bestScore = combined; best = cand; }
    }
    return [best.start, best.end];
  }

  const [x0, x1] = search(W, getColPixel, getColGrad, rowSamples);
  const [y0, y1] = search(H, getRowPixel, getRowGrad, colSamples);

  let metaPipeline = sharp(imgPath);
  if (rotateDeg) metaPipeline = metaPipeline.rotate(rotateDeg);
  const meta = await metaPipeline.toBuffer({ resolveWithObject: true });
  const naturalW = meta.info.width, naturalH = meta.info.height;
  const sx = naturalW / W, sy = naturalH / H;
  return { x: Math.round(x0 * sx), y: Math.round(y0 * sy), w: Math.round((x1 - x0) * sx), h: Math.round((y1 - y0) * sy) };
}

// ---- Perspective correction (same math as public/index.html) ----
function squareToQuadMatrix(a, d, c, e, g, n, l, p) {
  const m = c - g, h = e - n, f = l - g, k = p - n;
  const g2 = a - c + g - l, n2 = d - e + n - p;
  const q = m * k - f * h;
  const f2 = (g2 * k - f * n2) / q, m2 = (m * n2 - g2 * h) / q;
  return [c - a + f2 * c, e - d + f2 * e, f2, l - a + m2 * l, p - d + m2 * p, m2, a, d, 1];
}
function invert3x3(a) {
  const d = a[0], c = a[1], e = a[2], g = a[3], n = a[4], l = a[5], p = a[6], m = a[7], A = a[8];
  const f = d * n * A - d * l * m - c * g * A + c * l * p + e * g * m - e * n * p;
  return [
    (n * A - l * m) / f, (e * m - c * A) / f, (c * l - e * n) / f,
    (l * p - g * A) / f, (d * A - e * p) / f, (e * g - d * l) / f,
    (g * m - n * p) / f, (c * p - d * m) / f, (d * n - c * g) / f
  ];
}
function matMul3(c, e) {
  return [
    c[0]*e[0]+c[1]*e[3]+c[2]*e[6], c[0]*e[1]+c[1]*e[4]+c[2]*e[7], c[0]*e[2]+c[1]*e[5]+c[2]*e[8],
    c[3]*e[0]+c[4]*e[3]+c[5]*e[6], c[3]*e[1]+c[4]*e[4]+c[5]*e[7], c[3]*e[2]+c[4]*e[5]+c[5]*e[8],
    c[6]*e[0]+c[7]*e[3]+c[8]*e[6], c[6]*e[1]+c[7]*e[4]+c[8]*e[7], c[6]*e[2]+c[7]*e[5]+c[8]*e[8]
  ];
}
function perspectiveMatrix(before, after) {
  return matMul3(invert3x3(squareToQuadMatrix.apply(null, after)), squareToQuadMatrix.apply(null, before));
}

// Warps quad (8 numbers, natural pixel coords, tl/tr/br/bl) from the
// (rotated) source image into an outSize x outSize RGB buffer.
async function warpQuadToSquare(imgPath, rotateDeg, quad, outSize) {
  let pipeline = sharp(imgPath);
  if (rotateDeg) pipeline = pipeline.rotate(rotateDeg);
  const { data: srcData, info } = await pipeline.raw().toBuffer({ resolveWithObject: true });
  const naturalW = info.width, naturalH = info.height, channels = info.channels;

  const after = [0, 0, outSize, 0, outSize, outSize, 0, outSize];
  const M = perspectiveMatrix(quad, after);

  const out = Buffer.alloc(outSize * outSize * channels);
  for (let oy = 0; oy < outSize; oy++) {
    for (let ox = 0; ox < outSize; ox++) {
      const wx = M[0] * ox + M[3] * oy + M[6];
      const wy = M[1] * ox + M[4] * oy + M[7];
      const wz = M[2] * ox + M[5] * oy + M[8];
      const sx = wx / wz, sy = wy / wz;
      const di = (oy * outSize + ox) * channels;
      if (sx >= 0 && sx < naturalW - 1 && sy >= 0 && sy < naturalH - 1) {
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const fx = sx - x0, fy = sy - y0;
        const i00 = (y0 * naturalW + x0) * channels;
        const i10 = (y0 * naturalW + x0 + 1) * channels;
        const i01 = ((y0 + 1) * naturalW + x0) * channels;
        const i11 = ((y0 + 1) * naturalW + x0 + 1) * channels;
        for (let c = 0; c < channels; c++) {
          const top = srcData[i00 + c] * (1 - fx) + srcData[i10 + c] * fx;
          const bot = srcData[i01 + c] * (1 - fx) + srcData[i11 + c] * fx;
          out[di + c] = top * (1 - fy) + bot * fy;
        }
      }
    }
  }
  return { buffer: out, width: outSize, height: outSize, channels };
}

function rectToQuad(rect) {
  return [rect.x, rect.y, rect.x + rect.w, rect.y, rect.x + rect.w, rect.y + rect.h, rect.x, rect.y + rect.h];
}

// quad: optional explicit 8-number quad (natural coords). If omitted,
// auto-detects the board and uses its axis-aligned bounding box as the quad
// (matching the app's default when corners aren't manually dragged). Set
// env FULL_IMAGE=1 to bypass auto-detect and use the whole image instead —
// useful for testing against already-tightly-cropped source photos, where
// auto-detect's background-vs-board assumption doesn't apply.
async function sliceImageToSquares(imgPath, rotateDeg, quad) {
  if (!quad) {
    if (process.env.FULL_IMAGE) {
      let metaPipeline = sharp(imgPath);
      if (rotateDeg) metaPipeline = metaPipeline.rotate(rotateDeg);
      const meta = await metaPipeline.toBuffer({ resolveWithObject: true });
      quad = [0, 0, meta.info.width, 0, meta.info.width, meta.info.height, 0, meta.info.height];
    } else {
      const rect = await autoDetectCropRect(imgPath, rotateDeg);
      quad = rectToQuad(rect);
    }
  }
  const size = CELL_RENDER_SIZE * 8;
  const { buffer: data, channels } = await warpQuadToSquare(imgPath, rotateDeg, quad, size);

  const squares = [];
  for (let row = 0; row < 8; row++) {
    const rowArr = [];
    for (let col = 0; col < 8; col++) {
      const cell = new Uint8Array(SQ_SIZE * SQ_SIZE * channels);
      const x0 = col * CELL_RENDER_SIZE + CELL_MARGIN;
      const y0 = row * CELL_RENDER_SIZE + CELL_MARGIN;
      for (let y = 0; y < SQ_SIZE; y++) {
        for (let x = 0; x < SQ_SIZE; x++) {
          const srcIdx = ((y0 + y) * size + (x0 + x)) * channels;
          const dstIdx = (y * SQ_SIZE + x) * channels;
          for (let c = 0; c < channels; c++) cell[dstIdx + c] = data[srcIdx + c];
        }
      }
      rowArr.push(toGrayArray(cell, channels));
    }
    squares.push(rowArr);
  }
  return squares;
}

function normalizeGray(gray) {
  let mean = 0;
  for (const v of gray) mean += v;
  mean /= gray.length;
  let variance = 0;
  for (const v of gray) variance += (v - mean) * (v - mean);
  variance /= gray.length;
  const std = Math.sqrt(variance) || 1;
  return gray.map((v) => (v - mean) / std);
}

function sumSquaredDiff(a, b) {
  const na = normalizeGray(a), nb = normalizeGray(b);
  let sum = 0;
  for (let i = 0; i < na.length; i++) {
    const d = na[i] - nb[i];
    sum += d * d;
  }
  return sum / na.length;
}

function minDistToBank(grayArr, bank) {
  let dist = Infinity;
  for (const sample of bank) dist = Math.min(dist, sumSquaredDiff(grayArr, sample));
  return dist;
}

function edgeScore(gray) {
  const size = SQ_SIZE;
  let total = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const gx = gray[i - size + 1] + 2 * gray[i + 1] + gray[i + size + 1]
               - gray[i - size - 1] - 2 * gray[i - 1] - gray[i + size - 1];
      const gy = gray[i - size - 1] + 2 * gray[i - size] + gray[i - size + 1]
               - gray[i + size - 1] - 2 * gray[i + size] - gray[i + size + 1];
      total += Math.sqrt(gx * gx + gy * gy);
    }
  }
  return total / ((size - 2) * (size - 2));
}

function avgEdgeScore(bank) {
  if (bank.length === 0) return null;
  return bank.reduce((s, sample) => s + edgeScore(sample), 0) / bank.length;
}

function bankFor(refs, piece, squareColor) {
  const otherColor = squareColor === 'light' ? 'dark' : 'light';
  return refs[refKey(piece, squareColor)] || refs[refKey(piece, otherColor)] || [];
}

// True if this piece has no same-color-square calibration sample, meaning
// any match against it had to fall back to a cross-color comparison (always
// true for Kings/Queens, since each only ever sits on one square color in
// the starting position). Cross-color matches are noisier — the background
// itself differs — so callers should trust them less.
function usedFallback(refs, piece, squareColor) {
  return !refs[refKey(piece, squareColor)];
}

function matchSquare(grayArr, refs, squareColor, sensitivity, emptyEdgeByColor, dimFactor) {
  const emptyDist = minDistToBank(grayArr, bankFor(refs, null, squareColor));

  let bestPiece = null, bestPieceDist = Infinity, secondPieceDist = Infinity;
  for (const piece of PIECE_LABELS) {
    const bank = bankFor(refs, piece, squareColor);
    if (bank.length === 0) continue;
    const dist = minDistToBank(grayArr, bank);
    if (dist < bestPieceDist) {
      secondPieceDist = bestPieceDist;
      bestPieceDist = dist;
      bestPiece = piece;
    } else if (dist < secondPieceDist) {
      secondPieceDist = dist;
    }
  }

  const effSensitivity = sensitivity / (dimFactor || 1);

  const pixelVote = bestPieceDist * effSensitivity < emptyDist;
  const occupancyMargin = Math.abs(emptyDist - bestPieceDist * effSensitivity);
  const marginRatio = emptyDist > 0 ? occupancyMargin / emptyDist : 1;
  // If the nearest piece match required a cross-color fallback (K/Q only),
  // that comparison is inherently noisier — the background itself differs —
  // so treat the pixel vote as less trustworthy and consult the edge vote
  // even when the raw margin looks confident.
  const fellBack = bestPiece !== null && usedFallback(refs, bestPiece, squareColor);
  const CLOSE_CALL_RATIO = fellBack ? 0.6 : 0.15;

  let isPiece = pixelVote;
  let edgeDisagreed = false;
  const emptyEdge = emptyEdgeByColor && emptyEdgeByColor[squareColor];
  if (emptyEdge != null && marginRatio < CLOSE_CALL_RATIO) {
    const score = edgeScore(grayArr);
    const edgeVote = score > (emptyEdge * 1.5 + 3) / (dimFactor || 1);
    if (edgeVote !== pixelVote) {
      isPiece = edgeVote;
      edgeDisagreed = true;
    }
  }

  const label = isPiece ? bestPiece : null;

  const isUnsure = edgeDisagreed
    || marginRatio < CLOSE_CALL_RATIO
    || (isPiece && secondPieceDist - bestPieceDist < 0.2);

  const ownEdge = edgeScore(grayArr);
  return { label, isUnsure, emptyDist, bestPieceDist, bestPiece, ownEdge, emptyEdge, marginRatio };
}

function compressEmptySquares(rowStr) {
  let out = '';
  let runLength = 0;
  for (const ch of rowStr) {
    if (ch === '1') {
      runLength++;
    } else {
      if (runLength > 0) { out += runLength; runLength = 0; }
      out += ch;
    }
  }
  if (runLength > 0) out += runLength;
  return out;
}

// Ported alongside index.html's boardContrast(): whole-board raw-pixel
// std dev, a proxy for "how bright/contrasty was this photo".
function boardContrast(squares) {
  let sum = 0, sumSq = 0, n = 0;
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      for (const v of squares[row][col]) { sum += v; sumSq += v * v; n++; }
    }
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  return Math.sqrt(variance);
}

async function calibrateOne(calPath, rotateDeg, refs, contrastSamples) {
  const squares = await sliceImageToSquares(calPath, rotateDeg);
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = STARTING_FEN_ROWS[row][col];
      const label = piece === '-' ? null : piece;
      const color = squareColorAt(row, col);
      const key = refKey(label, color);
      if (!refs[key]) refs[key] = [];
      refs[key].push(squares[row][col]);
    }
  }
  if (contrastSamples) contrastSamples.push(boardContrast(squares));
}

// specs: array of "path" or "path@rotateDeg" (per-image rotation, since
// different photos may have been taken holding the phone differently)
async function calibrate(specs) {
  const refs = {};
  const contrastSamples = [];
  for (const spec of specs) {
    const [p, rot] = spec.split('@');
    await calibrateOne(p.trim(), rot ? parseFloat(rot) : 0, refs, contrastSamples);
  }
  const calibContrast = contrastSamples.reduce((a, b) => a + b, 0) / contrastSamples.length;
  return { refs, calibContrast };
}

async function analyze(testPath, refs, sensitivity, rotateDeg, calibContrast) {
  const squares = await sliceImageToSquares(testPath, rotateDeg);
  const emptyEdgeByColor = {
    light: avgEdgeScore(bankFor(refs, null, 'light')),
    dark: avgEdgeScore(bankFor(refs, null, 'dark'))
  };

  let dimFactor = 1;
  if (calibContrast > 0) {
    const analysisContrast = boardContrast(squares);
    const ratio = analysisContrast / calibContrast;
    dimFactor = Math.max(0.5, Math.min(1, ratio));
  }

  const rows = [];
  const debug = [];
  for (let row = 0; row < 8; row++) {
    let rowStr = '';
    for (let col = 0; col < 8; col++) {
      const color = squareColorAt(row, col);
      const result = matchSquare(squares[row][col], refs, color, sensitivity, emptyEdgeByColor, dimFactor);
      rowStr += result.label === null ? '1' : result.label;
      debug.push({ row, col, color, ...result });
    }
    rows.push(compressEmptySquares(rowStr));
  }
  const fen = rows.join('/') + ' w - - 0 1';
  return { fen, debug, dimFactor };
}

async function main() {
  const [,, calPathArg, testPath, expectedFen, sensitivityArg, rotateArg] = process.argv;
  if (!calPathArg || !testPath) {
    console.error('Usage: node test_match.js <calibration-image[,image2,...]|image@rotateDeg,...> <test-image> [expected-fen] [sensitivity] [testRotateDeg]');
    process.exit(1);
  }
  const sensitivity = sensitivityArg ? parseFloat(sensitivityArg) : 1.0;
  const rotateDeg = rotateArg ? parseFloat(rotateArg) : 0; // test image's rotation

  const calSpecs = calPathArg.split(',');
  const { refs, calibContrast } = await calibrate(calSpecs);
  const refCounts = Object.keys(refs).length;
  console.log(`Calibrated: ${refCounts} distinct refs from ${calSpecs.length} image(s)`);

  const { fen, debug, dimFactor } = await analyze(testPath, refs, sensitivity, rotateDeg, calibContrast);
  console.log(`Sensitivity: ${sensitivity}`);
  console.log(`dimFactor: ${dimFactor.toFixed(3)} (calibContrast=${calibContrast.toFixed(2)})`);
  console.log(`Result FEN placement: ${fen}`);
  if (expectedFen) {
    const expectedPlacement = expectedFen.split(' ')[0];
    const resultPlacement = fen.split(' ')[0];
    console.log(`Expected:              ${expectedPlacement}`);
    console.log(resultPlacement === expectedPlacement ? '*** MATCH ***' : '*** MISMATCH ***');
  }

  // Print occupied squares with confidence info for debugging.
  console.log('\nOccupied squares (row0=rank8...row7=rank1, col0=a...col7=h):');
  for (const d of debug) {
    if (d.label !== null) {
      console.log(`  row${d.row} col${d.col} (${d.color}) -> ${d.label}  emptyDist=${d.emptyDist.toFixed(3)} bestPieceDist=${d.bestPieceDist.toFixed(3)}${d.isUnsure ? '  [UNSURE]' : ''}`);
    }
  }

  if (process.env.DEBUG_CELL) {
    const [r, c] = process.env.DEBUG_CELL.split(',').map(Number);
    const d = debug.find((x) => x.row === r && x.col === c);
    console.log(`\nDEBUG row${r} col${c}:`, d);

    // Full ranking of every piece candidate's distance for this square.
    const squares = await sliceImageToSquares(testPath, rotateDeg);
    const grayArr = squares[r][c];
    const color = squareColorAt(r, c);
    const ranked = PIECE_LABELS.map((piece) => {
      const bank = bankFor(refs, piece, color);
      return { piece, dist: bank.length ? minDistToBank(grayArr, bank) : null, fallback: usedFallback(refs, piece, color) };
    }).sort((a, b) => (a.dist ?? Infinity) - (b.dist ?? Infinity));
    console.log('Ranked piece candidates:', ranked.map((r) => `${r.piece}${r.fallback ? '*' : ''}=${r.dist?.toFixed(3)}`).join('  '));
  }
}

if (require.main === module) {
  main().catch((e) => { console.error(e); process.exit(1); });
} else {
  module.exports = { warpQuadToSquare, autoDetectCropRect, sliceImageToSquares, matchSquare, bankFor, avgEdgeScore, squareColorAt, refKey, STARTING_FEN_ROWS, compressEmptySquares };
}
