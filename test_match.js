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

async function autoDetectCropRect(imgPath, rotateDeg) {
  const WORK_SIZE = 300;
  let pipeline = sharp(imgPath);
  if (rotateDeg) pipeline = pipeline.rotate(rotateDeg);
  const { data } = await pipeline.resize(WORK_SIZE, WORK_SIZE, { fit: 'fill' }).grayscale().raw().toBuffer({ resolveWithObject: true });
  const w = WORK_SIZE, h = WORK_SIZE;

  const edge = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = data[i - w + 1] + 2 * data[i + 1] + data[i + w + 1]
               - data[i - w - 1] - 2 * data[i - 1] - data[i + w - 1];
      const gy = data[i - w - 1] + 2 * data[i - w] + data[i - w + 1]
               - data[i + w - 1] - 2 * data[i + w] - data[i + w + 1];
      edge[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  const rowSum = new Float64Array(h), colSum = new Float64Array(w);
  for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) s += edge[y * w + x]; rowSum[y] = s; }
  for (let x = 0; x < w; x++) { let s = 0; for (let y = 0; y < h; y++) s += edge[y * w + x]; colSum[x] = s; }

  function smooth(arr, radius) {
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let s = 0, n = 0;
      for (let k = -radius; k <= radius; k++) { const j = i + k; if (j >= 0 && j < arr.length) { s += arr[j]; n++; } }
      out[i] = s / n;
    }
    return out;
  }
  function findBounds(profile) {
    let max = 0;
    for (const v of profile) if (v > max) max = v;
    const threshold = max * 0.35;
    let lo = 0, hi = profile.length - 1;
    while (lo < profile.length && profile[lo] < threshold) lo++;
    while (hi >= 0 && profile[hi] < threshold) hi--;
    if (hi <= lo) return [0, profile.length - 1];
    return [lo, hi];
  }
  const [y0, y1] = findBounds(smooth(rowSum, 4));
  const [x0, x1] = findBounds(smooth(colSum, 4));

  // metadata of the (rotated) working pipeline's output dimensions == natural
  // rotated size; get via a fresh call since sharp doesn't expose it post-hoc easily
  let metaPipeline = sharp(imgPath);
  if (rotateDeg) metaPipeline = metaPipeline.rotate(rotateDeg);
  const meta = await metaPipeline.toBuffer({ resolveWithObject: true });
  const naturalW = meta.info.width, naturalH = meta.info.height;
  const sx = naturalW / w, sy = naturalH / h;
  return { x: Math.round(x0 * sx), y: Math.round(y0 * sy), w: Math.round((x1 - x0) * sx), h: Math.round((y1 - y0) * sy) };
}

async function sliceImageToSquares(imgPath, rotateDeg, cropRect) {
  const size = CELL_RENDER_SIZE * 8;
  let pipeline = sharp(imgPath);
  if (rotateDeg) pipeline = pipeline.rotate(rotateDeg);
  if (cropRect) {
    pipeline = pipeline.extract({
      left: Math.max(0, cropRect.x), top: Math.max(0, cropRect.y),
      width: cropRect.w, height: cropRect.h
    });
  }
  const { data, info } = await pipeline
    .resize(size, size, { fit: 'fill' })
    .raw()
    .toBuffer({ resolveWithObject: true });
  const channels = info.channels;

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

function matchSquare(grayArr, refs, squareColor, sensitivity, emptyEdgeByColor) {
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

  const pixelVote = bestPieceDist * sensitivity < emptyDist;
  const occupancyMargin = Math.abs(emptyDist - bestPieceDist * sensitivity);
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
    const edgeVote = score > emptyEdge * 1.5 + 3;
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

async function calibrateOne(calPath, rotateDeg, refs) {
  const cropRect = await autoDetectCropRect(calPath, rotateDeg);
  const squares = await sliceImageToSquares(calPath, rotateDeg, cropRect);
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
}

// specs: array of "path" or "path@rotateDeg" (per-image rotation, since
// different photos may have been taken holding the phone differently)
async function calibrate(specs) {
  const refs = {};
  for (const spec of specs) {
    const [p, rot] = spec.split('@');
    await calibrateOne(p.trim(), rot ? parseFloat(rot) : 0, refs);
  }
  return refs;
}

async function analyze(testPath, refs, sensitivity, rotateDeg) {
  const cropRect = await autoDetectCropRect(testPath, rotateDeg);
  const squares = await sliceImageToSquares(testPath, rotateDeg, cropRect);
  const emptyEdgeByColor = {
    light: avgEdgeScore(bankFor(refs, null, 'light')),
    dark: avgEdgeScore(bankFor(refs, null, 'dark'))
  };

  const rows = [];
  const debug = [];
  for (let row = 0; row < 8; row++) {
    let rowStr = '';
    for (let col = 0; col < 8; col++) {
      const color = squareColorAt(row, col);
      const result = matchSquare(squares[row][col], refs, color, sensitivity, emptyEdgeByColor);
      rowStr += result.label === null ? '1' : result.label;
      debug.push({ row, col, color, ...result });
    }
    rows.push(compressEmptySquares(rowStr));
  }
  const fen = rows.join('/') + ' w - - 0 1';
  return { fen, debug };
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
  const refs = await calibrate(calSpecs);
  const refCounts = Object.keys(refs).length;
  console.log(`Calibrated: ${refCounts} distinct refs from ${calSpecs.length} image(s)`);

  const { fen, debug } = await analyze(testPath, refs, sensitivity, rotateDeg);
  console.log(`Sensitivity: ${sensitivity}`);
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
    const cropRect = await autoDetectCropRect(testPath, rotateDeg);
    const squares = await sliceImageToSquares(testPath, rotateDeg, cropRect);
    const grayArr = squares[r][c];
    const color = squareColorAt(r, c);
    const ranked = PIECE_LABELS.map((piece) => {
      const bank = bankFor(refs, piece, color);
      return { piece, dist: bank.length ? minDistToBank(grayArr, bank) : null, fallback: usedFallback(refs, piece, color) };
    }).sort((a, b) => (a.dist ?? Infinity) - (b.dist ?? Infinity));
    console.log('Ranked piece candidates:', ranked.map((r) => `${r.piece}${r.fallback ? '*' : ''}=${r.dist?.toFixed(3)}`).join('  '));
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
