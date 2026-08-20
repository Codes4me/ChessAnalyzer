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

async function sliceImageToSquares(imgPath, rotateDeg) {
  const size = CELL_RENDER_SIZE * 8;
  let pipeline = sharp(imgPath);
  if (rotateDeg) pipeline = pipeline.rotate(rotateDeg);
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

async function calibrate(calPath, rotateDeg) {
  const squares = await sliceImageToSquares(calPath, rotateDeg);
  const refs = {};
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
  return refs;
}

async function analyze(testPath, refs, sensitivity, rotateDeg) {
  const squares = await sliceImageToSquares(testPath, rotateDeg);
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
  const [,, calPath, testPath, expectedFen, sensitivityArg, rotateArg] = process.argv;
  if (!calPath || !testPath) {
    console.error('Usage: node test_match.js <calibration-image> <test-image> [expected-fen] [sensitivity] [rotateDeg]');
    process.exit(1);
  }
  const sensitivity = sensitivityArg ? parseFloat(sensitivityArg) : 1.0;
  const rotateDeg = rotateArg ? parseFloat(rotateArg) : 0;

  const refs = await calibrate(calPath, rotateDeg);
  const refCounts = Object.keys(refs).length;
  console.log(`Calibrated: ${refCounts} distinct refs`);

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

main().catch((e) => { console.error(e); process.exit(1); });
