// Estimates the JPEG quality setting a file was saved with, by reading its
// own embedded luminance quantization table (DQT marker) and comparing it
// against the standard IJG base table -- the same technique tools like
// jpeginfo/exiftool's "-Quality" use. Returns null for anything that isn't
// a standard baseline JPEG (WebP, PNG, a JPEG with no DQT we can parse,
// etc.) so callers can fall back to a default instead of guessing.
const fs = require('fs');

// Parses all DQT (0xFFDB) segments in a JPEG buffer, returning a map of
// table-id -> Uint8Array(64) (8-bit precision tables only; 16-bit-precision
// tables are rare in real-world camera/export JPEGs and are skipped).
function parseJpegQuantTables(buffer) {
  if (buffer.length < 4 || buffer[0] !== 0xFF || buffer[1] !== 0xD8) return null; // not a JPEG
  const tables = {};
  let i = 2;
  while (i + 4 <= buffer.length) {
    if (buffer[i] !== 0xFF) { i++; continue; }
    const marker = buffer[i + 1];
    if (marker === 0xD8 || marker === 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { i += 2; continue; } // no-length markers
    if (marker === 0xD9) break; // EOI
    if (i + 4 > buffer.length) break;
    const segLen = buffer.readUInt16BE(i + 2);
    if (marker === 0xDB) { // DQT
      let p = i + 4, end = i + 2 + segLen;
      while (p < end) {
        const pqTq = buffer[p]; p++;
        const precision = pqTq >> 4, id = pqTq & 0x0F;
        const count = precision === 0 ? 64 : 128;
        if (p + count > buffer.length) break;
        if (precision === 0) tables[id] = Uint8Array.from(buffer.slice(p, p + 64));
        p += count;
      }
    }
    if (marker === 0xDA) break; // start of scan -- no more headers after this
    i += 2 + segLen;
  }
  return tables;
}

// Standard IJG baseline luminance quantization table (quality=50), zigzag
// order doesn't matter here -- we only need table[0] (the DC coefficient).
const BASE_LUMA_DC = 16;

function estimateJpegQuality(bufferOrPath) {
  const buffer = Buffer.isBuffer(bufferOrPath) ? bufferOrPath : fs.readFileSync(bufferOrPath);
  const tables = parseJpegQuantTables(buffer);
  if (!tables || tables[0] == null) return null;
  const dc = tables[0][0];
  if (!dc) return null;
  // Invert the IJG scale-factor formula (jcparam.c jpeg_quality_scaling):
  //   quality<50:  scale = 5000/quality
  //   quality>=50: scale = 200 - 2*quality
  //   table_entry = round(base_entry * scale / 100)
  const scale = (dc * 100) / BASE_LUMA_DC;
  let quality = scale >= 100 ? 5000 / scale : (200 - scale) / 2;
  quality = Math.round(Math.max(1, Math.min(100, quality)));
  return quality;
}

module.exports = { estimateJpegQuality, parseJpegQuantTables };
