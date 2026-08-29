require('dotenv').config({ quiet: true });
const express = require('express');
const multer = require('multer');
const Anthropic = require('@anthropic-ai/sdk');
const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Windows path to Nibbler, overridable via .env. Converted to the /mnt/c/...
// form so it can be launched from WSL (only works when this server is
// actually running inside WSL on the same machine as Nibbler — not usable
// if this app is ever deployed elsewhere).
const NIBBLER_WIN_PATH = process.env.NIBBLER_PATH ||
  'C:\\Users\\natha\\OneDrive\\Backup\\Chess\\nibbler-2.4.1-windows\\nibbler-2.4.1-windows\\nibbler.exe';

function winPathToWsl(winPath) {
  return winPath.replace(/\\/g, '/').replace(/^([A-Za-z]):/, (_, drive) => '/mnt/' + drive.toLowerCase());
}

// Inverse of winPathToWsl — only valid for paths actually under /mnt/<drive>/...
function wslPathToWin(wslPath) {
  const m = wslPath.match(/^\/mnt\/([a-z])\/(.*)$/);
  if (!m) throw new Error(`Cannot convert to a Windows path (not under /mnt/<drive>): ${wslPath}`);
  return `${m[1].toUpperCase()}:\\${m[2].replace(/\//g, '\\')}`;
}

const FEN_RE = /^[pnbrqkPNBRQK1-8]+(\/[pnbrqkPNBRQK1-8]+){7} [wb] (-|[KQkq]{1,4}) (-|[a-h][36]) \d+ \d+$/;

// Nibbler's command-line handling always loads the given path as a *PGN*
// file, and silently does nothing if the path doesn't exist — passing a raw
// FEN string as the argument does not work. But Nibbler's PGN parser does
// honor a [FEN "..."] tag for the starting position, so we write a minimal
// one-game PGN carrying the FEN and hand Nibbler that file's path instead.
// The file must live somewhere the Windows exe can actually read, so it goes
// under this project folder (which is on the real NTFS filesystem via
// /mnt/c/...), not into WSL-only storage like /tmp.
const NIBBLER_TMP_DIR = path.join(__dirname, 'tmp');
const NIBBLER_TMP_PGN = path.join(NIBBLER_TMP_DIR, 'nibbler-fen.pgn');

function fenToMinimalPgn(fen) {
  return [
    '[Event "Chess Photo Analyzer"]',
    '[Site "?"]',
    '[Date "????.??.??"]',
    '[Round "?"]',
    '[White "?"]',
    '[Black "?"]',
    '[Result "*"]',
    `[FEN "${fen}"]`,
    '[SetUp "1"]',
    '',
    '*',
    ''
  ].join('\n');
}

app.use(express.static('public'));
app.use(express.json());

app.get('/regression', (req, res) => {
  res.sendFile(path.join(__dirname, 'Regression', 'regression.html'));
});

// Shared parsing/regression logic lives under Regression/ so the same files
// work both as Node modules (run_regression.js / backtest_models.js CLIs)
// and as plain browser scripts for the /regression page.
app.get('/js/data_parser.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'Regression', 'data_parser.js'));
});
app.get('/js/regression.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'Regression', 'regression.js'));
});
app.get('/js/model_candidates.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'Regression', 'model_candidates.js'));
});

app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

    const base64Image = req.file.buffer.toString('base64');
    const mediaType = req.file.mimetype;

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-5',
      max_tokens: 300,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Image } },
            {
              type: 'text',
              text:
                "Look at this chess board image. Determine the exact position of every piece. " +
                "Reply with ONLY valid JSON in this exact shape, no other text: " +
                '{"fen": "<FEN string for the position>", "sideToMove": "w" or "b", "confidence": "high" or "low", "notes": "<any uncertainty about squares you had trouble reading, or empty string>"}. ' +
                "If you cannot tell whose turn it is, assume white to move. Use standard FEN notation including castling rights as \"-\" if unknown."
            }
          ]
        }
      ]
    });

    const textBlock = message.content.find((b) => b.type === 'text');
    let parsed;
    try {
      parsed = JSON.parse(textBlock.text);
    } catch (e) {
      return res.status(502).json({ error: 'Could not parse AI response', raw: textBlock.text });
    }

    res.json(parsed);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Server error' });
  }
});

app.post('/api/open-nibbler', (req, res) => {
  const fen = (req.body && req.body.fen || '').trim();
  if (!FEN_RE.test(fen)) {
    return res.status(400).json({ error: 'That does not look like a valid FEN.' });
  }

  const wslNibblerPath = winPathToWsl(NIBBLER_WIN_PATH);
  if (!fs.existsSync(wslNibblerPath)) {
    return res.status(500).json({ error: `Nibbler not found at ${NIBBLER_WIN_PATH}. Set NIBBLER_PATH in .env to fix.` });
  }

  let winPgnPath;
  try {
    fs.mkdirSync(NIBBLER_TMP_DIR, { recursive: true });
    fs.writeFileSync(NIBBLER_TMP_PGN, fenToMinimalPgn(fen));
    winPgnPath = wslPathToWin(NIBBLER_TMP_PGN);
  } catch (err) {
    console.error('Failed to write temp PGN for Nibbler:', err);
    return res.status(500).json({ error: 'Could not write the temporary position file.' });
  }

  const child = execFile(wslNibblerPath, [winPgnPath], (err) => {
    // execFile's callback fires on process exit; Nibbler is a GUI app that
    // stays open, so a non-zero/late exit here isn't necessarily an error —
    // only report failure if it never even started.
    if (err && err.code === 'ENOENT') {
      console.error('Failed to launch Nibbler:', err);
    }
  });
  child.on('error', (err) => console.error('Failed to launch Nibbler:', err));

  res.json({ ok: true });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`ChessAnalyzer running at http://localhost:${PORT}`));
