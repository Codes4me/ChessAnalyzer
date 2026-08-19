require('dotenv').config({ quiet: true });
const express = require('express');
const multer = require('multer');
const Anthropic = require('@anthropic-ai/sdk');
const { execFile } = require('child_process');
const fs = require('fs');

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

const FEN_RE = /^[pnbrqkPNBRQK1-8]+(\/[pnbrqkPNBRQK1-8]+){7} [wb] (-|[KQkq]{1,4}) (-|[a-h][36]) \d+ \d+$/;

app.use(express.static('public'));
app.use(express.json());

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

  const wslPath = winPathToWsl(NIBBLER_WIN_PATH);
  if (!fs.existsSync(wslPath)) {
    return res.status(500).json({ error: `Nibbler not found at ${NIBBLER_WIN_PATH}. Set NIBBLER_PATH in .env to fix.` });
  }

  // Nibbler takes a FEN (or PGN file path) as its command-line argument.
  const child = execFile(wslPath, [fen], (err) => {
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
