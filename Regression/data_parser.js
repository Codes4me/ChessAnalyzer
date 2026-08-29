// Turns whatever format someone pasted/saved into clean { headers, columns }
// data ready for the regression program.
//
// Handles three input shapes:
//   1. Desmos-style lists, e.g.  \left[1,2,3\right]  (one per line, first = X, second = Y, ...)
//   2. A table with a header row: "time,value" then "1,10.2" etc. (comma, tab, or space separated)
//   3. A table with no header row (falls back to generic names x1, x2, ...)

// Desmos range shorthand, e.g. "16...75" meaning the whole sequence
// 16,17,18,...,75 (counts down if the second number is smaller). Some
// people write it as "16,...75" (comma right before the dots) — treat that
// the same as "16...75" before matching.
function expandRanges(listContent) {
  const normalized = listContent.replace(/,\s*\.\.\./g, '...');
  return normalized.replace(/(-?\d+)\s*\.\.\.\s*(-?\d+)/g, (_, a, b) => {
    const start = parseInt(a, 10);
    const end = parseInt(b, 10);
    const step = start <= end ? 1 : -1;
    const seq = [];
    for (let v = start; step > 0 ? v <= end : v >= end; v += step) seq.push(v);
    return seq.join(',');
  });
}

// Turns \frac{A}{B} into ((A)/(B)) so it can go through the plain-arithmetic
// evaluator below. Handles nested fractions by repeating until none are left.
function convertFractions(s) {
  let prev;
  do {
    prev = s;
    s = s.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '(($1)/($2))');
  } while (s !== prev);
  return s;
}

// Desmos LaTeX carries spacing commands (\ , \, , \; , \quad) that people's
// copy/pasted sums sometimes keep, e.g. "271.3\ +13.2". Strip them — they're
// just whitespace, not part of the number.
function stripLatexSpacing(s) {
  return s.replace(/\\(quad|qquad|[ ,;])/g, ' ');
}

// "\left(" / "\right)" are just LaTeX's auto-sized parens — same meaning as
// plain "(" / ")", but our evaluator only recognizes the plain form.
function stripLatexParens(s) {
  return s.replace(/\\left\(/g, '(').replace(/\\right\)/g, ')');
}

// Desmos allows implicit multiplication like ".5(3.25+3.2)" (no "*"
// needed) — JS doesn't, so insert the "*" wherever a number or ")" is
// immediately followed by "(".
function insertImplicitMultiplication(s) {
  return s.replace(/([\d)])\s*\(/g, '$1*(');
}

// Runs a token through every LaTeX-to-arithmetic cleanup step, then
// evaluates it as a plain expression. Only digits/+-*/()./spaces are ever
// allowed through to Function(), so nothing else — no matter what odd LaTeX
// survives cleanup — can execute. Returns NaN if it isn't a valid expression.
function evaluateScalarExpression(raw) {
  let s = stripLatexParens(raw);
  s = convertFractions(s);
  s = insertImplicitMultiplication(s);
  s = stripLatexSpacing(s).trim();
  if (!s) return NaN;
  if (/^[-+*/().\d\s]+$/.test(s)) {
    try {
      const val = Function(`"use strict"; return (${s});`)();
      if (Number.isFinite(val)) return val;
    } catch (e) { /* not a valid expression, fall through */ }
  }
  return NaN;
}

// A list entry can be a plain number, or a small arithmetic expression
// Desmos evaluates on the fly (e.g. "100-72", a fraction, or "100-72\ ").
function evaluateNumberToken(raw) {
  const trimmed = raw.trim();
  if (!trimmed) return NaN;
  if (/^-?[\d.]+(e-?\d+)?$/i.test(trimmed)) return parseFloat(trimmed);
  // Evaluate the *untrimmed* string — trimming first could eat a trailing
  // LaTeX-spacing backslash's space and leave a bare, unmatched backslash.
  return evaluateScalarExpression(raw);
}

function normalizeLabel(label) {
  return label ? label.replace(/[{}]/g, '').toLowerCase() : null;
}

// A list can reference another list's length instead of a literal count,
// e.g. "1...\operatorname{count}\left(y_1\right)" (Desmos's way of writing
// "1 through however many Y values there are"). `labels` gives each list's
// assigned variable name where one was written (e.g. "y_1=\left[...\right]"),
// parallel to rawContents — used to match the count() reference to the
// *specific* list it names, since a block can hold more than 2 lists (an
// unrelated extra list shouldn't be mistaken for the referenced one). Falls
// back to the first resolvable sibling list only when no name matches.
function resolveCountReferences(rawContents, labels) {
  const hasCountRef = c => /\\operatorname\{count\}|count\s*\(/i.test(c);
  const countRefName = c => {
    const m = c.match(/\\operatorname\{count\}\\left\(([\s\S]*?)\\right\)/i) || c.match(/count\s*\(([^)]*)\)/i);
    return m ? normalizeLabel(m[1].trim()) : null;
  };

  const lengths = rawContents.map(content => {
    if (hasCountRef(content)) return null;
    const expanded = expandRanges(content);
    const nums = expanded.split(',').map(evaluateNumberToken).filter(n => Number.isFinite(n));
    return nums.length || null;
  });

  const fallbackLength = lengths.find(l => l !== null);

  return rawContents.map((content, i) => {
    if (!hasCountRef(content)) return content;

    const wantedName = countRefName(content);
    let resolvedLength = fallbackLength;
    if (wantedName) {
      const namedIndex = labels.findIndex((label, j) => normalizeLabel(label) === wantedName && lengths[j] !== null);
      if (namedIndex !== -1) resolvedLength = lengths[namedIndex];
    }
    if (resolvedLength === undefined) return content;

    return content
      .replace(/\\operatorname\{count\}\\left\([\s\S]*?\\right\)/gi, String(resolvedLength))
      .replace(/count\s*\([^)]*\)/gi, String(resolvedLength));
  });
}

// A whole list can have an operation applied to it right after the closing
// bracket, e.g. "\left[...\right]-x_1" (subtract another list, element by
// element) or "\left[...\right]\cdot \frac{1}{3.3}" (scale every value by a
// number). `suffix` is whatever text follows the bracket on the same line.
function applyListSuffix(nums, suffix, i, lists, labels) {
  const s = (suffix || '').trim();
  if (!s) return nums;

  // Element-wise +/- another named list, e.g. "-x_1" or "+ y_2".
  let m = s.match(/^([+-])\s*([a-zA-Z]_?\{?\d*\}?)\s*$/);
  if (m) {
    const wantedName = normalizeLabel(m[2]);
    let otherIdx = labels.findIndex((label, j) => j !== i && normalizeLabel(label) === wantedName);
    // Fall back to the positional x/y convention (first list = x, second =
    // y) when the referenced list was never explicitly labeled.
    if (otherIdx === -1) {
      if (wantedName.startsWith('x') && i !== 0) otherIdx = 0;
      else if (wantedName.startsWith('y') && i !== 1 && lists.length > 1) otherIdx = 1;
    }
    if (otherIdx !== -1 && lists[otherIdx]) {
      const other = lists[otherIdx];
      const sign = m[1] === '-' ? -1 : 1;
      const len = Math.min(nums.length, other.length);
      return nums.slice(0, len).map((v, k) => v + sign * other[k]);
    }
  }

  // Scalar multiply/divide, e.g. "\cdot \frac{1}{3.3}", "* 2", "/ 4".
  m = s.match(/^(?:\\cdot|\*)\s*(.+)$/) || s.match(/^\/\s*(.+)$/);
  if (m) {
    const scalar = evaluateScalarExpression(m[1]);
    if (Number.isFinite(scalar)) {
      const isDivide = s.startsWith('/');
      return nums.map(v => (isDivide ? v / scalar : v * scalar));
    }
  }

  // Scalar add/subtract a plain number, e.g. "+ 5", "- 2.3".
  m = s.match(/^([+-])\s*(.+)$/);
  if (m) {
    const scalar = evaluateScalarExpression(m[2]);
    if (Number.isFinite(scalar)) {
      const sign = m[1] === '-' ? -1 : 1;
      return nums.map(v => v + sign * scalar);
    }
  }

  return nums; // unrecognized suffix — leave the list as-is
}

function parseDesmosLists(text) {
  // Capture an optional "name=" right before each list, e.g. "y_1=\left[...\right]"
  // or "x_{1}=\left[...\right]" — used to resolve count(y_1)-style references below.
  const re = /(?:([a-zA-Z]_?\{?\d*\}?)\s*=\s*)?\\left\[([^\]]*)\\right\]/g;
  const rawContents = [];
  const labels = [];
  const suffixes = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    labels.push(m[1] || null);
    rawContents.push(m[2]);
    suffixes.push(text.slice(re.lastIndex).split('\n')[0]);
  }

  const resolved = resolveCountReferences(rawContents, labels);

  let lists = resolved.map(content => {
    const expanded = expandRanges(content);
    return expanded.split(',').map(evaluateNumberToken).filter(n => Number.isFinite(n));
  });

  lists = lists.map((nums, i) => applyListSuffix(nums, suffixes[i], i, lists, labels));

  // Keep only non-empty lists, dropping the matching label alongside each
  // dropped list so the two arrays stay lined up by index.
  const keptLists = [];
  const keptLabels = [];
  lists.forEach((nums, i) => {
    if (nums.length) { keptLists.push(nums); keptLabels.push(labels[i]); }
  });

  return { lists: keptLists, labels: keptLabels };
}

function splitLine(line) {
  const delim = line.includes(',') ? ',' : line.includes('\t') ? '\t' : /\s+/;
  return line.split(delim).map(v => v.trim()).filter(v => v.length);
}

function looksNumeric(v) {
  return v.length > 0 && Number.isFinite(parseFloat(v)) && /^-?[\d.]+(e-?\d+)?$/i.test(v);
}

function parseTable(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim().length);
  if (!lines.length) return null;

  const firstRow = splitLine(lines[0]);
  const firstRowIsHeader = firstRow.some(v => !looksNumeric(v));

  let headers, dataLines;
  if (firstRowIsHeader) {
    headers = firstRow;
    dataLines = lines.slice(1);
  } else {
    headers = firstRow.map((_, i) => `x${i + 1}`);
    dataLines = lines;
  }

  const rows = dataLines.map(splitLine).filter(r => r.length === headers.length);
  if (!rows.length) return null;

  const columns = headers.map((h, i) => rows.map(r => parseFloat(r[i])));
  const allNumeric = columns.every(col => col.every(v => Number.isFinite(v)));
  if (!allNumeric) return null;

  return { headers, columns };
}

// Some saved files carry a leading row-number column (e.g. "1\t<list>"),
// left over from copy/pasting a Desmos list. Strip it — but only on lines
// that actually contain a Desmos list, so a plain CSV row like "1,-23112"
// (a legitimate integer X value) is never mistaken for an index prefix.
function stripLeadingIndexColumn(text) {
  return text
    .split(/\r?\n/)
    .map(line => (line.includes('\\left[') ? line.replace(/^\d+\s*[\t,]\s*/, '') : line))
    .join('\n');
}

// A block can start with a plain title/label line, with no blank line
// separating it from the data below (e.g. "Experiment 1" then "time,value").
// Detect that by comparing the title candidate's field count against the
// next real data line's field count, using whatever delimiter that data
// line uses — a genuine header/data row will match; a title won't.
function extractTitle(block) {
  const lines = block.split(/\r?\n/);
  if (lines.length < 2) return { title: null, body: block };
  const first = lines[0].trim();
  if (!first || first.includes('\\left[')) return { title: null, body: block };

  const nextLine = lines.slice(1).find(l => l.trim().length);
  if (!nextLine) return { title: null, body: block };

  if (nextLine.includes('\\left[')) {
    return { title: first, body: lines.slice(1).join('\n') };
  }

  const delim = nextLine.includes(',') ? ',' : nextLine.includes('\t') ? '\t' : /\s+/;
  const firstFields = splitLine(first).length ? first.split(delim).map(v => v.trim()).filter(v => v.length) : [];
  const nextFields = nextLine.split(delim).map(v => v.trim()).filter(v => v.length);

  if (firstFields.length !== nextFields.length) {
    return { title: first, body: lines.slice(1).join('\n') };
  }
  return { title: null, body: block };
}

function parseInput(rawBlock) {
  const { title, body } = extractTitle(rawBlock);
  const text = stripLeadingIndexColumn(body);

  const { lists: desmosLists, labels: desmosLabels } = parseDesmosLists(text);
  if (desmosLists.length >= 2) {
    // Prefer the actually-labeled x_N/y_N pair over plain position, so an
    // unrelated extra list in the same block (a stray middle list, say)
    // can't get mistaken for X or Y — and can't shrink them by truncation.
    const xIdx = desmosLabels.findIndex(l => normalizeLabel(l) && normalizeLabel(l).startsWith('x'));
    const yIdx = desmosLabels.findIndex(l => normalizeLabel(l) && normalizeLabel(l).startsWith('y'));
    const primaryX = xIdx !== -1 ? xIdx : 0;
    const primaryY = yIdx !== -1 && yIdx !== primaryX ? yIdx : (primaryX === 1 ? 0 : 1);

    const order = [primaryX, primaryY, ...desmosLists.map((_, i) => i).filter(i => i !== primaryX && i !== primaryY)];
    const orderedLists = order.map(i => desmosLists[i]);

    // The X/Y pair drives the row count; any other list in the block just
    // rides along truncated to match, since it isn't used in the fit.
    const pairLen = Math.min(orderedLists[0].length, orderedLists[1].length);
    const columns = orderedLists.map(l => l.slice(0, pairLen));
    const headers = columns.map((_, i) => (i === 0 ? 'x' : i === 1 ? 'y' : `list${i + 1}`));
    return { headers, columns, source: 'desmos-lists', title };
  }

  const table = parseTable(text);
  if (table) return { ...table, source: 'table', title };

  throw new Error('Could not recognize the data format. Expected a CSV/table or Desmos-style \\left[...\\right] lists.');
}

// A file can hold several unrelated groups of lists, separated by one or
// more blank lines. Each group is parsed independently (its own X/Y/etc.
// lists), so unrelated datasets never get mixed together.
function parseGroups(rawText) {
  const blocks = rawText
    .split(/\r?\n\s*\r?\n/)
    .map(b => b.trim())
    .filter(b => b.length);

  if (!blocks.length) throw new Error('No data found.');

  const groups = [];
  const errors = [];
  blocks.forEach((block, i) => {
    try {
      groups.push(parseInput(block));
    } catch (e) {
      errors.push(`Group ${i + 1}: ${e.message}`);
    }
  });

  if (!groups.length) throw new Error(errors.join('\n'));
  return { groups, errors };
}

const DataParser = { parseInput, parseGroups };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DataParser;
} else {
  window.DataParser = DataParser;
}
