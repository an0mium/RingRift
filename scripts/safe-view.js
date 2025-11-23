#!/usr/bin/env node

/**
 * safe-view.js
 *
 * Purpose:
 *  - Safely view large or "wide" files (e.g., logs, Jest output) without
 *    overwhelming VSCode or the Cline extension.
 *  - Guarantees that:
 *      * No output line exceeds a configured maximum length.
 *      * The total number of output lines is capped.
 *      * If truncation occurs, a clear marker is appended.
 *
 * Usage:
 *  node scripts/safe-view.js <inputFile> [outputFile] [--max-line-length=N] [--max-lines=N]
 *
 * Examples:
 *  # Create a wrapped, size-limited view of a log file
 *  node scripts/safe-view.js logs/ai/sandbox-ai-sim.log logs/ai/sandbox-ai-sim.view.txt
 *
 *  # Custom limits
 *  node scripts/safe-view.js logs/big.log logs/big.view.txt --max-line-length=200 --max-lines=800
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

function parseArgs(argv) {
  const args = argv.slice(2);
  if (args.length < 1) {
    console.error('Usage: node scripts/safe-view.js <inputFile> [outputFile] [--max-line-length=N] [--max-lines=N]');
    process.exit(1);
  }

  const positional = [];
  const options = {};

  for (const arg of args) {
    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      options[key] = value !== undefined ? value : true;
    } else {
      positional.push(arg);
    }
  }

  const inputFile = positional[0];
  const outputFile = positional[1] || null; // null => stdout

  const maxLineLength = options['max-line-length'] ? parseInt(options['max-line-length'], 10) : 240;
  const maxLines = options['max-lines'] ? parseInt(options['max-lines'], 10) : 800;

  if (!Number.isFinite(maxLineLength) || maxLineLength <= 0) {
    throw new Error(`Invalid --max-line-length: ${options['max-line-length']}`);
  }
  if (!Number.isFinite(maxLines) || maxLines <= 0) {
    throw new Error(`Invalid --max-lines: ${options['max-lines']}`);
  }

  return { inputFile, outputFile, maxLineLength, maxLines };
}

async function safeView({ inputFile, outputFile, maxLineLength, maxLines }) {
  const inPath = path.resolve(process.cwd(), inputFile);

  if (!fs.existsSync(inPath)) {
    console.error(`Input file does not exist: ${inPath}`);
    process.exit(1);
  }

  const inStream = fs.createReadStream(inPath, { encoding: 'utf8' });
  const rl = readline.createInterface({ input: inStream, crlfDelay: Infinity });

  const outStream = outputFile
    ? fs.createWriteStream(path.resolve(process.cwd(), outputFile), { encoding: 'utf8' })
    : process.stdout;

  let lineCount = 0;
  let truncated = false;

  function writeLine(line) {
    outStream.write(line + '\n');
  }

  function writeTruncationNotice() {
    if (truncated) return;
    truncated = true;
    writeLine('');
    writeLine('--- [TRUNCATED OUTPUT] ---');
    writeLine(`Displayed at most ${maxLines} wrapped lines.`);
    writeLine('Use the original file for full details if needed.');
  }

  for await (const rawLine of rl) {
    let line = rawLine;

    // If we've already hit the line limit, stop consuming further input.
    if (lineCount >= maxLines) {
      writeTruncationNotice();
      break;
    }

    if (line.length === 0) {
      // Preserve empty lines with standard limit enforcement
      writeLine('');
      lineCount += 1;
      continue;
    }

    // Wrap the line into chunks so that no single output line exceeds maxLineLength.
    let start = 0;
    while (start < line.length) {
      if (lineCount >= maxLines) {
        writeTruncationNotice();
        break;
      }
      const chunk = line.slice(start, start + maxLineLength);
      writeLine(chunk);
      lineCount += 1;
      start += maxLineLength;
    }

    if (lineCount >= maxLines) {
      break;
    }
  }

  // If the underlying stream had more content but we stopped early, ensure notice is written.
  if (lineCount >= maxLines && !truncated) {
    writeTruncationNotice();
  }

  // Close resources if we're writing to a file.
  await new Promise((resolve) => {
    rl.close();
    inStream.close();
    if (outStream !== process.stdout) {
      outStream.end(resolve);
    } else {
      resolve();
    }
  });
}

(async () => {
  try {
    const config = parseArgs(process.argv);
    await safeView(config);
  } catch (err) {
    console.error('[safe-view] Error:', err && err.message ? err.message : err);
    process.exit(1);
  }
})();
