#!/usr/bin/env node
"use strict";

// Rasterize the SVG figure sources as high-quality JPEGs for embedding in DOCX.
// Requires the `sharp` Node package in the execution environment.
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const bundledRequire = createRequire("C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/package.json");
const sharp = bundledRequire("sharp");
const out = path.join(root, "artifacts", "manuscript-review-2026-10-29");
const figures = ["figura1", "figura2", "figura3", "figura4", "figura5", "figura6"];

Promise.all(figures.map((name) =>
  sharp(path.join(out, `${name}.svg`))
    .jpeg({ quality: 94 })
    .toFile(path.join(out, `${name}.jpg`))
)).then((results) => {
  for (let index = 0; index < results.length; index += 1) {
    process.stdout.write(`${figures[index]}: ${results[index].width}x${results[index].height}\n`);
  }
}).catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
