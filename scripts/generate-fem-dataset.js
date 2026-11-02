import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseNasaPowerDailyCsv } from "../src/core/rainfall-history.js";
import { generateTa01Dataset, ta01DatasetToCsv } from "../src/core/study-case-ta01.js";

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const match = process.argv.slice(2).find((value) => value.startsWith(prefix));
  return match ? match.slice(prefix.length) : fallback;
}

const scenarioCount = Number(argument("scenarios", "24"));
const seed = Number(argument("seed", "20260915"));
const meshX = Number(argument("mesh-x", "14"));
const meshY = Number(argument("mesh-y", "9"));
const solverTolerance = Number(argument("solver-tolerance", "1e-8"));
const outputPath = path.resolve(projectDir, argument("output", "data/generated/ta01-fem-dataset.csv"));
const manifestPath = outputPath.replace(/\.csv$/i, "-manifest.json");
const rainfallPath = path.join(projectDir, "data", "rainfall", "pasco-nasa-power-2020-2025.csv");

const rainfallDataset = parseNasaPowerDailyCsv(await readFile(rainfallPath, "utf8"));
const startedAt = performance.now();
const dataset = generateTa01Dataset(rainfallDataset, { scenarioCount, seed, meshX, meshY, solverTolerance });
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, ta01DatasetToCsv(dataset), "utf8");
await writeFile(manifestPath, JSON.stringify({ ...dataset.manifest, scenarios: dataset.scenarios }, null, 2) + "\n", "utf8");

console.log(JSON.stringify({
  csv: path.relative(projectDir, outputPath),
  manifest: path.relative(projectDir, manifestPath),
  scenarios: dataset.manifest.scenarioCount,
  rows: dataset.manifest.rowCount,
  elapsedSeconds: Number(((performance.now() - startedAt) / 1000).toFixed(3))
}, null, 2));
