import { FEM_2D_METHOD, TA01_FEM_DEFAULTS, runFem2D } from "./fem-2d.js";
import { rainfallRecordForDate } from "./rainfall-history.js";

export const TA01_STUDY_CASE = Object.freeze({
  id: "TA01",
  name: "Talud Andino Experimental 01",
  objective: "Generar respuestas mecánicas FEM 2D ante lluvia diaria de reanálisis y variación geotécnica controlada.",
  geometry: { slopeHeightM: 100, slopeWidthM: 160, benchCount: 5, materialLayers: ["SOIL", "WEATHERED", "ROCK"] },
  forecastHorizonsHours: [1, 6, 24],
  dataClassification: "SEMI_SINTETICO",
  scientificStatus: "CASO_COMPUTACIONAL_NO_CALIBRADO"
});

export const TA01_PARAMETER_RANGES = Object.freeze({
  cohesionKpa: [60, 240],
  frictionAngleDeg: [26, 44],
  unitWeightKNm3: [18, 26],
  youngModulusMpa: [400, 3000],
  permeabilityMS: [1e-10, 3e-6],
  drainageEfficiency: [0, 0.85],
  waterTableRatio: [0.1, 0.65],
  storageCoefficient: [0.12, 0.35]
});

const CSV_COLUMNS = [
  "scenario_id", "source_date", "simulation_hour", "rainfall_mm_h", "cumulative_rainfall_mm", "observed_daily_rainfall_mm",
  "temporal_profile", "cohesion_kpa", "friction_angle_deg", "unit_weight_kn_m3", "young_modulus_mpa", "poisson_ratio",
  "permeability_m_s", "drainage_efficiency", "water_table_m", "storage_coefficient", "mesh_nodes", "mesh_elements",
  "gravity_total_max_displacement_mm", "rainfall_induced_max_displacement_mm", "crest_displacement_mm", "toe_displacement_mm",
  "maximum_pore_pressure_kpa", "mean_effective_stress_kpa", "mohr_coulomb_safety_index", "risk_level",
  "solver_iterations", "solver_residual", "rainfall_source", "mechanical_response_source", "parameter_source", "scientific_status"
];

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4_294_967_296;
  };
}

function shuffle(values, random) {
  const result = [...values];
  for (let index = result.length - 1; index > 0; index--) {
    const target = Math.floor(random() * (index + 1));
    [result[index], result[target]] = [result[target], result[index]];
  }
  return result;
}

function latinHypercube(count, dimensions, seed) {
  const random = seededRandom(seed);
  const samples = Array.from({ length: count }, () => ({}));
  for (const dimension of dimensions) {
    const bins = shuffle(Array.from({ length: count }, (_, index) => index), random);
    bins.forEach((bin, sampleIndex) => {
      samples[sampleIndex][dimension] = (bin + random()) / count;
    });
  }
  return samples;
}

function linear(minimum, maximum, sample) {
  return minimum + (maximum - minimum) * sample;
}

function logarithmic(minimum, maximum, sample) {
  return 10 ** linear(Math.log10(minimum), Math.log10(maximum), sample);
}

function rounded(value, decimals = 6) {
  return Number(value.toFixed(decimals));
}

export function estimatedRectangularRainfallProfile(dailyTotalMm, concentrationHours = 24) {
  const total = Number(dailyTotalMm);
  const duration = Number(concentrationHours);
  if (!Number.isFinite(total) || total < 0) throw new Error("dailyTotalMm debe ser no negativo");
  if (![6, 12, 24].includes(duration)) throw new Error("concentrationHours debe ser 6, 12 o 24");
  const profile = Array(24).fill(0);
  if (total === 0) return profile;
  const firstHour = Math.floor((24 - duration) / 2);
  const base = rounded(total / duration, 8);
  for (let hour = firstHour; hour < firstHour + duration; hour++) profile[hour] = base;
  profile[firstHour + duration - 1] = rounded(total - base * (duration - 1), 8);
  return profile;
}

function allowedParameterOverrides(input = {}) {
  const allowed = [
    "cohesionKpa", "frictionAngleDeg", "unitWeightKNm3", "youngModulusMpa", "poissonRatio", "permeabilityMS",
    "drainageEfficiency", "waterTableM", "storageCoefficient", "biotCoefficient", "meshX", "meshY"
  ];
  return Object.fromEntries(Object.entries(input).filter(([key]) => allowed.includes(key)));
}

export function runTa01FemCase(rainfallDataset, input = {}) {
  const date = input.date || rainfallDataset?.summary?.recommendedDate;
  const rainfall = rainfallRecordForDate(rainfallDataset, date);
  const concentrationHours = Number(input.concentrationHours || 24);
  const profile = estimatedRectangularRainfallProfile(rainfall.rainfallMmDay, concentrationHours);
  const scenario = {
    ...TA01_FEM_DEFAULTS,
    slopeHeightM: TA01_STUDY_CASE.geometry.slopeHeightM,
    slopeWidthM: TA01_STUDY_CASE.geometry.slopeWidthM,
    benchCount: TA01_STUDY_CASE.geometry.benchCount,
    ...allowedParameterOverrides(input.parameters)
  };
  const result = runFem2D(scenario, profile);
  return {
    id: `FEM-${Date.now()}`,
    generatedAt: new Date().toISOString(),
    studyCase: TA01_STUDY_CASE,
    rainfall: {
      source: rainfallDataset.metadata.id,
      date: rainfall.date,
      observedDailyTotalMm: rainfall.rainfallMmDay,
      temporalProfile: `RECTANGULAR_${concentrationHours}H_ESTIMATED`,
      hourlyProfileMmH: profile
    },
    ...result
  };
}

function designScenario(sample, index, rainfallDataset, options) {
  const ranges = TA01_PARAMETER_RANGES;
  const sortedRainfall = [...rainfallDataset.records].sort((a, b) => a.rainfallMmDay - b.rainfallMmDay || a.date.localeCompare(b.date));
  const rainfallIndex = Math.min(sortedRainfall.length - 1, Math.floor(sample.rainfall * sortedRainfall.length));
  const wettest = rainfallDataset.summary.wettestDays;
  const rainfall = index === 0
    ? wettest[0]
    : options.scenarioCount >= 3 && index === 1
      ? wettest[1]
      : options.scenarioCount >= 3 && index === options.scenarioCount - 1
        ? sortedRainfall[0]
        : sortedRainfall[rainfallIndex];
  const concentrationOptions = [24, 12, 6];
  const concentrationHours = rainfall.rainfallMmDay === 0 ? 24 : concentrationOptions[index % concentrationOptions.length];
  const scenario = {
    ...TA01_FEM_DEFAULTS,
    cohesionKpa: rounded(linear(...ranges.cohesionKpa, sample.cohesionKpa), 3),
    frictionAngleDeg: rounded(linear(...ranges.frictionAngleDeg, sample.frictionAngleDeg), 3),
    unitWeightKNm3: rounded(linear(...ranges.unitWeightKNm3, sample.unitWeightKNm3), 3),
    youngModulusMpa: rounded(linear(...ranges.youngModulusMpa, sample.youngModulusMpa), 3),
    permeabilityMS: logarithmic(...ranges.permeabilityMS, sample.permeabilityMS),
    drainageEfficiency: rounded(linear(...ranges.drainageEfficiency, sample.drainageEfficiency), 5),
    waterTableM: rounded(TA01_STUDY_CASE.geometry.slopeHeightM * linear(...ranges.waterTableRatio, sample.waterTableRatio), 3),
    storageCoefficient: rounded(linear(...ranges.storageCoefficient, sample.storageCoefficient), 5),
    meshX: options.meshX,
    meshY: options.meshY
  };
  return {
    scenarioId: `TA01-S${String(index + 1).padStart(4, "0")}`,
    rainfall,
    concentrationHours,
    profile: estimatedRectangularRainfallProfile(rainfall.rainfallMmDay, concentrationHours),
    scenario
  };
}

export function generateTa01Dataset(rainfallDataset, input = {}) {
  if (!rainfallDataset?.records?.length) throw new Error("Se requiere el histórico NASA POWER para generar TA-01");
  const scenarioCount = Number(input.scenarioCount ?? 24);
  const seed = Number(input.seed ?? 20260915);
  const meshX = Number(input.meshX ?? 14);
  const meshY = Number(input.meshY ?? 9);
  const solverTolerance = Number(input.solverTolerance ?? 1e-8);
  if (!Number.isInteger(scenarioCount) || scenarioCount < 1 || scenarioCount > 2000) throw new Error("scenarioCount debe ser un entero entre 1 y 2000");
  if (!Number.isInteger(seed) || seed < 0) throw new Error("seed debe ser un entero no negativo");
  if (!Number.isFinite(solverTolerance) || solverTolerance <= 0 || solverTolerance >= 1) throw new Error("solverTolerance debe estar entre 0 y 1");
  const dimensions = [...Object.keys(TA01_PARAMETER_RANGES), "rainfall"];
  const samples = latinHypercube(scenarioCount, dimensions, seed);
  const rows = [];
  const scenarios = [];
  for (let index = 0; index < scenarioCount; index++) {
    const design = designScenario(samples[index], index, rainfallDataset, { meshX, meshY, scenarioCount });
    const result = runFem2D(design.scenario, design.profile, { tolerance: solverTolerance });
    const temporalProfile = `RECTANGULAR_${design.concentrationHours}H_ESTIMATED`;
    for (const step of result.timeSeries) {
      rows.push({
        scenario_id: design.scenarioId,
        source_date: design.rainfall.date,
        simulation_hour: step.hour,
        rainfall_mm_h: step.rainfallMmH,
        cumulative_rainfall_mm: step.cumulativeRainfallMm,
        observed_daily_rainfall_mm: design.rainfall.rainfallMmDay,
        temporal_profile: temporalProfile,
        cohesion_kpa: design.scenario.cohesionKpa,
        friction_angle_deg: design.scenario.frictionAngleDeg,
        unit_weight_kn_m3: design.scenario.unitWeightKNm3,
        young_modulus_mpa: design.scenario.youngModulusMpa,
        poisson_ratio: design.scenario.poissonRatio,
        permeability_m_s: design.scenario.permeabilityMS,
        drainage_efficiency: design.scenario.drainageEfficiency,
        water_table_m: design.scenario.waterTableM,
        storage_coefficient: design.scenario.storageCoefficient,
        mesh_nodes: result.mesh.nodeCount,
        mesh_elements: result.mesh.elementCount,
        gravity_total_max_displacement_mm: step.maximumDisplacementMm,
        rainfall_induced_max_displacement_mm: step.maximumRainfallInducedDisplacementMm,
        crest_displacement_mm: step.crestDisplacementMm,
        toe_displacement_mm: step.toeDisplacementMm,
        maximum_pore_pressure_kpa: step.maximumPorePressureKpa,
        mean_effective_stress_kpa: step.meanEffectiveStressKpa,
        mohr_coulomb_safety_index: step.mohrCoulombSafetyIndex,
        risk_level: step.riskLevel,
        solver_iterations: step.solverIterations,
        solver_residual: step.solverResidual,
        rainfall_source: rainfallDataset.metadata.id,
        mechanical_response_source: FEM_2D_METHOD.id,
        parameter_source: "TA01_ASSUMPTION_RANGE",
        scientific_status: FEM_2D_METHOD.scientificStatus
      });
    }
    scenarios.push({
      scenarioId: design.scenarioId,
      sourceDate: design.rainfall.date,
      observedDailyRainfallMm: design.rainfall.rainfallMmDay,
      temporalProfile,
      parameters: design.scenario,
      summary: result.summary
    });
  }
  return {
    manifest: {
      id: `TA01-DATASET-S${scenarioCount}-SEED${seed}`,
      generatedAt: new Date().toISOString(),
      studyCase: TA01_STUDY_CASE,
      femMethod: FEM_2D_METHOD,
      rainfallSource: rainfallDataset.metadata,
      scenarioCount,
      rowCount: rows.length,
      hoursPerScenario: 24,
      seed,
      solverRelativeTolerance: solverTolerance,
      samplingMethod: "LATIN_HYPERCUBE",
      parameterRanges: TA01_PARAMETER_RANGES,
      splitRule: "Separar entrenamiento, validación y prueba por fecha de lluvia; ningún scenario_id ni sourceDate puede cruzar particiones.",
      targetRecommendation: "rainfall_induced_max_displacement_mm",
      scientificStatus: "SEMI_SINTETICO_NO_CALIBRADO"
    },
    scenarios,
    rows
  };
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function ta01DatasetToCsv(dataset) {
  return [CSV_COLUMNS.join(","), ...dataset.rows.map((row) => CSV_COLUMNS.map((column) => csvCell(row[column])).join(","))].join("\n") + "\n";
}

export { CSV_COLUMNS as TA01_CSV_COLUMNS };
