const GRAVITY_WATER_KN_M3 = 9.81;

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const norm = (vector) => Math.sqrt(dot(vector, vector));

export const TA01_FEM_DEFAULTS = Object.freeze({
  slopeHeightM: 100,
  slopeWidthM: 160,
  benchCount: 5,
  benchFlatRatio: 0.28,
  cohesionKpa: 80,
  frictionAngleDeg: 32,
  unitWeightKNm3: 22,
  youngModulusMpa: 1200,
  poissonRatio: 0.28,
  permeabilityMS: 1e-7,
  drainageEfficiency: 0.35,
  waterTableM: 28,
  biotCoefficient: 0.85,
  storageCoefficient: 0.22,
  meshX: 30,
  meshY: 20
});

export const FEM_2D_METHOD = Object.freeze({
  id: "FEM2D_CST_PLANE_STRAIN_V1",
  formulation: "Elementos triangulares CST, deformación plana y elasticidad lineal",
  coupling: "Presión de poros como carga inicial de Biot; respuesta cuasiestática por hora",
  solver: "Gradiente conjugado precondicionado de Jacobi",
  safetyMetric: "Índice Mohr-Coulomb posprocesado sobre tensiones efectivas",
  scientificStatus: "FEM_2D_REAL_LINEAL_NO_CALIBRADO",
  limitation: "No incluye plasticidad, reducción no lineal de resistencia ni flujo transitorio FEM. El índice de seguridad no equivale a un SRM validado."
});

function validateScenario(input) {
  const scenario = { ...TA01_FEM_DEFAULTS, ...input };
  const ranges = {
    slopeHeightM: [20, 300], slopeWidthM: [40, 600], benchCount: [1, 12], benchFlatRatio: [0.05, 0.65],
    cohesionKpa: [1, 500], frictionAngleDeg: [5, 55], unitWeightKNm3: [10, 35], youngModulusMpa: [10, 20_000],
    poissonRatio: [0.05, 0.48], permeabilityMS: [1e-12, 1e-3], drainageEfficiency: [0, 0.95],
    waterTableM: [0, 300], biotCoefficient: [0, 1], storageCoefficient: [0.03, 0.6], meshX: [6, 60], meshY: [4, 40]
  };
  for (const [key, [minimum, maximum]] of Object.entries(ranges)) {
    const value = Number(scenario[key]);
    if (!Number.isFinite(value) || value < minimum || value > maximum) throw new Error(`${key} debe estar entre ${minimum} y ${maximum}`);
    scenario[key] = value;
  }
  scenario.benchCount = Math.round(scenario.benchCount);
  scenario.meshX = Math.round(scenario.meshX);
  scenario.meshY = Math.round(scenario.meshY);
  scenario.waterTableM = Math.min(scenario.waterTableM, scenario.slopeHeightM);
  return scenario;
}

export function ta01SurfaceElevation(x, scenario = TA01_FEM_DEFAULTS) {
  const width = scenario.slopeWidthM;
  const height = scenario.slopeHeightM;
  const benchCount = scenario.benchCount;
  const normalized = clamp(x / width, 0, 1 - Number.EPSILON);
  const benchPosition = normalized * benchCount;
  const benchIndex = Math.min(benchCount - 1, Math.floor(benchPosition));
  const local = benchPosition - benchIndex;
  const drop = height / benchCount;
  const top = height - benchIndex * drop;
  if (local <= scenario.benchFlatRatio) return top;
  const faceProgress = (local - scenario.benchFlatRatio) / (1 - scenario.benchFlatRatio);
  return Math.max(0, top - faceProgress * drop);
}

function materialAt(x, y, scenario) {
  const depth = Math.max(0, ta01SurfaceElevation(x, scenario) - y);
  if (depth <= 12) {
    return {
      id: "SOIL", depthM: depth, cohesionKpa: scenario.cohesionKpa * 0.45,
      frictionAngleDeg: Math.max(12, scenario.frictionAngleDeg - 4), youngModulusKpa: scenario.youngModulusMpa * 1000 * 0.12,
      poissonRatio: Math.min(0.44, scenario.poissonRatio + 0.08), unitWeightKNm3: scenario.unitWeightKNm3 * 0.86
    };
  }
  if (depth <= 35) {
    return {
      id: "WEATHERED", depthM: depth, cohesionKpa: scenario.cohesionKpa,
      frictionAngleDeg: scenario.frictionAngleDeg, youngModulusKpa: scenario.youngModulusMpa * 1000 * 0.55,
      poissonRatio: scenario.poissonRatio, unitWeightKNm3: scenario.unitWeightKNm3
    };
  }
  return {
    id: "ROCK", depthM: depth, cohesionKpa: scenario.cohesionKpa * 2.2,
    frictionAngleDeg: Math.min(55, scenario.frictionAngleDeg + 6), youngModulusKpa: scenario.youngModulusMpa * 1000 * 2.5,
    poissonRatio: Math.max(0.08, scenario.poissonRatio - 0.04), unitWeightKNm3: scenario.unitWeightKNm3 * 1.07
  };
}

export function createTa01Mesh(input = {}) {
  const scenario = validateScenario(input);
  const dx = scenario.slopeWidthM / scenario.meshX;
  const dy = scenario.slopeHeightM / scenario.meshY;
  const rawNodes = [];
  for (let row = 0; row <= scenario.meshY; row++) {
    for (let column = 0; column <= scenario.meshX; column++) {
      rawNodes.push({ id: rawNodes.length, x: column * dx, y: row * dy });
    }
  }
  const rawIndex = (column, row) => row * (scenario.meshX + 1) + column;
  const rawElements = [];
  const addTriangle = (ids) => {
    const points = ids.map((id) => rawNodes[id]);
    const centroidX = points.reduce((sum, point) => sum + point.x, 0) / 3;
    const centroidY = points.reduce((sum, point) => sum + point.y, 0) / 3;
    if (centroidY <= ta01SurfaceElevation(centroidX, scenario) + dy * 0.08) rawElements.push(ids);
  };
  for (let row = 0; row < scenario.meshY; row++) {
    for (let column = 0; column < scenario.meshX; column++) {
      const a = rawIndex(column, row);
      const b = rawIndex(column + 1, row);
      const c = rawIndex(column, row + 1);
      const d = rawIndex(column + 1, row + 1);
      addTriangle([a, b, d]);
      addTriangle([a, d, c]);
    }
  }
  const used = new Set(rawElements.flat());
  const remap = new Map();
  const nodes = rawNodes.filter((node) => used.has(node.id)).map((node, id) => {
    remap.set(node.id, id);
    return { id, x: node.x, y: node.y };
  });
  const elements = rawElements.map((nodeIds, id) => {
    const mapped = nodeIds.map((nodeId) => remap.get(nodeId));
    const points = mapped.map((nodeId) => nodes[nodeId]);
    const centroid = { x: points.reduce((sum, point) => sum + point.x, 0) / 3, y: points.reduce((sum, point) => sum + point.y, 0) / 3 };
    return { id, nodeIds: mapped, centroid, material: materialAt(centroid.x, centroid.y, scenario) };
  });
  return { nodes, elements, scenario, spacing: { dx, dy } };
}

function elementMatrices(element, nodes) {
  const [p1, p2, p3] = element.nodeIds.map((nodeId) => nodes[nodeId]);
  const twiceArea = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
  const area = Math.abs(twiceArea) / 2;
  if (area < 1e-10) throw new Error(`Elemento degenerado ${element.id}`);
  const b = [p2.y - p3.y, p3.y - p1.y, p1.y - p2.y];
  const c = [p3.x - p2.x, p1.x - p3.x, p2.x - p1.x];
  const scale = 1 / (2 * area);
  const B = [
    [b[0] * scale, 0, b[1] * scale, 0, b[2] * scale, 0],
    [0, c[0] * scale, 0, c[1] * scale, 0, c[2] * scale],
    [c[0] * scale, b[0] * scale, c[1] * scale, b[1] * scale, c[2] * scale, b[2] * scale]
  ];
  const { youngModulusKpa: E, poissonRatio: nu } = element.material;
  const factor = E / ((1 + nu) * (1 - 2 * nu));
  const D = [
    [factor * (1 - nu), factor * nu, 0],
    [factor * nu, factor * (1 - nu), 0],
    [0, 0, factor * (1 - 2 * nu) / 2]
  ];
  const DB = D.map((row) => B[0].map((_, column) => row.reduce((sum, value, index) => sum + value * B[index][column], 0)));
  const stiffness = Array.from({ length: 6 }, (_, row) => Array.from({ length: 6 }, (_, column) => {
    let value = 0;
    for (let index = 0; index < 3; index++) value += B[index][row] * DB[index][column];
    return value * area;
  }));
  return { area, B, D, stiffness, dofs: element.nodeIds.flatMap((nodeId) => [nodeId * 2, nodeId * 2 + 1]) };
}

export function verifyCstPatchTest() {
  const nodes = [{ id: 0, x: 0, y: 0 }, { id: 1, x: 2, y: 0 }, { id: 2, x: 0, y: 1 }, { id: 3, x: 2, y: 1 }];
  const material = { youngModulusKpa: 1_000_000, poissonRatio: 0.28 };
  const elements = [
    { id: 0, nodeIds: [0, 1, 2], material },
    { id: 1, nodeIds: [1, 3, 2], material }
  ];
  const affine = { uxX: 0.0012, uxY: -0.0004, uyX: 0.0007, uyY: -0.0009, ux0: 0.003, uy0: -0.002 };
  const expectedStrain = [affine.uxX, affine.uyY, affine.uxY + affine.uyX];
  const elementResults = elements.map((element) => {
    const matrices = elementMatrices(element, nodes);
    const displacement = element.nodeIds.flatMap((nodeId) => {
      const node = nodes[nodeId];
      return [affine.uxX * node.x + affine.uxY * node.y + affine.ux0, affine.uyX * node.x + affine.uyY * node.y + affine.uy0];
    });
    const computedStrain = multiplyMatrixVector(matrices.B, displacement);
    return { elementId: element.id, computedStrain, maximumAbsoluteError: Math.max(...computedStrain.map((value, index) => Math.abs(value - expectedStrain[index]))) };
  });
  return {
    name: "CST_AFFINE_CONSTANT_STRAIN_PATCH_TEST",
    analyticalExpectation: "Un campo de desplazamiento afín produce deformación constante exacta en todo elemento CST.",
    expectedStrain,
    elementResults,
    maximumAbsoluteError: Math.max(...elementResults.map((result) => result.maximumAbsoluteError)),
    passed: elementResults.every((result) => result.maximumAbsoluteError < 1e-12)
  };
}

/** Comprueba ensamblaje, contornos y carga de Biot con una solución afín analítica. */
function verifyGlobalAffineBenchmark(porePressureKpa, biotCoefficient) {
  const meshX = 12;
  const meshY = 8;
  const widthM = 20;
  const heightM = 10;
  const strainY = 0.001;
  const youngModulusKpa = 1_000_000;
  const poissonRatio = 0.28;
  const dx = widthM / meshX;
  const dy = heightM / meshY;
  const index = (column, row) => row * (meshX + 1) + column;
  const nodes = [];
  for (let row = 0; row <= meshY; row++) {
    for (let column = 0; column <= meshX; column++) nodes.push({ id: index(column, row), x: column * dx, y: row * dy });
  }
  const material = { youngModulusKpa, poissonRatio, unitWeightKNm3: 0 };
  const elements = [];
  for (let row = 0; row < meshY; row++) {
    for (let column = 0; column < meshX; column++) {
      const a = index(column, row), b = index(column + 1, row), c = index(column, row + 1), d = index(column + 1, row + 1);
      elements.push({ id: elements.length, nodeIds: [a, b, d], material });
      elements.push({ id: elements.length, nodeIds: [a, d, c], material });
    }
  }
  const prepared = prepareSystem({ nodes, elements, spacing: { dx, dy } });
  const load = new Float64Array(prepared.freeDofs.length);
  addBiotInitialStressLoad(prepared, load, () => porePressureKpa, biotCoefficient);
  const addForce = (nodeId, axis, force) => {
    const free = prepared.dofToFree[nodeId * 2 + axis];
    if (free >= 0) load[free] += force;
  };
  // u=(0,εy) satisface div(σ)=0 y todos los desplazamientos impuestos.
  // La tracción externa corresponde a σ_total = Dε − αpI; la contribución
  // de αpI entra por separado en el vector de cargas de Biot.
  const factor = youngModulusKpa / ((1 + poissonRatio) * (1 - 2 * poissonRatio));
  const sigmaXX = factor * poissonRatio * strainY;
  const sigmaYY = factor * (1 - poissonRatio) * strainY;
  const rightBoundaryTractionKpa = sigmaXX - biotCoefficient * porePressureKpa;
  const topBoundaryTractionKpa = sigmaYY - biotCoefficient * porePressureKpa;
  for (let row = 0; row < meshY; row++) {
    addForce(index(meshX, row), 0, rightBoundaryTractionKpa * dy / 2);
    addForce(index(meshX, row + 1), 0, rightBoundaryTractionKpa * dy / 2);
  }
  for (let column = 0; column < meshX; column++) {
    addForce(index(column, meshY), 1, topBoundaryTractionKpa * dx / 2);
    addForce(index(column + 1, meshY), 1, topBoundaryTractionKpa * dx / 2);
  }
  const solved = solvePcg(prepared.rows, load);
  const displacement = fullDisplacement(prepared, solved.solution);
  const maximumAbsoluteErrorM = nodes.reduce((maximum, node) => Math.max(maximum,
    Math.abs(displacement[node.id * 2]),
    Math.abs(displacement[node.id * 2 + 1] - strainY * node.y)), 0);
  return {
    name: porePressureKpa ? "GLOBAL_AFFINE_BIOT_PRESSURE_ANALYTICAL_BENCHMARK" : "GLOBAL_AFFINE_ELASTICITY_ANALYTICAL_BENCHMARK",
    scientificStatus: porePressureKpa ? "VERIFICACION_ANALITICA_CARGA_BIOT_UNIFORME_NO_VALIDACION_HIDROGEOLOGICA" : "VERIFICACION_ANALITICA_ELASTICIDAD_LINEAL_NO_VALIDACION_GEOTECNICA",
    analyticalExpectation: "u_x=0, u_y=εy; tracción total (Dε−αpI)n en derecha y techo; empotramiento inferior y rodillo izquierdo.",
    meshX, meshY, nodeCount: nodes.length, elementCount: elements.length,
    youngModulusKpa, poissonRatio, strainY, sigmaXXKpa: sigmaXX, sigmaYYKpa: sigmaYY,
    porePressureKpa, biotCoefficient, rightBoundaryTractionKpa, topBoundaryTractionKpa,
    maximumAbsoluteDisplacementErrorM: maximumAbsoluteErrorM,
    relativeMaximumDisplacementError: maximumAbsoluteErrorM / (strainY * heightM),
    solverRelativeResidual: solved.relativeResidual,
    passed: solved.converged && maximumAbsoluteErrorM < 1e-7
  };
}

export function verifyGlobalAffineElasticityBenchmark() {
  return verifyGlobalAffineBenchmark(0, 0);
}

export function verifyGlobalAffineBiotPressureBenchmark() {
  return verifyGlobalAffineBenchmark(120, 0.85);
}

function prepareSystem(mesh) {
  const totalDofs = mesh.nodes.length * 2;
  const fixed = new Set();
  const tolerance = Math.min(mesh.spacing.dx, mesh.spacing.dy) * 1e-5;
  for (const node of mesh.nodes) {
    if (node.y <= tolerance) {
      fixed.add(node.id * 2);
      fixed.add(node.id * 2 + 1);
    } else if (node.x <= tolerance) {
      fixed.add(node.id * 2);
    }
  }
  const freeDofs = [];
  const dofToFree = new Int32Array(totalDofs).fill(-1);
  for (let dof = 0; dof < totalDofs; dof++) {
    if (!fixed.has(dof)) {
      dofToFree[dof] = freeDofs.length;
      freeDofs.push(dof);
    }
  }
  const rows = Array.from({ length: freeDofs.length }, () => new Map());
  const gravityLoad = new Float64Array(freeDofs.length);
  const preparedElements = mesh.elements.map((element) => ({ ...element, ...elementMatrices(element, mesh.nodes) }));
  for (const element of preparedElements) {
    for (let localRow = 0; localRow < 6; localRow++) {
      const globalRow = element.dofs[localRow];
      const freeRow = dofToFree[globalRow];
      if (freeRow < 0) continue;
      if (localRow % 2 === 1) gravityLoad[freeRow] -= element.material.unitWeightKNm3 * element.area / 3;
      for (let localColumn = 0; localColumn < 6; localColumn++) {
        const freeColumn = dofToFree[element.dofs[localColumn]];
        if (freeColumn < 0) continue;
        rows[freeRow].set(freeColumn, (rows[freeRow].get(freeColumn) || 0) + element.stiffness[localRow][localColumn]);
      }
    }
  }
  return { rows, gravityLoad, freeDofs, dofToFree, fixed, preparedElements, totalDofs };
}

function multiplySparse(rows, vector) {
  const result = new Float64Array(rows.length);
  for (let row = 0; row < rows.length; row++) {
    let value = 0;
    for (const [column, coefficient] of rows[row]) value += coefficient * vector[column];
    result[row] = value;
  }
  return result;
}

function solvePcg(rows, rightHandSide, tolerance = 1e-8, maximumIterations = 1800) {
  const size = rightHandSide.length;
  const x = new Float64Array(size);
  const r = Float64Array.from(rightHandSide);
  const diagonal = Float64Array.from(rows, (row, index) => row.get(index) || 1);
  const z = Float64Array.from(r, (value, index) => value / diagonal[index]);
  const p = Float64Array.from(z);
  let rz = dot(r, z);
  const rightNorm = Math.max(norm(rightHandSide), 1e-20);
  let relativeResidual = norm(r) / rightNorm;
  let iteration = 0;
  while (relativeResidual > tolerance && iteration < maximumIterations) {
    const ap = multiplySparse(rows, p);
    const denominator = dot(p, ap);
    if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-30) break;
    const alpha = rz / denominator;
    for (let index = 0; index < size; index++) {
      x[index] += alpha * p[index];
      r[index] -= alpha * ap[index];
      z[index] = r[index] / diagonal[index];
    }
    const nextRz = dot(r, z);
    relativeResidual = norm(r) / rightNorm;
    if (relativeResidual <= tolerance) {
      iteration += 1;
      break;
    }
    const beta = nextRz / rz;
    for (let index = 0; index < size; index++) p[index] = z[index] + beta * p[index];
    rz = nextRz;
    iteration += 1;
  }
  return { solution: x, iterations: iteration, relativeResidual, converged: relativeResidual <= tolerance };
}

function infiltrationEfficiency(permeabilityMS) {
  const normalizedLog = (Math.log10(permeabilityMS) + 12) / 9;
  return clamp(0.12 + normalizedLog * 0.78, 0.08, 0.9);
}

function porePressureAt(x, y, cumulativeRainfallMm, scenario) {
  const hydrostaticHead = Math.max(0, scenario.waterTableM - y);
  const depth = Math.max(0, ta01SurfaceElevation(x, scenario) - y);
  const retainedRainM = cumulativeRainfallMm / 1000 * infiltrationEfficiency(scenario.permeabilityMS) * (1 - scenario.drainageEfficiency);
  const rainfallHead = retainedRainM / scenario.storageCoefficient * Math.exp(-depth / 14);
  return GRAVITY_WATER_KN_M3 * (hydrostaticHead + rainfallHead);
}

function addBiotInitialStressLoad(prepared, load, pressureAtElement, biotCoefficient) {
  for (const element of prepared.preparedElements) {
    const initialStressKpa = biotCoefficient * pressureAtElement(element);
    for (let localDof = 0; localDof < 6; localDof++) {
      const freeDof = prepared.dofToFree[element.dofs[localDof]];
      if (freeDof < 0) continue;
      load[freeDof] += (element.B[0][localDof] + element.B[1][localDof]) * initialStressKpa * element.area;
    }
  }
}

function buildLoad(prepared, cumulativeRainfallMm, scenario) {
  const load = Float64Array.from(prepared.gravityLoad);
  addBiotInitialStressLoad(prepared, load,
    (element) => porePressureAt(element.centroid.x, element.centroid.y, cumulativeRainfallMm, scenario),
    scenario.biotCoefficient);
  return load;
}

function fullDisplacement(prepared, freeSolution) {
  const displacement = new Float64Array(prepared.totalDofs);
  prepared.freeDofs.forEach((globalDof, index) => { displacement[globalDof] = freeSolution[index]; });
  return displacement;
}

function multiplyMatrixVector(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
}

function summarizeStep(mesh, prepared, displacement, baselineDisplacement, cumulativeRainfallMm) {
  const nodalPore = new Float64Array(mesh.nodes.length);
  const nodalPoreWeight = new Uint16Array(mesh.nodes.length);
  const safetyIndices = [];
  let effectiveStressSum = 0;
  let effectiveStressArea = 0;
  for (const element of prepared.preparedElements) {
    const localDisplacement = element.dofs.map((dof) => displacement[dof]);
    const strain = multiplyMatrixVector(element.B, localDisplacement);
    const stress = multiplyMatrixVector(element.D, strain);
    const porePressureKpa = porePressureAt(element.centroid.x, element.centroid.y, cumulativeRainfallMm, mesh.scenario);
    element.nodeIds.forEach((nodeId) => {
      nodalPore[nodeId] += porePressureKpa;
      nodalPoreWeight[nodeId] += 1;
    });
    const compressionX = -stress[0];
    const compressionY = -stress[1];
    const meanCompression = (compressionX + compressionY) / 2;
    const maximumShear = Math.sqrt(((compressionX - compressionY) / 2) ** 2 + stress[2] ** 2);
    if (element.material.depthM > 1 && meanCompression > 0.5 && maximumShear > 0.05) {
      const phi = element.material.frictionAngleDeg * Math.PI / 180;
      const capacity = element.material.cohesionKpa + meanCompression * Math.tan(phi);
      safetyIndices.push(capacity / maximumShear);
      effectiveStressSum += meanCompression * element.area;
      effectiveStressArea += element.area;
    }
  }
  const nodeResults = mesh.nodes.map((node) => {
    const uxMm = displacement[node.id * 2] * 1000;
    const uyMm = displacement[node.id * 2 + 1] * 1000;
    const deltaUxMm = (displacement[node.id * 2] - baselineDisplacement[node.id * 2]) * 1000;
    const deltaUyMm = (displacement[node.id * 2 + 1] - baselineDisplacement[node.id * 2 + 1]) * 1000;
    return {
      id: node.id, xM: Number(node.x.toFixed(4)), yM: Number(node.y.toFixed(4)),
      uxMm: Number(uxMm.toFixed(5)), uyMm: Number(uyMm.toFixed(5)),
      displacementMm: Number(Math.hypot(uxMm, uyMm).toFixed(5)),
      deltaUxMm: Number(deltaUxMm.toFixed(5)), deltaUyMm: Number(deltaUyMm.toFixed(5)),
      rainfallInducedDisplacementMm: Number(Math.hypot(deltaUxMm, deltaUyMm).toFixed(5)),
      porePressureKpa: Number((nodalPoreWeight[node.id] ? nodalPore[node.id] / nodalPoreWeight[node.id] : 0).toFixed(4))
    };
  });
  const sortedSafety = safetyIndices.filter(Number.isFinite).sort((a, b) => a - b);
  const safetyIndex = sortedSafety.length ? sortedSafety[Math.floor((sortedSafety.length - 1) * 0.05)] : null;
  const surfaceNodes = nodeResults.filter((node) => Math.abs(node.yM - ta01SurfaceElevation(node.xM, mesh.scenario)) <= mesh.spacing.dy * 0.7);
  const crestCandidates = surfaceNodes.filter((node) => node.xM <= mesh.scenario.slopeWidthM * 0.2);
  const toeCandidates = surfaceNodes.filter((node) => node.xM >= mesh.scenario.slopeWidthM * 0.8);
  const maximum = nodeResults.reduce((best, current) => current.displacementMm > best.displacementMm ? current : best, nodeResults[0]);
  const maximumRainfallInduced = nodeResults.reduce((best, current) => current.rainfallInducedDisplacementMm > best.rainfallInducedDisplacementMm ? current : best, nodeResults[0]);
  const maximumFrom = (items) => items.length ? Math.max(...items.map((node) => node.displacementMm)) : 0;
  return {
    nodes: nodeResults,
    maximumDisplacementMm: maximum.displacementMm,
    maximumDisplacementNodeId: maximum.id,
    maximumRainfallInducedDisplacementMm: maximumRainfallInduced.rainfallInducedDisplacementMm,
    maximumRainfallInducedNodeId: maximumRainfallInduced.id,
    crestDisplacementMm: Number(maximumFrom(crestCandidates).toFixed(5)),
    toeDisplacementMm: Number(maximumFrom(toeCandidates).toFixed(5)),
    maximumPorePressureKpa: Number(Math.max(...nodeResults.map((node) => node.porePressureKpa)).toFixed(4)),
    meanEffectiveStressKpa: Number((effectiveStressArea ? effectiveStressSum / effectiveStressArea : 0).toFixed(4)),
    mohrCoulombSafetyIndex: safetyIndex === null ? null : Number(clamp(safetyIndex, 0.05, 8).toFixed(4))
  };
}

function riskFromSafetyIndex(index) {
  if (index === null) return "UNAVAILABLE";
  if (index < 1) return "CRITICO";
  if (index < 1.15) return "ALERTA";
  if (index < 1.3) return "VIGILANCIA";
  return "NORMAL";
}

export function runFem2D(input = {}, rainfallProfileMmH = [0], solverOptions = {}) {
  const solverTolerance = Number(solverOptions.tolerance ?? 1e-8);
  if (!Number.isFinite(solverTolerance) || solverTolerance <= 0 || solverTolerance >= 1) {
    throw new Error("La tolerancia relativa del solver debe estar entre 0 y 1");
  }
  const mesh = createTa01Mesh(input);
  const scenario = mesh.scenario;
  if (!Array.isArray(rainfallProfileMmH) || !rainfallProfileMmH.length || rainfallProfileMmH.length > 168) throw new Error("El perfil de lluvia FEM debe contener entre 1 y 168 horas");
  const rainfall = rainfallProfileMmH.map((value) => Number(value));
  if (rainfall.some((value) => !Number.isFinite(value) || value < 0 || value > 300)) throw new Error("Cada valor de lluvia FEM debe estar entre 0 y 300 mm/h");
  const prepared = prepareSystem(mesh);
  const baselineLoad = buildLoad(prepared, 0, scenario);
  const baselineSolution = solvePcg(prepared.rows, baselineLoad, solverTolerance);
  if (!baselineSolution.converged) throw new Error(`El estado gravitacional inicial no convergió; residuo ${baselineSolution.relativeResidual}`);
  const baselineDisplacement = fullDisplacement(prepared, baselineSolution.solution);
  const timeSeries = [];
  let cumulativeRainfallMm = 0;
  let finalSummary = null;
  let totalIterations = 0;
  let maximumResidual = 0;
  for (let hour = 1; hour <= rainfall.length; hour++) {
    cumulativeRainfallMm += rainfall[hour - 1];
    const load = buildLoad(prepared, cumulativeRainfallMm, scenario);
    const solution = solvePcg(prepared.rows, load, solverTolerance);
    if (!solution.converged) throw new Error(`El solver FEM no convergió en la hora ${hour}; residuo ${solution.relativeResidual}`);
    const displacement = fullDisplacement(prepared, solution.solution);
    const summary = summarizeStep(mesh, prepared, displacement, baselineDisplacement, cumulativeRainfallMm);
    totalIterations += solution.iterations;
    maximumResidual = Math.max(maximumResidual, solution.relativeResidual);
    timeSeries.push({
      hour,
      rainfallMmH: Number(rainfall[hour - 1].toFixed(6)),
      cumulativeRainfallMm: Number(cumulativeRainfallMm.toFixed(6)),
      maximumDisplacementMm: summary.maximumDisplacementMm,
      maximumRainfallInducedDisplacementMm: summary.maximumRainfallInducedDisplacementMm,
      crestDisplacementMm: summary.crestDisplacementMm,
      toeDisplacementMm: summary.toeDisplacementMm,
      maximumPorePressureKpa: summary.maximumPorePressureKpa,
      meanEffectiveStressKpa: summary.meanEffectiveStressKpa,
      mohrCoulombSafetyIndex: summary.mohrCoulombSafetyIndex,
      riskLevel: riskFromSafetyIndex(summary.mohrCoulombSafetyIndex),
      solverIterations: solution.iterations,
      solverResidual: Number(solution.relativeResidual.toExponential(6)),
      // Se conserva el campo nodal de cada estado para reproducir en el visor
      // la respuesta FEM horaria, no una interpolación del pronóstico reducido.
      nodes: summary.nodes
    });
    finalSummary = summary;
  }
  const finalNodes = finalSummary.nodes;
  const elements = mesh.elements.map((element) => ({ id: element.id, nodeIds: element.nodeIds, material: element.material.id }));
  return {
    method: FEM_2D_METHOD,
    scenario,
    mesh: { nodeCount: mesh.nodes.length, elementCount: mesh.elements.length, nodes: finalNodes, elements },
    timeSeries,
    summary: {
      durationHours: rainfall.length,
      totalRainfallMm: Number(cumulativeRainfallMm.toFixed(6)),
      maximumDisplacementMm: finalSummary.maximumDisplacementMm,
      maximumRainfallInducedDisplacementMm: finalSummary.maximumRainfallInducedDisplacementMm,
      crestDisplacementMm: finalSummary.crestDisplacementMm,
      toeDisplacementMm: finalSummary.toeDisplacementMm,
      maximumPorePressureKpa: finalSummary.maximumPorePressureKpa,
      meanEffectiveStressKpa: finalSummary.meanEffectiveStressKpa,
      mohrCoulombSafetyIndex: finalSummary.mohrCoulombSafetyIndex,
      riskLevel: riskFromSafetyIndex(finalSummary.mohrCoulombSafetyIndex),
      converged: true,
      baselineSolverIterations: baselineSolution.iterations,
      baselineSolverResidual: Number(baselineSolution.relativeResidual.toExponential(6)),
      solverRelativeTolerance: solverTolerance,
      averageSolverIterations: Number((totalIterations / rainfall.length).toFixed(2)),
      maximumSolverResidual: Number(maximumResidual.toExponential(6))
    }
  };
}

/** Exporta el problema incremental K Δu = Δf de TA-01 para un benchmark neuronal de forma débil. */
export function buildTa01IncrementalEquilibrium(input = {}, cumulativeRainfallMm = 48) {
  const rainfallMm = Number(cumulativeRainfallMm);
  if (!Number.isFinite(rainfallMm) || rainfallMm < 0 || rainfallMm > 500) throw new Error("cumulativeRainfallMm debe estar entre 0 y 500");
  const mesh = createTa01Mesh(input);
  const prepared = prepareSystem(mesh);
  const dryLoad = buildLoad(prepared, 0, mesh.scenario);
  const wetLoad = buildLoad(prepared, rainfallMm, mesh.scenario);
  const incrementalLoad = Float64Array.from(wetLoad, (value, index) => value - dryLoad[index]);
  const solution = solvePcg(prepared.rows, incrementalLoad);
  if (!solution.converged) throw new Error(`El problema incremental no convergió: ${solution.relativeResidual}`);
  const full = fullDisplacement(prepared, solution.solution);
  return {
    id: "TA01-INCREMENTAL-RAIN-EQUILIBRIUM-V1",
    scientificStatus: "FEM_LINEAL_SEMISINTETICO_NO_CALIBRADO",
    formulation: "Equilibrio débil discreto K Δu = Δf de presión de poros, condiciones de desplazamiento homogéneas heredadas del FEM 2D",
    cumulativeRainfallMm: rainfallMm,
    scenario: mesh.scenario,
    mesh: {
      nodes: mesh.nodes.map((node) => ({ id: node.id, xM: node.x, yM: node.y })),
      elements: mesh.elements.map((element) => ({ id: element.id, nodeIds: element.nodeIds, material: element.material.id }))
    },
    system: {
      totalDofs: prepared.totalDofs,
      freeDofs: prepared.freeDofs,
      stiffnessRows: prepared.rows.map((row) => [...row].sort((a, b) => a[0] - b[0])),
      incrementalLoadKn: [...incrementalLoad],
      referenceDisplacementM: [...full],
      pcgIterations: solution.iterations,
      pcgRelativeResidual: solution.relativeResidual
    }
  };
}
