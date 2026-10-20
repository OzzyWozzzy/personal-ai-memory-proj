/* Headless contract probe for the dependency-free temporal graph canvas. */
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

let callbacks = [];
let arcs = [];
let edgeStrokes = 0;
let pathLines = 0;
let pathCurves = false;
const noop = () => {};
const ctx = new Proxy({
  clearRect: () => { arcs = []; edgeStrokes = 0; },
  beginPath: () => { pathLines = 0; pathCurves = false; },
  arc: (x, y, r) => arcs.push({ x, y, r }),
  lineTo: () => { pathLines++; },
  quadraticCurveTo: () => { pathCurves = true; },
  stroke: () => { if (pathLines === 1 && !pathCurves) edgeStrokes++; },
  measureText: (text) => ({ width: String(text).length * 6 }),
}, {
  get: (target, key) => key in target ? target[key] : noop,
  set: (target, key, value) => { target[key] = value; return true; },
});
const canvas = {
  getContext: () => ctx,
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 1200, height: 800 }),
  addEventListener: noop,
  classList: { add: noop, remove: noop, toggle: noop },
};
global.requestAnimationFrame = (callback) => { callbacks.push(callback); };
global.window = { devicePixelRatio: 1, addEventListener: noop };

vm.runInThisContext(fs.readFileSync("frontend/graph.js", "utf8"));

const entities = Array.from({ length: 18 }, (_, i) => ({
  id: `entity:${i}`,
  label: i === 0 ? "Alice" : `Entity ${i}`,
  type: i % 3 === 0 ? "person" : "concept",
  node_kind: "entity",
  degree: 2,
}));
const sources = ["chat", "document", "json"].map((name) => ({
  id: `source:${name}`,
  label: name,
  type: "source",
  node_kind: "source",
  degree: 8,
}));
const episodes = Array.from({ length: 24 }, (_, i) => ({
  id: `episode:${i}`,
  label: `Episode ${i}`,
  type: "episode",
  node_kind: "episode",
  degree: 1,
}));
const relationships = entities.map((node, i) => ({
  source: node.id,
  target: entities[(i + 1) % entities.length].id,
  kind: "edge",
}));
const provenance = episodes.map((node, i) => ({
  source: node.id,
  target: sources[i % sources.length].id,
  kind: "provenance",
}));
const nodes = [...entities, ...sources, ...episodes];
const edges = [...relationships, ...provenance];

const graph = window.GraphView(canvas, {});
graph.setData({ nodes, edges });
callbacks.shift()();

assert.strictEqual(arcs.length, nodes.length, "every node should render once");
assert.strictEqual(edgeStrokes, edges.length, "every relationship should render");
assert.strictEqual(graph.setSearch("alice"), 1, "entity search should find Alice");
assert.strictEqual(graph.edgesFor("entity:0").length, 2, "adjacent facts stay inspectable");
assert.strictEqual(graph.getNode("episode:0").node_kind, "episode");
assert.ok(arcs.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)));

console.log(JSON.stringify({ nodes: nodes.length, edges: edges.length, status: "ok" }));
