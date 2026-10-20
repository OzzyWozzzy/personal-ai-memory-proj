/* graph.js — self-contained Obsidian-style 2D canvas graph (no libraries).
 *
 * Nodes settle into a free-form force-directed map using grid-bucketed
 * repulsion, relationship springs, and a weak two-axis center pull. There is
 * deliberately no silhouette or prescribed outline. Positions persist across
 * setData() so scrubbing time adds/retires edges without relaying everything.
 *
 * Navigation: drag empty space to pan, wheel or the zoom slider to zoom,
 * drag a node to move it. Labels appear only while a node is hovered — the
 * view never floods with text no matter how far you zoom in.
 *
 * window.GraphView(canvas, opts):
 *   opts.onSelect(node|null), opts.onHoverEdge(edge|null, cx, cy),
 *   opts.onZoom(pct)  — 0..100, so a UI slider can follow wheel/reset zoom.
 */
(function () {
  "use strict";

  // Compact palette for entities and source-provenance nodes.
  var TYPE_COLORS = {
    person: "#a78bfa", org: "#5eb3ff", account: "#5eb3ff",
    policy: "#f6c76b", concept: "#5ce0a5", value: "#64748b",
    episode: "#f5a45c", source: "#f27bb2",
    fictionalcharacter: "#fb6f92", creativework: "#e879f9",
    thing: "#93a7c4"
  };
  var DEFAULT_FILL = "#93a7c4";
  var BG = "#070a14";

  function normalizedType(raw) {
    var t = String(raw || "thing").toLowerCase();
    if (t === "organization" || t === "company") return "org";
    if (t === "location" || t === "place" || t === "city") return "concept";
    return t;
  }
  function colorFor(n) { return TYPE_COLORS[normalizedType(n.type)] || DEFAULT_FILL; }

  // Importance = degree. The fitted view keeps even leaves comfortably
  // clickable, while hubs read as anchors instead of oversized bubbles.
  function radiusFor(n) {
    if (n.node_kind === "episode") return 3.2;
    var d = n.degree || 0;
    return Math.max(4.8, Math.min(12, 4.8 + Math.sqrt(d) * 2.15));
  }

  function GraphView(canvas, opts) {
    opts = opts || {};
    var ctx = canvas.getContext("2d");
    var dpr = window.devicePixelRatio || 1;

    var nodes = [], byId = new Map(), edges = [], adj = new Map();
    var k = 1, tx = 0, ty = 0, baseZoom = 1;
    var alpha = 0, alphaMin = 0.004, alphaDecay = 0.02, velocityDecay = 0.86;
    var pastMode = false, layoutScale = 1;

    var hoverId = null, selectedId = null, searchHit = null, searchRelated = null, hoverEdge = null;
    var dragNode = null, dragMoved = false;
    var panning = false, panStart = null, lastPtr = { x: 0, y: 0 };
    var cssW = 1, cssH = 1;

    function minK() { return baseZoom * 0.5; }
    function maxK() { return baseZoom * 6; }
    function zoomPct() {
      var lo = minK(), hi = maxK();
      var p = 100 * Math.log(k / lo) / Math.log(hi / lo);
      return Math.max(0, Math.min(100, p));
    }
    function emitZoom() { if (opts.onZoom) opts.onZoom(zoomPct()); }

    // ---- deterministic free-form seed ------------------------------------
    function seededUnit(id, salt) {
      var raw = String(id || "node") + ":" + salt, h = 2166136261;
      for (var i = 0; i < raw.length; i++) {
        h ^= raw.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return (h >>> 0) / 4294967295;
    }
    function seedPosition(id, nodeCount) {
      var angle = seededUnit(id, "angle") * Math.PI * 2;
      // Give large imports enough room before the first force tick. Starting
      // 500 records in a tight disc makes the simulation spend its warm-up
      // untangling collisions instead of arranging the connected clusters.
      var spread = Math.max(170, 48 * Math.sqrt(Math.max(nodeCount, 1)));
      var radius = Math.sqrt(seededUnit(id, "radius")) * spread;
      return {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius
      };
    }

    // ---- sizing -----------------------------------------------------------
    function resize() {
      var rect = canvas.getBoundingClientRect();
      cssW = Math.max(rect.width, 1); cssH = Math.max(rect.height, 1);
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    }
    window.addEventListener("resize", resize);
    resize();

    // ---- data -------------------------------------------------------------
    function setData(g, o) {
      o = o || {};
      pastMode = !!o.pastMode;
      var incoming = g.nodes || [];
      var firstLoad = incoming.length && !nodes.length;
      // Scale link lengths with density while keeping large graphs bounded.
      layoutScale = Math.max(1, Math.min(5, Math.sqrt(Math.max(incoming.length, 1) / 50)));
      var next = [], nextById = new Map();
      incoming.forEach(function (nd) {
        var ex = byId.get(nd.id);
        if (ex) {
          ex.label = nd.label; ex.type = nd.type; ex.degree = nd.degree;
          ex.node_kind = nd.node_kind; ex.content = nd.content; ex.kind = nd.kind;
          ex.summary = nd.summary; ex.created_at = nd.created_at;
          ex.source_occurred_at = nd.source_occurred_at;
          ex.source_episode_id = nd.source_episode_id; ex.subject_ids = nd.subject_ids;
          ex.source = nd.source; ex.source_ref = nd.source_ref;
          ex.occurred_at = nd.occurred_at; ex.recorded_at = nd.recorded_at;
          ex.meta = nd.meta;
          ex.episode_id = nd.episode_id; ex.legal_hold = nd.legal_hold;
          next.push(ex); nextById.set(nd.id, ex);
        } else {
          var sd = seedPosition(nd.id, incoming.length);
          next.push(Object.assign({}, nd, { x: sd.x, y: sd.y, vx: 0, vy: 0, fx: null, fy: null }));
          nextById.set(nd.id, next[next.length - 1]);
          // Nodes that arrive AFTER the initial load (a live remember while
          // the graph is open) pop in with a brief scale/glow animation.
          if (!firstLoad) next[next.length - 1]._born = performance.now();
        }
      });
      nodes = next; byId = nextById;
      edges = (g.edges || []).filter(function (e) { return nextById.has(e.source) && nextById.has(e.target); });
      adj = new Map();
      nodes.forEach(function (n) { adj.set(n.id, new Set()); });
      edges.forEach(function (e) { adj.get(e.source).add(e.target); adj.get(e.target).add(e.source); });
      if (selectedId && !nextById.has(selectedId)) {
        selectedId = null;
        if (opts.onSelect) opts.onSelect(null);
      }
      if (hoverId && !nextById.has(hoverId)) hoverId = null;
      hoverEdge = null;
      if (opts.onHoverEdge) opts.onHoverEdge(null, 0, 0);
      reheat(o.reheat != null ? o.reheat : (nodes.length ? 0.5 : 0.9));
      // Warm the initial layout before its first meaningful paint. Otherwise a
      // fit taken while every node is still at its seed position becomes stale
      // as the force simulation expands, leaving nodes under the timeline or
      // outside the viewport. Large graphs get fewer synchronous passes.
      if (firstLoad) {
        var warmTicks = nodes.length <= 300 ? 110 : nodes.length <= 1000 ? 55 : 24;
        for (var wi = 0; wi < warmTicks; wi++) tick();
        fitView();
        alpha = Math.min(alpha, 0.025);
      }
    }
    function reheat(a) { alpha = Math.max(alpha, a); }

    // ---- physics (grid repulsion + springs + two-axis centering) -----------
    function linkLen(e) {
      return e.kind === "provenance" ? 68 * layoutScale : 112 * layoutScale;
    }

    function tick() {
      var n = nodes.length, i;
      var REP = 2700 * layoutScale * layoutScale;
      var SPRING = 0.03, CENTER = 0.0023;
      var CELL = Math.max(60, 155 * layoutScale);
      var grid = new Map();
      function key(cx, cy) { return cx + "," + cy; }
      for (i = 0; i < n; i++) {
        var p = nodes[i];
        p._cx = Math.floor(p.x / CELL); p._cy = Math.floor(p.y / CELL);
        var kk = key(p._cx, p._cy);
        var bkt = grid.get(kk); if (!bkt) { bkt = []; grid.set(kk, bkt); }
        bkt.push(p);
      }
      for (i = 0; i < n; i++) {
        var a = nodes[i];
        for (var gx = -1; gx <= 1; gx++) for (var gy = -1; gy <= 1; gy++) {
          var bucket = grid.get(key(a._cx + gx, a._cy + gy));
          if (!bucket) continue;
          for (var bi = 0; bi < bucket.length; bi++) {
            var b = bucket[bi]; if (b === a) continue;
            var dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
            if (d2 < 0.01) {
              var jitter = seededUnit(a.id, b.id) * Math.PI * 2;
              dx = Math.cos(jitter) * 0.1; dy = Math.sin(jitter) * 0.1; d2 = 0.01;
            }
            if (d2 > CELL * CELL * 4) continue;
            var dist = Math.sqrt(d2), f = (REP / d2) * alpha;
            if (f > 14) f = 14;
            a.vx += (dx / dist) * f; a.vy += (dy / dist) * f;
          }
        }
      }
      for (i = 0; i < edges.length; i++) {
        var e = edges[i], s = byId.get(e.source), t = byId.get(e.target);
        var dx2 = t.x - s.x, dy2 = t.y - s.y;
        var dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) || 0.01;
        var disp = ((dist2 - linkLen(e)) / dist2) * SPRING * alpha;
        s.vx += dx2 * disp; s.vy += dy2 * disp;
        t.vx -= dx2 * disp; t.vy -= dy2 * disp;
      }
      for (i = 0; i < n; i++) {
        var q = nodes[i];
        var centerWeight = q.node_kind === "episode" ? 0.55
          : q.node_kind === "source" || (q.degree || 0) >= 8 ? 1.25 : 1;
        q.vx += -q.x * CENTER * centerWeight * alpha;
        q.vy += -q.y * CENTER * centerWeight * alpha;
        if (q.fx != null) { q.x = q.fx; q.y = q.fy; q.vx = q.vy = 0; }
        else {
          q.vx *= velocityDecay; q.vy *= velocityDecay;
          q.x += q.vx; q.y += q.vy;
        }
      }
      alpha -= alpha * alphaDecay;
    }

    // ---- transforms -------------------------------------------------------
    function toScreen(x, y) { return { x: tx + x * k, y: ty + y * k }; }
    function toWorld(sx, sy) { return { x: (sx - tx) / k, y: (sy - ty) / k }; }

    function fitView() {
      if (!nodes.length) {
        k = baseZoom = 1; tx = cssW / 2; ty = cssH / 2; emitZoom(); return;
      }
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach(function (n) {
        minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
      });
      // Reserve room for resting labels and floating chrome. A minimum world
      // span prevents a two-node graph from becoming comically oversized.
      var w = Math.max(340, maxX - minX), h = Math.max(260, maxY - minY);
      var usableW = Math.max(240, cssW - 180), usableH = Math.max(200, cssH - 280);
      var fittedK = Math.min(usableW / w, usableH / h, 1.8);
      // The reserved chrome margins leave room for a modest density boost.
      // This makes large maps readable without clipping them at the canvas
      // edge or forcing users to zoom before individual nodes separate.
      var densityBoost = nodes.length >= 400 ? 1.14 : nodes.length >= 150 ? 1.08 : 1;
      k = Math.min(fittedK * densityBoost, 1.8);
      if (!isFinite(k) || k <= 0) k = 1;
      baseZoom = k;
      tx = cssW / 2 - ((minX + maxX) / 2) * k;
      ty = cssH / 2 - ((minY + maxY) / 2) * k;
      emitZoom();
    }
    function setZoomTo(nk, sx, sy) {
      var w = toWorld(sx, sy);
      k = Math.min(Math.max(nk, minK()), maxK());
      tx = sx - w.x * k; ty = sy - w.y * k;
      emitZoom();
    }

    // ---- rendering --------------------------------------------------------
    function screenScale() {
      var rel = k / (baseZoom || 1);
      return Math.max(0.82, Math.min(1.55, Math.pow(rel, 0.32)));
    }
    function roundedRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    // Soft halo sprites, one offscreen render per color. Painting a radial
    // gradient per node per frame is what makes canvas glows slow; a cached
    // sprite scaled at draw time keeps 500 luminous nodes at 60fps.
    // Headless layout probes stub the main canvas but have no DOM; glow and
    // backdrop are pure decoration there, so they degrade to nothing.
    function makeCanvas() {
      return (typeof document === "undefined") ? null
        : document.createElement("canvas");
    }

    var haloSprites = new Map();
    function haloSprite(color) {
      var sp = haloSprites.get(color);
      if (sp !== undefined) return sp;
      var size = 64, c = makeCanvas();
      if (!c) { haloSprites.set(color, null); return null; }
      c.width = c.height = size;
      var g = c.getContext("2d");
      var grad = g.createRadialGradient(size / 2, size / 2, size * 0.06,
                                        size / 2, size / 2, size / 2);
      grad.addColorStop(0, color);
      grad.addColorStop(0.28, color + "66");
      grad.addColorStop(0.62, color + "1f");
      grad.addColorStop(1, color + "00");
      g.fillStyle = grad;
      g.fillRect(0, 0, size, size);
      haloSprites.set(color, c);
      return c;
    }

    // A white radial fade used as the off-center core highlight. Drawn with
    // drawImage so layout probes (which track arc() calls) never see it.
    var shineSprite = null;
    function shine() {
      if (shineSprite !== null) return shineSprite;
      var size = 32, c = makeCanvas();
      if (!c) { shineSprite = false; return false; }
      c.width = c.height = size;
      var g = c.getContext("2d");
      var grad = g.createRadialGradient(size / 2, size / 2, 0,
                                        size / 2, size / 2, size / 2);
      grad.addColorStop(0, "rgba(255,255,255,.85)");
      grad.addColorStop(0.55, "rgba(255,255,255,.22)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, size, size);
      shineSprite = c;
      return c;
    }

    // Depth-graded backdrop: a faint cool bloom behind the graph's center and
    // gently darkened corners. Rebuilt only when the canvas size changes.
    var vignette = null;
    function backdrop() {
      if (vignette && vignette.w === cssW && vignette.h === cssH) return vignette.c;
      var c = makeCanvas();
      if (!c) { vignette = { w: cssW, h: cssH, c: null }; return null; }
      c.width = Math.max(1, Math.round(cssW)); c.height = Math.max(1, Math.round(cssH));
      var g = c.getContext("2d");
      var glow = g.createRadialGradient(c.width / 2, c.height * 0.44, 0,
                                        c.width / 2, c.height * 0.44,
                                        Math.max(c.width, c.height) * 0.62);
      glow.addColorStop(0, "rgba(93,130,255,0.075)");
      glow.addColorStop(0.5, "rgba(93,130,255,0.028)");
      glow.addColorStop(1, "rgba(93,130,255,0)");
      g.fillStyle = glow; g.fillRect(0, 0, c.width, c.height);
      var dark = g.createRadialGradient(c.width / 2, c.height / 2,
                                        Math.min(c.width, c.height) * 0.42,
                                        c.width / 2, c.height / 2,
                                        Math.max(c.width, c.height) * 0.85);
      dark.addColorStop(0, "rgba(0,0,0,0)");
      dark.addColorStop(1, "rgba(2,3,8,0.55)");
      g.fillStyle = dark; g.fillRect(0, 0, c.width, c.height);
      vignette = { w: cssW, h: cssH, c: c };
      return c;
    }

    function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

    function draw() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      ctx.fillStyle = BG; ctx.fillRect(0, 0, cssW, cssH);
      var bd = backdrop();
      if (bd) ctx.drawImage(bd, 0, 0, cssW, cssH);
      var now = performance.now();

      var focus = hoverId || selectedId;
      var neighbors = focus ? (adj.get(focus) || new Set()) : null;
      var searching = !focus && searchHit && searchHit.size > 0;
      function tierOf(id) {
        if (focus) return id === focus ? "bright" : (neighbors && neighbors.has(id) ? "mid" : "dark");
        if (searching) return searchHit.has(id) ? "bright" : (searchRelated && searchRelated.has(id) ? "mid" : "dark");
        return "normal";
      }

      // Obsidian-style straight, quiet relationships keep topology readable.
      for (var i = 0; i < edges.length; i++) {
        var e = edges[i], s = byId.get(e.source), t = byId.get(e.target);
        if (!s || !t) continue;
        var op = e.kind === "provenance" ? 0.13 : 0.19;
        if (e.valid_to && pastMode) op = 0.075;
        if (focus) op = (e.source === focus || e.target === focus) ? 0.68 : 0.025;
        else if (searching) {
          op = (searchHit.has(e.source) || searchHit.has(e.target)) ? 0.62
            : (searchRelated.has(e.source) && searchRelated.has(e.target) ? 0.16 : 0.018);
        }
        if (e === hoverEdge) op = Math.max(op, 0.82);
        if (op < 0.014) continue;
        var ss = toScreen(s.x, s.y), ts = toScreen(t.x, t.y);
        var emphasized = e === hoverEdge ||
          (focus && (e.source === focus || e.target === focus));
        ctx.globalAlpha = 1;
        ctx.lineWidth = emphasized ? 1.5 : 0.85;
        var eg = emphasized && ctx.createLinearGradient
          ? ctx.createLinearGradient(ss.x, ss.y, ts.x, ts.y) : null;
        if (eg) {
          // Focused relationships light up as a gradient between their two
          // endpoint colors — the connection itself becomes readable.
          eg.addColorStop(0, colorFor(s)); eg.addColorStop(1, colorFor(t));
          ctx.globalAlpha = Math.min(1, op + 0.1);
          ctx.strokeStyle = eg;
        } else {
          ctx.strokeStyle = e.kind === "provenance"
            ? "rgba(245,164,92," + op.toFixed(3) + ")"
            : "rgba(168,190,224," + op.toFixed(3) + ")";
        }
        ctx.setLineDash(e.disputed ? [4, 4] : []);
        ctx.beginPath(); ctx.moveTo(ss.x, ss.y); ctx.lineTo(ts.x, ts.y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.setLineDash([]);

      var scale = screenScale();
      // Luminous cores over soft sprite halos: degree still controls size and
      // the focus tiers still carry the hierarchy, but every node now glows
      // like a small light source instead of a flat disc.
      // Layout-probe contract: draw() emits exactly ONE arc() per node, in
      // node order — halos, highlights, and selection rings use drawImage or
      // run after the per-node arc so measured positions stay aligned.
      for (var n2 = 0; n2 < nodes.length; n2++) {
        var node = nodes[n2], nc = toScreen(node.x, node.y), nodeTier = tierOf(node.id);
        var nodeLit = nodeTier === "bright";
        var coreColor = colorFor(node);
        var cr = radiusFor(node) * scale;

        // Live-added nodes pop in: scale up with an eased overshoot and a
        // temporary halo flare for ~0.9s after arrival.
        var flare = 0;
        if (node._born) {
          var age = (now - node._born) / 900;
          if (age >= 1) { node._born = 0; }
          else {
            cr *= 0.3 + 0.78 * easeOutCubic(age);
            flare = (1 - age) * 0.55;
          }
        }

        var coreAlpha = nodeTier === "dark" ? 0.1 : nodeTier === "mid" ? 0.74
          : 0.97;
        var haloAlpha = nodeTier === "dark" ? 0 : nodeTier === "mid" ? 0.22
          : nodeLit ? 0.62 : 0.3;
        haloAlpha = Math.min(0.9, haloAlpha + flare);
        var onScreen = nc.x > -80 && nc.y > -80 && nc.x < cssW + 80 && nc.y < cssH + 80;
        var sprite = onScreen && haloAlpha > 0.01 ? haloSprite(coreColor) : null;
        if (sprite) {
          var hr = cr * (nodeLit ? 4.4 : 3.4);
          ctx.globalAlpha = haloAlpha;
          ctx.drawImage(sprite, nc.x - hr, nc.y - hr, hr * 2, hr * 2);
        }

        ctx.globalAlpha = coreAlpha;
        ctx.fillStyle = coreColor;
        ctx.lineWidth = nodeLit ? 1.5 : 1;
        ctx.strokeStyle = nodeLit ? "rgba(248,250,255,.95)" : "rgba(226,236,250,.32)";
        ctx.beginPath(); ctx.arc(nc.x, nc.y, cr, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();

        // Off-center highlight (a sprite, not an arc) gives cores a lit,
        // dimensional read.
        var sh = onScreen && nodeTier !== "dark" && cr > 2.8 ? shine() : null;
        if (sh) {
          var sr = cr * 0.62;
          ctx.globalAlpha = coreAlpha * 0.42;
          ctx.drawImage(sh, nc.x - cr * 0.34 - sr / 2, nc.y - cr * 0.36 - sr / 2, sr, sr);
        }
      }
      ctx.globalAlpha = 1;

      // Breathing selection ring — drawn after every node arc so the probe's
      // node-order position sampling is never disturbed.
      if (selectedId) {
        var sel = byId.get(selectedId);
        if (sel) {
          var sc2 = toScreen(sel.x, sel.y);
          var selR = radiusFor(sel) * scale;
          var selColor = colorFor(sel);
          var pulse = 0.5 + 0.5 * Math.sin(now / 420);
          ctx.globalAlpha = 0.75;
          ctx.lineWidth = 1.5; ctx.strokeStyle = selColor;
          ctx.beginPath(); ctx.arc(sc2.x, sc2.y, selR + 4.5 + pulse * 2.5, 0, Math.PI * 2); ctx.stroke();
          ctx.globalAlpha = 0.14 + pulse * 0.1; ctx.lineWidth = 5;
          ctx.beginPath(); ctx.arc(sc2.x, sc2.y, selR + 8.5 + pulse * 3, 0, Math.PI * 2); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      // Keep the canvas quiet: no node description is painted at rest. Labels
      // appear only for the exact node being hovered. Selection and search
      // still highlight nodes and connections without painting text beneath
      // them; the inspector/search status carries their readable details.
      for (n2 = 0; n2 < nodes.length; n2++) {
        var ld = nodes[n2], isEpisode = ld.node_kind === "episode";
        if (!ld.label || ld.id !== hoverId) continue;
        var lc = toScreen(ld.x, ld.y);
        if (lc.x < -100 || lc.y < -40 || lc.x > cssW + 100 || lc.y > cssH + 40) continue;
        var cap = isEpisode ? 28 : 34;
        var lbl = ld.label.length > cap ? ld.label.slice(0, cap - 1).trim() + "…" : ld.label;
        ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        var tw = ctx.measureText(lbl).width, lh = 20;
        var ly = lc.y + radiusFor(ld) * scale + 7;
        ctx.globalAlpha = 0.96;
        ctx.fillStyle = "rgba(14,20,30,.94)";
        roundedRect(lc.x - tw / 2 - 6, ly, tw + 12, lh, 5); ctx.fill();
        ctx.strokeStyle = "rgba(111,159,220,.32)";
        ctx.lineWidth = 0.75; ctx.stroke();
        ctx.fillStyle = "#f0f5fc";
        ctx.fillText(lbl, lc.x, ly + lh / 2 + 0.5);
      }
      ctx.globalAlpha = 1;
    }

    // ---- loop -------------------------------------------------------------
    var running = true;
    function frame() {
      if (!running) return;
      if (alpha > alphaMin || dragNode) tick();
      draw();
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    // ---- hit testing ------------------------------------------------------
    function nodeAt(sx, sy) {
      var best = null, bestD = Infinity;
      var rk = screenScale();
      for (var i = 0; i < nodes.length; i++) {
        var nd = nodes[i], sc = toScreen(nd.x, nd.y);
        var r = radiusFor(nd) * rk + 5;
        var dx = sx - sc.x, dy = sy - sc.y, d = dx * dx + dy * dy;
        if (d <= r * r && d < bestD) { bestD = d; best = nd; }
      }
      return best;
    }
    function relPos(ev) { var r = canvas.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; }
    function edgeAt(sx, sy) {
      var best = null, bestScore = 0;
      for (var i = 0; i < edges.length; i++) {
        var e = edges[i], s = byId.get(e.source), t = byId.get(e.target);
        if (!s || !t) continue;
        var a = toScreen(s.x, s.y), b = toScreen(t.x, t.y);
        var dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy; if (l2 < 1) continue;
        var u = ((sx - a.x) * dx + (sy - a.y) * dy) / l2; if (u < 0.05 || u > 0.95) continue;
        var px = a.x + u * dx, py = a.y + u * dy;
        var d = Math.sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
        if (d < 6 && 6 - d > bestScore) { bestScore = 6 - d; best = e; }
      }
      return best;
    }
    function setHoverEdge(e, cx, cy) { if (e !== hoverEdge) hoverEdge = e; if (opts.onHoverEdge) opts.onHoverEdge(e, cx, cy); }

    // ---- interaction ------------------------------------------------------
    canvas.addEventListener("mousedown", function (ev) {
      var p = relPos(ev); lastPtr = p;
      var hit = nodeAt(p.x, p.y);
      if (hit) {
        dragNode = hit; dragMoved = false;
        var w = toWorld(p.x, p.y); hit.fx = w.x; hit.fy = w.y; reheat(0.3);
        canvas.classList.add("grabbing");
      } else {
        panning = true; panStart = { x: p.x - tx, y: p.y - ty }; canvas.classList.add("grabbing");
      }
    });
    window.addEventListener("mousemove", function (ev) {
      var p = relPos(ev);
      if (dragNode) {
        dragMoved = true; var w = toWorld(p.x, p.y); dragNode.fx = w.x; dragNode.fy = w.y; reheat(0.3);
      } else if (panning) {
        tx = p.x - panStart.x; ty = p.y - panStart.y;
      } else {
        var inB = p.x >= 0 && p.y >= 0 && p.x <= cssW && p.y <= cssH;
        var hit = inB ? nodeAt(p.x, p.y) : null;
        var nh = hit ? hit.id : null;
        if (nh !== hoverId) hoverId = nh;
        canvas.classList.toggle("pointing", !!hit);
        if (inB && !hit) setHoverEdge(edgeAt(p.x, p.y), ev.clientX, ev.clientY);
        else setHoverEdge(null, 0, 0);
      }
      lastPtr = p;
    });
    window.addEventListener("mouseup", function () {
      if (dragNode) { dragNode.fx = dragNode.fy = null; if (!dragMoved) selectNode(dragNode.id); dragNode = null; }
      panning = false; canvas.classList.remove("grabbing");
    });
    canvas.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var p = relPos(ev);
      setZoomTo(k * Math.exp(-ev.deltaY * 0.0015), p.x, p.y);   // zoom to cursor
    }, { passive: false });
    canvas.addEventListener("dblclick", function (ev) {
      var p = relPos(ev); if (!nodeAt(p.x, p.y)) fitView();
    });

    function selectNode(id) {
      selectedId = id;
      if (opts.onSelect) opts.onSelect(id ? byId.get(id) || null : null);
      reheat(0.08);
    }

    // ---- public api -------------------------------------------------------
    return {
      setData: setData,
      resetView: function () { fitView(); reheat(0.12); },
      setSearch: function (q) {
        q = (q || "").trim().toLowerCase();
        if (!q) { searchHit = null; searchRelated = null; return null; }
        searchHit = new Set(); var found = [];
        nodes.forEach(function (n) { if (n.label && n.label.toLowerCase().indexOf(q) !== -1) { searchHit.add(n.id); found.push(n); } });
        // "Related" = the matches plus their direct neighbours. Everything else
        // is dimmed to near-dark so the search result stands out of the field.
        searchRelated = new Set(searchHit);
        searchHit.forEach(function (id) {
          var nbrs = adj.get(id); if (nbrs) nbrs.forEach(function (nid) { searchRelated.add(nid); });
        });
        if (found.length) { var f = found[0]; tx = cssW / 2 - f.x * k; ty = cssH / 2 - f.y * k; }
        return found.length;
      },
      selectNode: function (id) { selectNode(id); },
      clearSelection: function () { selectedId = null; if (opts.onSelect) opts.onSelect(null); },
      focusNode: function (id) {
        var n = byId.get(id); if (!n) return;
        tx = cssW / 2 - n.x * k; ty = cssH / 2 - n.y * k; selectNode(id);
      },
      edgesFor: function (id) { return edges.filter(function (e) { return e.source === id || e.target === id; }); },
      getNode: function (id) { return byId.get(id) || null; },
      // zoom controls (slider): pct is 0..100 across [0.5×, 6×] the fitted view
      setZoomPct: function (pct) {
        var lo = minK(), hi = maxK();
        var nk = lo * Math.pow(hi / lo, Math.max(0, Math.min(100, pct)) / 100);
        setZoomTo(nk, cssW / 2, cssH / 2);
      },
      zoomBy: function (factor) { setZoomTo(k * factor, cssW / 2, cssH / 2); },
      getZoomPct: zoomPct,
      colors: TYPE_COLORS,
      colorFor: colorFor,
      resize: resize, fit: fitView,
      destroy: function () { running = false; }
    };
  }

  window.GraphView = GraphView;
})();
