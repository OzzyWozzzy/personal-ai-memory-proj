(function () {
  "use strict";

  var workspaceReady = false;
  var graph = null;
  var graphPayload = { nodes: [], edges: [], counts: {} };
  var toastTimer = null;

  function $(id) { return document.getElementById(id); }
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }
  function formatDate(value) {
    if (!value) return "timeless";
    var date = new Date(value);
    return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString();
  }
  function setStatus(message, kind) {
    $("status-text").textContent = message;
    $("status-dot").className = kind || "";
  }
  function toast(message) {
    $("toast").textContent = message;
    $("toast").classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { $("toast").classList.remove("show"); }, 2800);
  }
  function showResult(target, value, kind) {
    target.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    target.className = "result compact " + (kind || "");
  }

  async function api(path, options) {
    options = options || {};
    var headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    var response = await fetch(path, Object.assign({}, options, {
      headers: headers,
      credentials: "same-origin"
    }));
    var data = null;
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) {
      var detail = data.detail;
      if (Array.isArray(detail)) detail = detail.map(function (item) { return item.msg; }).join("; ");
      throw new Error(detail || ("Request failed: " + response.status));
    }
    return data;
  }

  async function startWorkspace() {
    try {
      var session = await api("/v1/auth/preview", { method: "POST" });
      workspaceReady = true;
      setStatus("Local workspace · " + session.workspace.id.slice(0, 8), "ready");
      await Promise.all([loadGraph(), loadGovernance()]);
    } catch (error) {
      setStatus(error.message, "error");
      toast("Could not start local workspace: " + error.message);
    }
  }

  function graphQuery() {
    var query = new URLSearchParams();
    query.set("include_historical", $("historical").checked ? "true" : "false");
    query.set("include_episodes", $("episodes").checked ? "true" : "false");
    var asOf = $("as-of").value;
    if (asOf) query.set("as_of", new Date(asOf).toISOString());
    return query.toString();
  }

  async function loadGraph() {
    if (!workspaceReady) return;
    try {
      graphPayload = await api("/v1/graph?" + graphQuery());
      graph.setData(graphPayload, { pastMode: $("historical").checked });
      var counts = graphPayload.counts || {};
      $("entity-count").textContent = counts.entities || 0;
      $("edge-count").textContent = counts.edges || 0;
      $("episode-count").textContent = counts.episodes_total || 0;
      $("empty-graph").classList.toggle("show", !(counts.nodes || 0));
    } catch (error) {
      toast("Graph refresh failed: " + error.message);
    }
  }

  function addDetail(container, text, when) {
    var item = el("div", "detail", text || "Untitled relationship");
    if (when) item.appendChild(el("time", "", when));
    container.appendChild(item);
  }

  function inspectNode(node) {
    var panel = $("inspector");
    panel.replaceChildren();
    panel.className = "inspector";
    if (!node) {
      panel.className = "inspector empty-copy";
      panel.textContent = "Select a node to inspect its facts, validity windows, and provenance.";
      return;
    }
    panel.appendChild(el("div", "kind", node.node_kind || node.type || "entity"));
    panel.appendChild(el("h2", "", node.label || "Untitled"));
    if (node.summary) panel.appendChild(el("p", "muted", node.summary));
    if (node.content) panel.appendChild(el("p", "muted", node.content));

    var details = el("div", "detail-list");
    graph.edgesFor(node.id).forEach(function (edge) {
      var windowText = edge.valid_from
        ? formatDate(edge.valid_from) + (edge.valid_to ? " → " + formatDate(edge.valid_to) : " → present")
        : "";
      addDetail(details, edge.fact || edge.relation, windowText);
    });
    panel.appendChild(details);

    if (node.node_kind === "entity") {
      var story = el("button", "", "View chronological story");
      story.addEventListener("click", async function () {
        try {
          var response = await api("/v1/subjects/" + encodeURIComponent(node.id) + "/story");
          details.replaceChildren();
          response.story.forEach(function (entry) { addDetail(details, entry.text, entry.when); });
        } catch (error) { toast(error.message); }
      });
      panel.appendChild(story);
    }
    if (node.node_kind === "episode") {
      var hold = el("button", "", node.legal_hold ? "Release legal hold" : "Place legal hold");
      hold.addEventListener("click", async function () {
        try {
          await api("/v1/admin/episodes/" + node.episode_id + "/legal_hold", {
            method: "POST", body: JSON.stringify({ held: !node.legal_hold })
          });
          toast("Legal hold updated");
          await loadGraph();
        } catch (error) { toast(error.message); }
      });
      var evict = el("button", "danger", "Preview eviction");
      evict.addEventListener("click", async function () {
        try {
          var plan = await api("/v1/admin/episodes/" + node.episode_id + "/evict?dry_run=true", { method: "POST" });
          details.replaceChildren();
          addDetail(details, "Dry-run deletion plan", JSON.stringify(plan.plan));
        } catch (error) { toast(error.message); }
      });
      panel.appendChild(hold);
      panel.appendChild(document.createTextNode(" "));
      panel.appendChild(evict);
    }
  }

  async function remember(event) {
    event.preventDefault();
    var text = $("memory-text").value.trim();
    if (!text) return;
    var button = event.currentTarget.querySelector("button");
    button.disabled = true;
    showResult($("remember-result"), "Extracting temporal facts…");
    try {
      var body = { text: text };
      if ($("occurred-at").value.trim()) body.occurred_at = $("occurred-at").value.trim();
      var response = await api("/v1/remember", { method: "POST", body: JSON.stringify(body) });
      var lines = [];
      (response.understood || []).forEach(function (item) { lines.push("Added · " + item); });
      (response.replaced || []).forEach(function (item) { lines.push("Closed · " + item); });
      (response.remembered || []).forEach(function () { lines.push("Source stored · no graph assertion extracted"); });
      if (!lines.length && response.details.deduped) lines.push("Already present · deduplicated");
      showResult($("remember-result"), lines.join("\n") || "Episode stored", "success");
      $("memory-text").value = "";
      await Promise.all([loadGraph(), loadGovernance()]);
    } catch (error) {
      showResult($("remember-result"), error.message, "error");
    } finally { button.disabled = false; }
  }

  async function recall(event) {
    event.preventDefault();
    var query = $("recall-query").value.trim();
    if (!query) return;
    $("recall-result").textContent = "Retrieving and reranking…";
    try {
      var body = {
        query: query,
        token_budget: Number($("token-budget").value || 1200),
        include_historical: $("recall-history").checked,
        format: "block"
      };
      if ($("recall-as-of").value) body.as_of = new Date($("recall-as-of").value).toISOString();
      var response = await api("/v1/context", { method: "POST", body: JSON.stringify(body) });
      $("recall-result").textContent = response.block || JSON.stringify(response.items || response, null, 2);
      var meta = $("recall-meta");
      meta.replaceChildren();
      [
        ["route", response.route],
        ["tokens", response.tokens || response.tokens_estimated],
        ["items", (response.items || []).length]
      ].forEach(function (pair) {
        if (pair[1] != null) meta.appendChild(el("span", "metric-pill", pair[0] + " · " + pair[1]));
      });
      loadGovernance();
    } catch (error) { $("recall-result").textContent = error.message; }
  }

  async function loadGovernance() {
    if (!workspaceReady) return;
    try {
      var results = await Promise.all([
        api("/v1/stats"), api("/v1/admin/token-metrics"), api("/v1/admin/retention")
      ]);
      var stats = results[0];
      var cards = $("stats-cards");
      cards.replaceChildren();
      [["episodes", stats.episodes], ["entities", stats.entities], ["facts", stats.facts], ["current", stats.facts_current]].forEach(function (pair) {
        var card = el("div", "stat");
        card.appendChild(el("b", "", String(pair[1] || 0)));
        card.appendChild(el("span", "", pair[0]));
        cards.appendChild(card);
      });
      var metrics = results[1];
      showResult($("token-metrics"), "Queries " + metrics.queries + " · mean " + metrics.avg_tokens + " · p50 " + metrics.p50_tokens + " · p95 " + metrics.p95_tokens);
      $("retention-json").value = JSON.stringify(results[2].retention || {}, null, 2);
    } catch (error) { toast("Governance refresh failed: " + error.message); }
  }

  async function saveRetention() {
    try {
      var policy = JSON.parse($("retention-json").value);
      var response = await api("/v1/admin/retention", { method: "PUT", body: JSON.stringify({ retention: policy }) });
      showResult($("retention-result"), response, "success");
    } catch (error) { showResult($("retention-result"), error.message, "error"); }
  }
  async function planSweep() {
    try {
      showResult($("retention-result"), await api("/v1/admin/retention/sweep?dry_run=true", { method: "POST" }));
    } catch (error) { showResult($("retention-result"), error.message, "error"); }
  }
  async function clearWorkspace(apply) {
    if (apply && !window.confirm("Delete every episode not protected by legal hold?")) return;
    try {
      var result = await api("/v1/admin/clear?dry_run=" + (apply ? "false" : "true"), { method: "POST" });
      showResult($("clear-result"), result, apply ? "success" : "");
      if (apply) await Promise.all([loadGraph(), loadGovernance()]);
    } catch (error) { showResult($("clear-result"), error.message, "error"); }
  }

  function selectView(name) {
    document.querySelectorAll(".tab").forEach(function (tab) { tab.classList.toggle("active", tab.dataset.view === name); });
    document.querySelectorAll(".view").forEach(function (view) { view.classList.toggle("active", view.id === "view-" + name); });
    if (name === "graph") { graph.resize(); graph.fit(); }
    if (name === "governance") loadGovernance();
  }

  function init() {
    graph = window.GraphView($("graph-canvas"), { onSelect: inspectNode });
    document.querySelectorAll(".tab").forEach(function (tab) {
      tab.addEventListener("click", function () { selectView(tab.dataset.view); });
    });
    $("remember-form").addEventListener("submit", remember);
    $("recall-form").addEventListener("submit", recall);
    $("refresh-graph").addEventListener("click", loadGraph);
    $("fit-graph").addEventListener("click", function () { graph.fit(); });
    $("historical").addEventListener("change", loadGraph);
    $("episodes").addEventListener("change", loadGraph);
    $("as-of").addEventListener("change", loadGraph);
    $("graph-search").addEventListener("input", function (event) { graph.setSearch(event.target.value); });
    $("save-retention").addEventListener("click", saveRetention);
    $("plan-sweep").addEventListener("click", planSweep);
    $("plan-clear").addEventListener("click", function () { clearWorkspace(false); });
    $("apply-clear").addEventListener("click", function () { clearWorkspace(true); });
    startWorkspace();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
