/* SMSA front-end: talks to the Flask REST API with fetch(). */
const SMSA = (() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const LABELS = ["Positive", "Negative", "Neutral"];
  const TYPE_NAMES = {
    news: "Financial news", announcement: "Company announcement", analyst: "Analyst commentary",
    investor: "Investor comment", social: "Social-media comment", other: "Other",
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
  const badge = (s) => `<span class="badge ${esc((s || "").toLowerCase())}">${esc(s)}</span>`;
  const engineLabel = (r) => (r.engine === "groq" ? "Groq · gpt-oss-120b"
    : r.engine === "gemini" ? `Gemini${r.fallback_from ? " (fallback)" : ""}`
    : r.engine === "llm" ? `Qwen · ${r.prompt_version}` : "FinBERT");
  const trunc = (s, n = 110) => (s && s.length > n ? `${s.slice(0, n)}…` : s || "");

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" }, ...options,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || body.status === "error") throw new Error(body.error || `HTTP ${res.status}`);
    return body;
  }

  /* ---------------------------------------------------------- animation helpers */
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Resolves after the next paint - or after 60 ms if the browser is not painting (background tab).
  const nextFrame = () => new Promise((r) => {
    let done = false;
    const go = () => { if (!done) { done = true; r(); } };
    requestAnimationFrame(() => requestAnimationFrame(go));
    setTimeout(go, 60);
  });

  // Fade/slide sections in as they scroll into view.
  function initReveal() {
    const els = document.querySelectorAll(".reveal:not(.in)");
    if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    }), { threshold: 0.08 });
    els.forEach((e, i) => { e.style.transitionDelay = `${Math.min(i, 4) * 70}ms`; io.observe(e); });
    // Safety net: whatever is on screen after 1.2 s is shown even if the observer has not fired.
    setTimeout(() => els.forEach((e) => {
      if (e.getBoundingClientRect().top < window.innerHeight) e.classList.add("in");
    }), 1200);
  }

  // Animated number: countUp(el, 0.873, v => pct(v))
  function countUp(el, to, fmt = (v) => Math.round(v).toLocaleString(), ms = 1100) {
    if (!el) return;
    if (reduceMotion || !isFinite(to)) { el.textContent = fmt(to); return; }
    const t0 = performance.now();
    let finished = false;
    const step = () => {
      if (finished) return;
      const k = Math.min(1, (performance.now() - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(to * e);
      if (k < 1) requestAnimationFrame(step); else finished = true;
    };
    requestAnimationFrame(step);
    setTimeout(() => { finished = true; el.textContent = fmt(to); }, ms + 150);   // always end on the exact value
  }

  // Bars are rendered with data-w="NN" and width 0, then grow to their value.
  async function animateBars(root = document) {
    await nextFrame();
    root.querySelectorAll(".fill[data-w]").forEach((f) => { f.style.width = `${f.dataset.w}%`; });
  }

  // Ticker tape in the header: the latest sentiments stored in the database.
  async function loadTicker() {
    const track = $("#ticker");
    if (!track) return;
    try {
      const { results } = await api("/api/results?limit=24");
      if (!results.length) { track.innerHTML = `<span class="tick">No analyses yet — try the analyser!</span>`; return; }
      const arrow = { Positive: "▲", Negative: "▼", Neutral: "●" };
      const items = results.map((r) => {
        const c = r.sentiment.toLowerCase();
        const sc = (r.sentiment_score >= 0 ? "+" : "") + r.sentiment_score.toFixed(2);
        return `<span class="tick"><b>${esc(r.company_name || r.detected_company || "Unknown")}</b>
          <span class="${c}">${arrow[r.sentiment]} ${r.sentiment}</span><span class="${c}">${sc}</span></span>`;
      }).join("");
      track.innerHTML = items + items;          // duplicated for a seamless loop
    } catch { track.innerHTML = ""; }
  }

  // Decorative candlestick chart in the hero (random walk with an upward drift).
  function buildHeroChart() {
    const svg = $("#hero-chart");
    if (!svg) return;
    const W = 420, H = 220, n = 26, pad = 10, cw = (W - pad * 2) / n;
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    let price = 60;
    const candles = [];
    for (let i = 0; i < n; i++) {
      const open = price;
      const close = open + (rnd() - 0.38) * 9;
      const high = Math.max(open, close) + rnd() * 5, low = Math.min(open, close) - rnd() * 5;
      candles.push({ open, close, high, low });
      price = close;
    }
    const lo = Math.min(...candles.map((c) => c.low)), hi = Math.max(...candles.map((c) => c.high));
    const y = (v) => H - 14 - ((v - lo) / (hi - lo)) * (H - 34);
    let out = `<defs>
      <linearGradient id="trend-grad" x1="0" x2="1"><stop offset="0" stop-color="#8b5cf6"/><stop offset="1" stop-color="#22d3ee"/></linearGradient>
      <linearGradient id="area-grad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#5b8cff" stop-opacity=".35"/><stop offset="1" stop-color="#5b8cff" stop-opacity="0"/></linearGradient>
    </defs>`;
    for (let g = 1; g < 5; g++) out += `<line class="gl" x1="0" x2="${W}" y1="${(H / 5) * g}" y2="${(H / 5) * g}"/>`;
    const pts = [];
    candles.forEach((c, i) => {
      const x = pad + i * cw + cw / 2, cls = c.close >= c.open ? "up" : "down";
      const top = y(Math.max(c.open, c.close)), bot = y(Math.min(c.open, c.close));
      out += `<g class="candle ${cls}" style="animation-delay:${i * 45}ms">
        <line class="wick" x1="${x}" x2="${x}" y1="${y(c.high)}" y2="${y(c.low)}"/>
        <rect x="${x - cw * 0.3}" y="${top}" width="${cw * 0.6}" height="${Math.max(2, bot - top)}" rx="1.5"/></g>`;
      pts.push([x, y((c.open + c.close) / 2) - 6]);
    });
    const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
    const last = pts[pts.length - 1];
    out += `<path class="area" d="${line} L${last[0]} ${H} L${pts[0][0]} ${H} Z"/>`;
    out += `<path class="trend" d="${line}"/>`;
    out += `<circle class="head-ring" cx="${last[0]}" cy="${last[1]}" r="5"/><circle class="head-dot" cx="${last[0]}" cy="${last[1]}" r="4.5"/>`;
    svg.innerHTML = out;
  }

  // Needle of the sentiment gauge: score -1 ... +1 -> -90deg ... +90deg.
  async function setGauge(score) {
    const needle = $("#g-needle");
    if (!needle) return;
    needle.style.transition = "none";
    needle.style.transform = "rotate(-90deg)";
    await nextFrame();
    needle.style.transition = "";
    needle.style.transform = `rotate(${Math.max(-1, Math.min(1, score)) * 90}deg)`;
  }

  /* ---------------------------------------------------------- model status pill */
  const ENGINE_NAMES = { groq: "Groq", gemini: "Gemini", llm: "Qwen", finbert: "FinBERT" };
  const usable = (state) => state === "ready" || state === "unchecked";

  // Engine states from /api/health: cloud engines -> ready | unchecked | no API key | unavailable: <reason>;
  // local engines -> not loaded | loading | ready | error.
  function engineStates(h) {
    return { groq: h.groq, gemini: h.gemini, llm: h.models.llm, finbert: h.models.finbert };
  }

  let onStatus = null;          // page-specific hook (the Analyse page updates its model list)

  async function pollStatus() {
    const pill = $("#model-status");
    if (!pill) return;
    try {
      const h = await api("/api/health");
      const st = engineStates(h);
      const mark = (e) => {
        const v = st[e];
        if (v === "ready") return "✓";
        if (v === "loading" || v === "unchecked" || v === "not loaded") return "…";
        return "✗";
      };
      pill.textContent = Object.keys(ENGINE_NAMES)
        .filter((e) => st[e] !== "no API key")
        .map((e) => `${ENGINE_NAMES[e]} ${mark(e)}`).join(" · ");
      pill.title = Object.entries(st).map(([e, v]) => `${ENGINE_NAMES[e]}: ${v}`).join("\n");
      const anyReady = ["groq", "gemini", "llm"].some((e) => st[e] === "ready");
      pill.className = `status-pill ${anyReady ? "ok" : "wait"}`;
      if (onStatus) onStatus(st);
      if (Object.values(st).some((v) => v === "loading" || v === "unchecked")) setTimeout(pollStatus, 3000);
    } catch {
      pill.textContent = "server offline";
      pill.className = "status-pill bad";
    }
  }

  // Readable text for the fallback notice, e.g. "Groq could not be used (rate limit reached)."
  function fallbackText(r) {
    const parts = String(r.fallback_from).split("; ").map((f) => {
      const i = f.indexOf(": ");
      const name = ENGINE_NAMES[f.slice(0, i)] || f.slice(0, i);
      let reason = f.slice(i + 2);
      const inner = reason.match(/'message': '([^']+)'/);       // older records stored raw SDK errors
      if (inner) reason = inner[1];
      reason = reason.replace(/[.\s]+$/, "");
      return `${name} could not be used (${reason})`;
    });
    return `${parts.join(". ")}. This answer was produced by the fallback model ${r.model_name}.`;
  }

  function probBars(p) {
    return LABELS.map((l) => `
      <div class="prob">
        <span>${l}</span>
        <div class="bar"><div class="fill ${l.toLowerCase()}" style="width:0" data-w="${(p[l] || 0) * 100}"></div></div>
        <strong>${pct(p[l])}</strong>
      </div>`).join("");
  }

  function resultsTable(rows, { withDelete = false } = {}) {
    if (!rows.length) return `<p class="muted">No results stored yet.</p>`;
    return `<table>
      <thead><tr><th>#</th><th>Time</th><th>Company</th><th>Type</th><th>Text</th>
        <th>Sentiment</th><th>Conf.</th><th>Model</th>${withDelete ? "<th></th>" : ""}</tr></thead>
      <tbody>${rows.map((r) => `
        <tr data-id="${r.result_id}">
          <td>${r.result_id}</td>
          <td class="nowrap small">${esc(r.submitted_at)}</td>
          <td>${esc(r.company_name || r.detected_company || "—")}</td>
          <td class="small">${esc(TYPE_NAMES[r.text_type] || r.text_type)}</td>
          <td class="text-cell">${esc(trunc(r.input_text))}</td>
          <td>${badge(r.sentiment)}</td>
          <td>${pct(r.confidence)}</td>
          <td class="small">${esc(engineLabel(r))}</td>
          ${withDelete ? `<td><button class="btn tiny danger" data-del="${r.result_id}">Delete</button></td>` : ""}
        </tr>`).join("")}</tbody></table>`;
  }

  /* ---------------------------------------------------------- analyse page */
  function initAnalyzePage() {
    const form = $("#analyze-form"), text = $("#text"), err = $("#form-error");
    const engineSel = $("#engine"), promptSel = $("#prompt_version");

    // Grey out engines whose API key failed the start-up check, with the reason as a tooltip.
    for (const o of engineSel.options) o.dataset.label = o.textContent;
    onStatus = (st) => {
      for (const o of engineSel.options) {
        const state = st[o.value];
        const off = state !== undefined && !usable(state) && state !== "not loaded" && state !== "loading";
        o.disabled = off;
        o.textContent = off ? `${o.dataset.label} — unavailable` : o.dataset.label;
        o.title = off ? state : "";
      }
      if (engineSel.selectedOptions[0]?.disabled) {
        const first = [...engineSel.options].find((o) => !o.disabled);
        if (first) { engineSel.value = first.value; syncPrompt(); }
      }
    };
    pollStatus();
    buildHeroChart();
    api("/api/stats").then(({ stats }) => countUp($("#hs-total"), stats.total_analyses)).catch(() => {});
    api("/api/test-runs").then(({ runs }) => {
      const best = runs.find((r) => r.engine === "groq" && r.total_items === 60 && r.finished_at)
        || runs.find((r) => r.total_items === 60 && r.finished_at);
      if (best) countUp($("#hs-acc"), best.accuracy * 100, (v) => `${v.toFixed(1)}%`);
    }).catch(() => {});
    let samples = null;

    const syncPrompt = () => {
      promptSel.disabled = engineSel.value !== "llm";
      if (["groq", "gemini"].includes(engineSel.value)) promptSel.value = "fewshot-v2";
    };
    engineSel.addEventListener("change", syncPrompt);
    syncPrompt();
    text.addEventListener("input", () => { $("#char-count").textContent = `${text.value.length} / 4000`; });

    $("#sample-btn").addEventListener("click", async () => {
      if (!samples) samples = (await api("/api/testset")).items;
      const s = samples[Math.floor(Math.random() * samples.length)];
      text.value = s.input_text;
      $("#text_type").value = s.text_type;
      $("#company").value = "";
      text.dispatchEvent(new Event("input"));
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      err.hidden = true;
      if (text.value.trim().length < 10) {
        err.textContent = "Please enter a financial text of at least 10 characters.";
        err.hidden = false;
        return;
      }
      const payload = {
        text: text.value, company: $("#company").value, text_type: $("#text_type").value,
        engine: engineSel.value, prompt_version: promptSel.value, source: "web",
      };
      $("#submit-btn").disabled = true;
      $("#result-empty").hidden = true; $("#result").hidden = true; $("#result-loading").hidden = false;
      const steps = [...document.querySelectorAll(".steps li")];
      const mark = (k) => steps.forEach((li, i) => {
        li.classList.toggle("done", i < k); li.classList.toggle("active", i === k);
      });
      mark(0);
      const t1 = setTimeout(() => mark(1), 450);
      try {
        const { result: r } = await api("/api/analyze", { method: "POST", body: JSON.stringify(payload) });
        clearTimeout(t1);
        mark(2); await new Promise((ok) => setTimeout(ok, 260));
        mark(3); await new Promise((ok) => setTimeout(ok, 260));
        mark(4);
        $("#result-loading").hidden = true;
        showResult(r);
        loadTicker();
        if (window.innerWidth < 860) $("#result-card").scrollIntoView({ behavior: "smooth", block: "start" });
        loadRecent();
      } catch (ex) {
        clearTimeout(t1);
        err.textContent = ex.message; err.hidden = false;
        $("#result-empty").hidden = false;
      } finally {
        $("#submit-btn").disabled = false; $("#result-loading").hidden = true;
      }
    });

    loadRecent();

    // /?result=<id> re-opens a stored result (linked from the History page).
    const rid = new URLSearchParams(location.search).get("result");
    if (rid) {
      api(`/api/results/${encodeURIComponent(rid)}`).then(({ result: r }) => {
        text.value = r.input_text;
        $("#text_type").value = r.text_type;
        $("#company").value = r.company_name || "";
        if ([...engineSel.options].some((o) => o.value === r.engine && !o.disabled)) engineSel.value = r.engine;
        syncPrompt();
        text.dispatchEvent(new Event("input"));
        $("#result-empty").hidden = true;
        showResult(r);
      }).catch((ex) => { err.textContent = ex.message; err.hidden = false; });
    }
  }

  function showResult(r) {
    const b = $("#r-badge");
    b.textContent = r.sentiment;
    b.className = `badge big ${r.sentiment.toLowerCase()}`;
    $("#r-company").textContent = r.company_name || r.detected_company || "Not identified";
    countUp($("#r-conf"), r.confidence, pct);
    $("#r-score").textContent = (r.sentiment_score >= 0 ? "+" : "") + r.sentiment_score.toFixed(3);
    $("#r-lat").textContent = `${(r.latency_ms / 1000).toFixed(1)} s`;
    $("#r-probs").innerHTML = probBars({
      Positive: r.prob_positive, Negative: r.prob_negative, Neutral: r.prob_neutral });
    const f = r.key_factors || [];
    $("#r-factors").innerHTML = f.length ? f.map((x) => `<li>${esc(x)}</li>`).join("")
      : `<li class="muted">—</li>`;
    $("#r-reason").textContent = r.reasoning || "—";
    const fb = $("#r-fallback");
    fb.hidden = !r.fallback_from;
    if (r.fallback_from) {
      fb.textContent = fallbackText(r);
    }
    $("#r-id").textContent = `#${r.result_id}`;
    $("#r-model").textContent = `${r.model_name}${r.prompt_version !== "n/a" ? ` · prompt ${r.prompt_version}` : ""}`;
    $("#result").hidden = false;
    animateBars($("#result"));
    setGauge(r.sentiment_score);
  }

  async function loadRecent() {
    const { results } = await api("/api/results?limit=5");
    $("#recent").innerHTML = resultsTable(results);
  }

  /* ---------------------------------------------------------- history page */
  function initHistoryPage() {
    pollStatus();
    const form = $("#filters"), limit = 20;
    let offset = 0, total = 0;

    async function load() {
      const params = new URLSearchParams(new FormData(form));
      for (const [k, v] of [...params]) if (!v) params.delete(k);
      params.set("limit", limit); params.set("offset", offset);
      const data = await api(`/api/results?${params}`);
      total = data.total;
      $("#history").innerHTML = resultsTable(data.results, { withDelete: true });
      $("#history-count").textContent = total
        ? `Showing ${offset + 1}–${offset + data.results.length} of ${total} stored results (click a row for details)`
        : "";
      $("#prev").disabled = offset === 0;
      $("#next").disabled = offset + limit >= total;
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); offset = 0; load(); });
    $("#prev").addEventListener("click", () => { offset = Math.max(0, offset - limit); load(); });
    $("#next").addEventListener("click", () => { offset += limit; load(); });

    $("#history").addEventListener("click", async (e) => {
      const del = e.target.closest("[data-del]");
      if (del) {
        await api(`/api/results/${del.dataset.del}`, { method: "DELETE" });
        load();
        return;
      }
      const row = e.target.closest("tr[data-id]");
      if (!row) return;
      const { result: r } = await api(`/api/results/${row.dataset.id}`);
      $("#detail-body").innerHTML = `
        <h2>Result #${r.result_id} ${badge(r.sentiment)}</h2>
        <p class="quote">${esc(r.input_text)}</p>
        <p><strong>Company:</strong> ${esc(r.company_name || "—")} ·
           <strong>Detected by LLM:</strong> ${esc(r.detected_company || "—")}</p>
        <div class="probs">${probBars({ Positive: r.prob_positive, Negative: r.prob_negative, Neutral: r.prob_neutral })}</div>
        <p><strong>Key factors:</strong> ${esc((r.key_factors || []).join("; ") || "—")}</p>
        <p><strong>Reasoning:</strong> ${esc(r.reasoning || "—")}</p>
        ${r.fallback_from ? `<p class="notice">${esc(fallbackText(r))}</p>` : ""}
        <p class="muted small">${esc(r.model_name)} · prompt ${esc(r.prompt_version)} ·
           ${r.latency_ms} ms · ${esc(r.submitted_at)} · source ${esc(r.source)} ·
           <a href="/?result=${r.result_id}">open in analyser</a></p>`;
      $("#detail").showModal();
      animateBars($("#detail"));
    });
    load();
  }

  /* ---------------------------------------------------------- dashboard page */
  async function initDashboardPage() {
    pollStatus();
    const { stats: s } = await api("/api/stats");
    const kpis = [
      ["Texts analysed", s.total_analyses, "", (v) => Math.round(v).toLocaleString()],
      ["Positive", s.by_sentiment.Positive, "pos", (v) => Math.round(v)],
      ["Negative", s.by_sentiment.Negative, "neg", (v) => Math.round(v)],
      ["Neutral", s.by_sentiment.Neutral, "neu", (v) => Math.round(v)],
      ["Avg. confidence", s.average_confidence, "", pct],
      ["Avg. latency", (s.average_latency_ms || 0) / 1000, "", (v) => `${v.toFixed(1)} s`],
    ];
    $("#kpis").innerHTML = kpis.map(([k, , cls], i) =>
      `<div class="kpi ${cls}"><span>${k}</span><strong id="kpi-${i}">0</strong></div>`).join("");
    kpis.forEach(([, v, , fmt], i) => countUp($(`#kpi-${i}`), v || 0, fmt));

    // Donut chart of the sentiment distribution.
    const tot = s.total_analyses || 1, R = 70, C = 2 * Math.PI * R;
    const colors = { Positive: "#22c58b", Negative: "#ff5d6c", Neutral: "#a3aec4" };
    let offset = 0;
    const segs = LABELS.map((l) => {
      const len = (s.by_sentiment[l] / tot) * C;
      const seg = { l, len, offset };
      offset += len;
      return seg;
    });
    $("#dist").innerHTML = `<div class="donut-wrap">
      <svg class="donut" viewBox="0 0 190 190">
        <circle cx="95" cy="95" r="${R}" fill="none" stroke="rgba(148,163,184,.1)" stroke-width="22"/>
        ${segs.map((g) => `<circle class="seg" cx="95" cy="95" r="${R}" stroke="${colors[g.l]}"
            stroke-dasharray="0 ${C}" data-da="${Math.max(0, g.len - 3)} ${C}" stroke-dashoffset="${-g.offset}"
            transform="rotate(-90 95 95)"><title>${g.l}: ${s.by_sentiment[g.l]}</title></circle>`).join("")}
        <text x="95" y="96" text-anchor="middle" class="d-total" id="d-total">0</text>
        <text x="95" y="116" text-anchor="middle" class="d-lab">texts analysed</text>
      </svg>
      <div class="legend-list">${LABELS.map((l) => `<div><i style="background:${colors[l]}"></i>${l}
        <b>${s.by_sentiment[l]}</b><small>${((s.by_sentiment[l] / tot) * 100).toFixed(0)}%</small></div>`).join("")}</div>
    </div>`;
    countUp($("#d-total"), s.total_analyses);
    nextFrame().then(() => document.querySelectorAll(".donut .seg").forEach((c) =>
      c.setAttribute("stroke-dasharray", c.dataset.da)));

    const types = {};
    s.by_text_type.forEach((r) => { (types[r.text_type] ??= {})[r.sentiment] = r.n; });
    $("#by-type").innerHTML = Object.entries(types).map(([t, c]) => {
      const n = LABELS.reduce((a, l) => a + (c[l] || 0), 0);
      return `<div class="stack-row"><span>${esc(TYPE_NAMES[t] || t)}</span><div class="stack">
        ${LABELS.map((l) => (c[l] ? `<div class="fill ${l.toLowerCase()}" style="width:0" data-w="${(c[l] / n) * 100}" title="${l}: ${c[l]}">${c[l]}</div>` : "")).join("")}
      </div></div>`;
    }).join("") + `<div class="legend">${LABELS.map((l) => `<span><i class="${l.toLowerCase()}"></i>${l}</span>`).join("")}</div>`;

    $("#by-company").innerHTML = s.by_company.length ? `<table><thead><tr><th>Company</th><th>Texts</th>
      <th>Positive</th><th>Negative</th><th>Neutral</th><th>Avg. score</th></tr></thead><tbody>
      ${s.by_company.map((c) => `<tr><td>${esc(c.company)}</td><td>${c.n}</td><td>${c.pos}</td>
        <td>${c.neg}</td><td>${c.neu}</td>
        <td><span class="score ${c.avg_score > 0.15 ? "positive" : c.avg_score < -0.15 ? "negative" : "neutral"}">${c.avg_score >= 0 ? "+" : ""}${c.avg_score.toFixed(2)}</span></td></tr>`).join("")}
      </tbody></table>` : `<p class="muted">No data yet.</p>`;
    animateBars();
  }

  /* ---------------------------------------------------------- test results page */
  async function initTestResultsPage() {
    pollStatus();
    const { runs } = await api("/api/test-runs");
    if (!runs.length) {
      $("#runs").innerHTML = `<p class="muted">No test runs yet. Run <code>python run_tests.py</code>.</p>`;
      return;
    }
    $("#runs").innerHTML = `<table><thead><tr><th>Run</th><th>Model</th><th>Prompt</th><th>Items</th>
      <th>Correct</th><th>Accuracy</th><th>Macro-F1</th><th>Avg. latency</th><th>Finished</th></tr></thead><tbody>
      ${runs.map((r) => `<tr data-run="${r.run_id}" class="clickable"><td>${r.run_id}</td>
        <td>${esc(r.model_name)}</td><td>${esc(r.prompt_version)}</td><td>${r.total_items}</td>
        <td>${r.correct_items}</td><td><strong>${pct(r.accuracy)}</strong></td><td>${pct(r.macro_f1)}</td>
        <td>${r.avg_latency_ms ? `${(r.avg_latency_ms / 1000).toFixed(2)} s` : "—"}</td>
        <td class="small">${esc(r.finished_at || "incomplete")}</td></tr>`).join("")}</tbody></table>
      <p class="muted small">Click a run to see its details.</p>`;
    $("#runs").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-run]");
      if (tr) showRun(tr.dataset.run);
    });
    // Open the latest complete run over the whole data set (partial runs are skipped).
    const full = Math.max(...runs.map((r) => r.total_items));
    showRun((runs.find((r) => r.finished_at && r.total_items === full)
      || runs.find((r) => r.finished_at) || runs[0]).run_id);
  }

  async function showRun(id) {
    const { run, items } = await api(`/api/test-runs/${id}`);
    document.querySelectorAll("tr[data-run]").forEach((tr) =>
      tr.classList.toggle("selected", tr.dataset.run === String(id)));
    $("#run-detail").hidden = false;
    // These sections start hidden, so the scroll observer never saw them - show them directly.
    document.querySelectorAll("#run-detail .reveal").forEach((e) => e.classList.add("in"));
    $("#run-kpis").innerHTML = [
      ["Run", `#${run.run_id}`], ["Model", engineLabel(run)],
      ["Accuracy", pct(run.accuracy)], ["Macro-F1", pct(run.macro_f1)],
      ["Correct", `${run.correct_items} / ${run.total_items}`],
    ].map(([k, v]) => `<div class="kpi"><span>${k}</span><strong>${esc(v)}</strong></div>`).join("");

    const cm = {};
    LABELS.forEach((a) => { cm[a] = {}; LABELS.forEach((b) => { cm[a][b] = 0; }); });
    items.forEach((i) => { cm[i.expected][i.predicted] += 1; });
    const max = Math.max(...items.map(() => 1), ...LABELS.flatMap((a) => LABELS.map((b) => cm[a][b])));
    $("#confusion").innerHTML = `<table class="cm"><thead><tr><th>expected ↓ / predicted →</th>
      ${LABELS.map((l) => `<th>${l}</th>`).join("")}</tr></thead><tbody>
      ${LABELS.map((a) => `<tr><th>${a}</th>${LABELS.map((b) => `<td class="${a === b ? "diag" : "off"}"
        style="--a:${cm[a][b] / max}">${cm[a][b]}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

    const per = {};
    items.forEach((i) => { const t = (per[i.text_type] ??= { n: 0, c: 0 }); t.n += 1; t.c += i.is_correct; });
    $("#per-type").innerHTML = Object.entries(per).map(([t, v]) => `
      <div class="prob"><span>${esc(TYPE_NAMES[t])}</span>
        <div class="bar"><div class="fill positive" style="width:0" data-w="${(v.c / v.n) * 100}"></div></div>
        <strong>${v.c}/${v.n}</strong></div>`).join("");

    const render = () => {
      const only = $("#only-errors").checked;
      const rows = items.filter((i) => !only || !i.is_correct);
      $("#items").innerHTML = `<table><thead><tr><th>#</th><th>Type</th><th>Company</th><th>Text</th>
        <th>Expected</th><th>Predicted</th><th>Conf.</th><th>Reasoning</th></tr></thead><tbody>
        ${rows.map((i) => `<tr class="${i.is_correct ? "" : "wrong"}"><td>${i.test_id}</td>
          <td class="small">${esc(TYPE_NAMES[i.text_type])}</td><td>${esc(i.company_name)}</td>
          <td class="text-cell">${esc(trunc(i.input_text, 140))}</td><td>${badge(i.expected)}</td>
          <td>${badge(i.predicted)} ${i.is_correct ? "✓" : "✗"}</td><td>${pct(i.confidence)}</td>
          <td class="small">${esc(i.reasoning || "")}</td></tr>`).join("")}</tbody></table>`;
    };
    $("#only-errors").onchange = render;
    render();
    animateBars($("#run-detail"));
  }

  // Runs on every page.
  initReveal();
  loadTicker();

  return { initAnalyzePage, initHistoryPage, initDashboardPage, initTestResultsPage };
})();
