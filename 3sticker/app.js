// 3sticker — main thread: UI, mask logic, styles, border and export. ML runs in worker.js.
import { applyStyle, edt } from "./fx.js";
const $ = id => document.getElementById(id);
const toast = (t, ms = 2600) => { const el = $("toast"); el.textContent = t; el.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove("show"), ms); };
const MAX_SIDE = 1600, MAX_UPSCALED = 3000;
const PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#a855f7", "#ec4899", "#14b8a6", "#84cc16", "#f97316"];
const TEXT_COLOR = "#f59e0b", BG_COLOR = "#94a3b8";
const STAGE = {
  rmbg: "Background model (RMBG-1.4, ~44 MB)",
  detr: "Object model (DETR panoptic, ~44 MB)",
  text: "Text detector (PP-OCR, ~5 MB)",
  sam: "Click-select model (SlimSAM, ~14 MB)",
  upscale: "Upscaler (Swin2SR)",
};

// ---------------- worker plumbing ----------------
let worker, jobSeq = 0, pending = new Map(), busy = false;
function startWorker() {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data: m }) => {
    if (m.type === "progress") return onProgress(m);
    const p = pending.get(m.job); if (!p) return; pending.delete(m.job);
    m.type === "error" ? p.reject(new Error(m.message)) : p.resolve(m);
  };
  worker.onerror = e => { const msg = e.message || "The AI worker failed to start (is this browser up to date?)"; for (const p of pending.values()) p.reject(new Error(msg)); pending.clear(); };
}
startWorker();
function call(type, payload, transfer) {
  const job = ++jobSeq;
  return new Promise((resolve, reject) => { pending.set(job, { resolve, reject }); worker.postMessage({ type, job, ...payload }, transfer || []); });
}
function cancelWork() {
  worker.terminate(); for (const p of pending.values()) p.reject(new Error("Cancelled")); pending.clear(); startWorker();
}
window.__3sticker = { call };

// ---------------- status bar ----------------
const fmtMB = b => (b / 1048576).toFixed(b > 1e7 ? 0 : 1) + " MB";
function status(text, frac, opts = {}) {
  const s = $("status"); s.classList.add("show"); s.classList.toggle("err", !!opts.err);
  $("stTxt").textContent = text;
  const bar = $("stBar"); bar.hidden = !!opts.err || frac === null; bar.classList.toggle("ind", frac === undefined);
  bar.firstElementChild.style.width = frac === undefined ? "" : Math.round(frac * 100) + "%";
  $("stCancel").hidden = !opts.cancel; $("stClose").hidden = !opts.err && !opts.close;
}
function statusDone(text) { if (text) { status(text, null, { close: true }); clearTimeout(statusDone.t); statusDone.t = setTimeout(() => $("status").classList.remove("show"), 3500); } else $("status").classList.remove("show"); }
$("stClose").onclick = () => $("status").classList.remove("show");
$("stCancel").onclick = () => { cancelWork(); statusDone("Cancelled."); setBusy(false); };
let stagePrefix = "";
function onProgress(m) {
  const name = STAGE[m.stage] || m.stage;
  if (m.phase === "download") status(`${stagePrefix}Downloading ${name} — ${fmtMB(m.loaded)} / ${fmtMB(m.total)} (only once, then cached)`, m.loaded / m.total, { cancel: true });
  else if (m.phase === "load") status(`${stagePrefix}Loading ${name}…`, undefined, { cancel: true });
  else if (m.phase === "fallback") status(`${stagePrefix}Text model unavailable, using a simple caption detector…`, undefined, { cancel: true });
  else if (m.stage === "upscale" && m.total) status(`Enhancing ${$("srScale").value}× on ${m.device === "webgpu" ? "GPU" : "CPU"} — tile ${m.done} / ${m.total}${eta(m)}`, m.done / m.total, { cancel: true });
  else {
    const what = { rmbg: "Finding the main subject", detr: "Finding people, animals and objects", text: "Looking for text", sam: m.note === "embedding" ? "Preparing click-select for this image" : "Selecting the clicked area" }[m.stage] || "Working";
    status(`${stagePrefix}${what}…`, undefined, { cancel: true });
  }
}
let etaStart = 0;
function eta(m) {
  if (m.done === 0) { etaStart = performance.now(); return ""; }
  const per = (performance.now() - etaStart) / m.done, left = Math.round(per * (m.total - m.done) / 1000);
  return left > 2 ? ` — about ${left < 90 ? left + " s" : Math.round(left / 60) + " min"} left` : "";
}
function setBusy(b) {
  busy = b;
  $("srBtn").disabled = b || !src; $("openBtn").disabled = b;
  $("srcView").style.opacity = b ? .85 : 1;
}

// ---------------- state ----------------
let src = null;          // ImageData of the working image
let srcKey = 0;          // bumps whenever src pixels change
let elements = [];       // {id,name,sub,color,kind,alpha:Float32Array,edge:Uint8Array,checked}
let hovered = null, tool = "select", style = "original";
let lastThings = null;   // DETR instances from the last full analysis (reused after enhancing)
let enhanced = 0;
let lastS = null;        // combined subject matte of the last analysis (Uint8)

// ---------------- loading images ----------------
async function loadFile(file) {
  if (!file) return;
  if (busy) return toast("Please wait for the current step to finish (or cancel it).");
  if (!/^image\//.test(file.type) && !/\.(png|jpe?g|gif|webp|bmp|avif)$/i.test(file.name || "")) return toast("That doesn't look like an image file.");
  let bmp;
  try { bmp = await createImageBitmap(file); } // GIF/APNG/WebP: first frame
  catch { try { bmp = await viaImg(file); } catch { return status("Couldn't read this image. Try PNG, JPG, WebP or GIF.", null, { err: true }); } }
  let w = bmp.width, h = bmp.height; const sc = Math.min(1, MAX_SIDE / Math.max(w, h));
  w = Math.max(1, Math.round(w * sc)); h = Math.max(1, Math.round(h * sc));
  const c = new OffscreenCanvasOr(w, h), ctx = c.getContext("2d"); ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close && bmp.close();
  src = ctx.getImageData(0, 0, w, h); srcKey++; enhanced = 0; elements = []; lastThings = null;
  $("empty").hidden = true; $("imgInfo").textContent = `${w}×${h}px${sc < 1 ? " (scaled down)" : ""}`;
  $("srBanner").hidden = Math.max(w, h) >= 500;
  $("srBanner").textContent = "Low-resolution image — try Enhance (2× or 4×) for sharper sticker edges.";
  checkSrLimits();
  renderList(); drawSrc(); renderSticker();
  await analyze(true);
}
function OffscreenCanvasOr(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
function viaImg(file) {
  return new Promise((res, rej) => { const u = URL.createObjectURL(file), i = new Image(); i.onload = () => { res(i); }; i.onerror = rej; i.src = u; });
}
$("openBtn").onclick = () => $("fileIn").click();
$("empty").onclick = () => $("fileIn").click();
$("fileIn").onchange = e => { loadFile(e.target.files[0]); e.target.value = ""; };
const desk = $("desk");
addEventListener("dragover", e => { e.preventDefault(); desk.classList.add("over"); });
addEventListener("dragleave", e => { if (!e.relatedTarget) desk.classList.remove("over"); });
addEventListener("drop", e => {
  e.preventDefault(); desk.classList.remove("over");
  const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/")) || e.dataTransfer?.files?.[0];
  if (f) loadFile(f); else toast("Drop an image file (PNG, JPG, WebP or GIF).");
});
addEventListener("paste", e => {
  const items = [...(e.clipboardData?.items || [])];
  const it = items.find(i => i.kind === "file" && i.type.startsWith("image/"));
  if (it) { e.preventDefault(); loadFile(it.getAsFile()); }
  else if (items.length) toast("The clipboard doesn't contain an image.");
});

// ---------------- analysis ----------------
async function analyze(full) {
  setBusy(true); stagePrefix = "";
  const img = { data: src.data.slice(), width: src.width, height: src.height };
  const key = srcKey;
  let hint = null;
  if (!full && lastThings?.subjPts) { const k = src.width / lastThings.w; hint = { pts: lastThings.subjPts.pts.map(([x, y]) => [x * k, y * k]), labels: lastThings.subjPts.labels }; }
  try {
    status("Starting the AI models…", undefined, { cancel: true });
    const r = await call("analyze", { img, key, skipDetr: !full, hint }, [img.data.buffer]);
    if (key !== srcKey) return;
    window.__3sticker.dbg = r.dbg;
    let things = r.things;
    if (full) lastThings = { things, w: src.width, h: src.height, subjPts: r.subjPts };
    else if (lastThings) things = lastThings.things.map(t => { const mask = resizeMask(t.mask, lastThings.w, lastThings.h, src.width, src.height); let area = 0; for (let i = 0; i < mask.length; i++) if (mask[i] > 127) area++; return { ...t, mask, area }; });
    buildElements(r.sal, things, r.text.mask, r.text.boxes, r.subj, full || !lastThings?.S ? null : resizeMask(lastThings.S, lastThings.w, lastThings.h, src.width, src.height));
    if (full && lastThings) lastThings.S = lastS;
    statusDone(r.warn ? "Done (object detection failed: " + r.warn + ")" : `Found ${elements.length} element${elements.length === 1 ? "" : "s"}.`);
  } catch (e) {
    if (e.message !== "Cancelled") status("Analysis failed: " + e.message + " — you can still use “+ Add area” or reload to try again.", null, { err: true });
  } finally { setBusy(false); }
}

function resizeMask(m, w, h, W, H) { // bilinear, Uint8 -> Uint8
  const o = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + .5) * h / H - .5)), y0 = fy | 0, y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + .5) * w / W - .5)), x0 = fx | 0, x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      o[y * W + x] = (m[y0 * w + x0] * (1 - tx) + m[y0 * w + x1] * tx) * (1 - ty) + (m[y1 * w + x0] * (1 - tx) + m[y1 * w + x1] * tx) * ty;
    }
  }
  return o;
}

function dilateBin(bin, W, H, r) { // bin: Uint8 0/1 ; returns Uint8 0/1 within distance r
  const d = edt(bin, W, H); const o = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) o[i] = d[i] <= r * r ? 1 : 0; return o;
}
function boxBlurF(a, W, H, r) {
  if (r < 1) return a; const t = new Float32Array(W * H), o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) { let s = 0; const row = y * W; for (let x = -r; x <= r; x++) s += a[row + Math.min(W - 1, Math.max(0, x))]; for (let x = 0; x < W; x++) { t[row + x] = s / (2 * r + 1); s += a[row + Math.min(W - 1, x + r + 1)] - a[row + Math.max(0, x - r)]; } }
  for (let x = 0; x < W; x++) { let s = 0; for (let y = -r; y <= r; y++) s += t[Math.min(H - 1, Math.max(0, y)) * W + x]; for (let y = 0; y < H; y++) { o[y * W + x] = s / (2 * r + 1); s += t[Math.min(H - 1, y + r + 1) * W + x] - t[Math.max(0, y - r) * W + x]; } }
  return o;
}
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

// Fill small enclosed gaps in the selection (e.g. between an arm and the caption) so stickers stay solid.
function fillHoles(A, W, H, maxArea) {
  const n = W * H, lab = new Int32Array(n), st = []; let id = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] || A[i] > 0.5) continue; id++; const px = []; let edge = false; st.push(i); lab[i] = id;
    while (st.length) { const j = st.pop(); px.push(j); const x = j % W, y = (j / W) | 0; if (x === 0 || y === 0 || x === W - 1 || y === H - 1) edge = true; for (const k of [j - 1, j + 1, j - W, j + W]) { if (k < 0 || k >= n || lab[k] || A[k] > 0.5) continue; if ((k === j - 1 && x === 0) || (k === j + 1 && x === W - 1)) continue; lab[k] = id; st.push(k); } }
    if (!edge && px.length <= maxArea) for (const j of px) A[j] = 1;
  }
}

// Remove small islands (< minArea px) from a soft mask, in place.
function despeckle(A, W, H, minArea) {
  const n = W * H, lab = new Int32Array(n), keep = new Uint8Array(n), st = []; let id = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] || A[i] <= 0.5) continue; id++; const px = []; st.push(i); lab[i] = id;
    while (st.length) { const j = st.pop(); px.push(j); const x = j % W; for (const k of [j - 1, j + 1, j - W, j + W]) { if (k < 0 || k >= n || lab[k] || A[k] <= 0.5) continue; if ((k === j - 1 && x === 0) || (k === j + 1 && x === W - 1)) continue; lab[k] = id; st.push(k); } }
    if (px.length >= minArea) for (const j of px) keep[j] = 1;
  }
  const near = dilateBin(keep, W, H, 2);
  for (let i = 0; i < n; i++) if (!near[i]) A[i] = 0;
}

// Split the image into a partition of soft masks: text, objects (refined by the saliency matte), leftover subject, background.
function buildElements(sal, things, textMask, textBoxes, subj, prevS) {
  const W = src.width, H = src.height, n = W * H, keepUser = elements.filter(e => e.kind === "add" || e.kind === "rem");
  const remaining = new Float32Array(n).fill(1), els = [];
  // tighten the soft matte a little so stickers don't get hazy semi-transparent fringes
  const S = new Float32Array(n); for (let i = 0; i < n; i++) { const t = Math.max(0, Math.min(1, (sal[i] / 255 - 0.12) / 0.7)); S[i] = t * t * (3 - 2 * t); }
  if (subj) {
    // after enhancing, the completion may only grow inside (a dilated copy of) the subject found before
    let gate = null;
    if (prevS) { const pb = new Uint8Array(n); for (let i = 0; i < n; i++) pb[i] = prevS[i] > 127 ? 1 : 0; gate = dilateBin(pb, W, H, Math.max(2, Math.round(Math.max(W, H) / 80))); }
    const sb = boxBlurF(Float32Array.from(subj, (v, i) => gate && !gate[i] ? 0 : v / 255), W, H, 1); for (let i = 0; i < n; i++) S[i] = Math.max(S[i], sb[i]);
  } // SAM-completed body/clothes
  lastS = new Uint8Array(n); for (let i = 0; i < n; i++) lastS[i] = S[i] * 255;
  const take = A => { const e = new Float32Array(n); for (let i = 0; i < n; i++) { const v = Math.min(A[i], remaining[i]); e[i] = v; remaining[i] -= v; } return e; };
  const area = A => { let s = 0; for (let i = 0; i < n; i++) s += A[i]; return s / n; };
  // text first: it wins over the person underneath it
  let tArea = 0; for (let i = 0; i < n; i++) tArea += textMask[i] ? 1 : 0;
  let textEl = null;
  if (tArea > n * 0.002) {
    const tb = new Uint8Array(n); for (let i = 0; i < n; i++) tb[i] = textMask[i] ? 1 : 0;
    const A = boxBlurF(Float32Array.from(tb), W, H, Math.max(0, Math.round(Math.min(W, H) / 400)));
    textEl = { name: "Text", sub: "caption / lettering", color: TEXT_COLOR, kind: "text", alpha: take(A), checked: true };
    // between the letters, only keep clearly-salient pixels for the subject (avoids ghost letters on "person only")
    for (const b of textBoxes || []) for (let y = Math.floor(b.y0); y <= Math.ceil(b.y1) && y < H; y++) for (let x = Math.floor(b.x0); x <= Math.ceil(b.x1) && x < W; x++) { const i = y * W + x; if (S[i] < 0.85) S[i] = 0; }
  }
  // objects: big & confident only; the rest stays part of the background
  const minA = n * 0.012, R = Math.max(2, Math.round(Math.max(W, H) * 0.04));
  const cand = things.filter(t => t.score >= 0.6 && t.area >= minA).map(t => {
    let s = 0; for (let i = 0; i < n; i++) if (t.mask[i]) s += S[i]; return { ...t, sal: s / t.area };
  }).sort((a, b) => (b.sal > 0.35) - (a.sal > 0.35) || b.area - a.area);
  const counts = {}, objEls = [];
  const covered = new Uint8Array(n);
  for (const t of cand.slice(0, 7)) {
    const bin = new Uint8Array(n); for (let i = 0; i < n; i++) bin[i] = t.mask[i] > 127 ? 1 : 0;
    let A;
    if (t.sal > 0.35) { const dil = dilateBin(bin, W, H, R); A = new Float32Array(n); for (let i = 0; i < n; i++) { A[i] = dil[i] ? S[i] : 0; if (dil[i]) covered[i] = 1; } }
    else { A = boxBlurF(Float32Array.from(bin), W, H, Math.max(1, Math.round(Math.min(W, H) / 300))); for (let i = 0; i < n; i++) if (bin[i]) covered[i] = 1; }
    despeckle(A, W, H, n * 0.004);
    const a = take(A); if (area(a) < 0.004) continue;
    const label = cap(t.label); counts[label] = (counts[label] || 0) + 1;
    objEls.push({ name: label, sub: t.sal > 0.35 ? "main subject" : `${Math.round(t.score * 100)}% sure`, color: PALETTE[objEls.length % PALETTE.length], kind: "obj", alpha: a, checked: t.sal > 0.35, label });
  }
  const seen = {};
  for (const e of objEls) { seen[e.label] = (seen[e.label] || 0) + 1; if (seen[e.label] > 1) e.name += " " + seen[e.label]; }
  // salient pixels no detector explained -> "Main subject"
  const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = covered[i] ? 0 : S[i];
  despeckle(L, W, H, n * 0.004);
  let lA = 0; for (let i = 0; i < n; i++) lA += L[i] > 0.5 ? 1 : 0;
  if (lA > n * (objEls.length ? 0.02 : 0.003)) {
    objEls.push({ name: objEls.length ? "Other subject" : "Main subject", sub: "salient object", color: PALETTE[objEls.length % PALETTE.length], kind: "obj", alpha: take(L), checked: true });
  }
  if (textEl) els.push(textEl);
  els.unshift(...objEls);
  els.push({ name: "Background", sub: "everything else", color: BG_COLOR, kind: "bg", alpha: remaining, checked: false });
  // carry over user refinements (rescaled if the image size changed)
  for (const u of keepUser) {
    if (u.w !== W || u.h !== H) { const m8 = new Uint8Array(u.w * u.h); for (let i = 0; i < m8.length; i++) m8[i] = u.alpha[i] * 255; const r = resizeMask(m8, u.w, u.h, W, H); u.alpha = new Float32Array(n); for (let i = 0; i < n; i++) u.alpha[i] = r[i] / 255; u.w = W; u.h = H; }
    els.push(u);
  }
  const prev = new Map(elements.map(e => [e.name, e.checked]));
  for (const e of els) if (prev.has(e.name) && e.kind !== "add" && e.kind !== "rem") e.checked = prev.get(e.name);
  elements = els.map((e, i) => finishEl(e, i));
  hovered = null; renderList(); drawSrc(); renderSticker();
}
let elSeq = 0;
function finishEl(e) {
  const W = src.width, H = src.height; e.id = e.id || ++elSeq; e.w = W; e.h = H;
  const edge = new Uint8Array(W * H), a = e.alpha;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x; if (a[i] <= 0.5) continue;
    if (x === 0 || y === 0 || x === W - 1 || y === H - 1 || a[i - 1] <= 0.5 || a[i + 1] <= 0.5 || a[i - W] <= 0.5 || a[i + W] <= 0.5) edge[i] = 1;
  }
  e.edge = edge; const rgb = parseInt(e.color.slice(1), 16); e.rgb = [rgb >> 16, (rgb >> 8) & 255, rgb & 255];
  return e;
}

// ---------------- element list ----------------
function renderList() {
  const ul = $("elList"); ul.textContent = "";
  $("elEmpty").hidden = elements.length > 0; $("elActs").hidden = !elements.length;
  if (src && !elements.length) $("elEmpty").textContent = busy ? "Scanning the image…" : "No elements yet.";
  for (const e of elements) {
    const li = document.createElement("li"); li.dataset.id = e.id; if (hovered === e) li.classList.add("hov");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = e.checked; cb.setAttribute("aria-label", e.name);
    cb.onclick = ev => ev.stopPropagation();
    cb.onchange = () => { e.checked = cb.checked; drawSrc(); renderSticker(); };
    const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = e.color;
    if (e.kind === "rem") sw.style.background = `repeating-linear-gradient(45deg,${e.color} 0 3px,transparent 3px 6px)`;
    const nm = document.createElement("span"); nm.className = "nm"; nm.innerHTML = ""; nm.textContent = e.name + " ";
    const sub = document.createElement("span"); sub.className = "sub"; sub.textContent = e.sub || ""; nm.append(sub);
    li.append(cb, sw, nm);
    if (e.kind === "add" || e.kind === "rem") { const d = document.createElement("button"); d.className = "del"; d.textContent = "✕"; d.title = "Delete"; d.onclick = ev => { ev.stopPropagation(); elements = elements.filter(x => x !== e); if (hovered === e) hovered = null; renderList(); drawSrc(); renderSticker(); }; li.append(d); }
    li.onclick = () => { e.checked = !e.checked; cb.checked = e.checked; drawSrc(); renderSticker(); };
    li.onmouseenter = () => setHover(e); li.onmouseleave = () => setHover(null);
    ul.append(li);
  }
}
function setHover(e) {
  if (hovered === e) return; hovered = e;
  for (const li of $("elList").children) li.classList.toggle("hov", +li.dataset.id === e?.id);
  drawSrc();
}
$("allBtn").onclick = () => { elements.forEach(e => e.checked = e.kind !== "rem" ? true : e.checked); renderList(); drawSrc(); renderSticker(); };
$("noneBtn").onclick = () => { elements.forEach(e => { if (e.kind !== "rem") e.checked = false; }); renderList(); drawSrc(); renderSticker(); };

// ---------------- fit canvases into their views (upscales small images too) ----------------
function fit(c) {
  const v = c.parentElement, cs = getComputedStyle(v), pw = v.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight), ph = v.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  if (pw <= 0 || ph <= 0) return; const k = Math.min(pw / c.width, ph / c.height, c === $("outCanvas") ? 4 : 8);
  c.style.width = Math.floor(c.width * k) + "px"; c.style.height = Math.floor(c.height * k) + "px";
}
new ResizeObserver(() => { fit($("srcCanvas")); fit($("outCanvas")); }).observe($("desk"));

// ---------------- image view with overlays ----------------
const sc = $("srcCanvas");
function drawSrc() {
  if (!src) return;
  const W = src.width, H = src.height; if (sc.width !== W || sc.height !== H) { sc.width = W; sc.height = H; }
  const out = new ImageData(new Uint8ClampedArray(src.data), W, H), d = out.data, n = W * H;
  for (let i = 0; i < n; i++) {
    let r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2];
    for (const e of elements) {
      const a = e.alpha[i]; if (a < 0.02 && !e.edge[i]) continue;
      if (hovered) {
        if (e === hovered) { const k = .55 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; }
        else if (e.kind !== "rem") { const k = .45 * a; r -= r * k; g -= g * k; b -= b * k; }
      } else if (e.kind === "rem") { if (e.checked && ((i % W + (i / W | 0)) & 4)) { const k = .6 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; } }
      else if (e.checked) { const k = .32 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; }
      else { const k = .5 * a; r -= r * k; g -= g * k; b -= b * k; }
      if (e.edge[i] && (e === hovered || (!hovered && e.checked))) { r = e.rgb[0]; g = e.rgb[1]; b = e.rgb[2]; }
    }
    d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
  }
  sc.getContext("2d").putImageData(out, 0, 0); fit(sc);
}
function pixelAt(ev) {
  const r = sc.getBoundingClientRect();
  const x = Math.floor((ev.clientX - r.left) / r.width * sc.width), y = Math.floor((ev.clientY - r.top) / r.height * sc.height);
  return x >= 0 && y >= 0 && x < sc.width && y < sc.height ? [x, y] : null;
}
function elementAt(x, y) {
  const i = y * src.width + x; let best = null, bv = 0.15;
  for (const e of elements) if (e.kind !== "rem" && e.alpha[i] > bv) { bv = e.alpha[i]; best = e; }
  return best;
}
sc.addEventListener("mousemove", ev => {
  if (!src || !elements.length) return; const p = pixelAt(ev); const e = p && elementAt(...p);
  setHover(tool === "select" ? e : null);
  const tip = $("tip"); if (!p) { tip.style.display = "none"; return; }
  const vr = $("srcView").getBoundingClientRect();
  tip.textContent = tool === "select" ? (e ? `${e.name} — click to ${e.checked ? "remove" : "add"}` : "") : tool === "add" ? "Click to add this part" : "Click to cut this part out";
  tip.style.display = tip.textContent ? "block" : "none"; tip.style.left = (ev.clientX - vr.left + 14) + "px"; tip.style.top = (ev.clientY - vr.top + 10) + "px";
});
sc.addEventListener("mouseleave", () => { setHover(null); $("tip").style.display = "none"; });
sc.addEventListener("click", async ev => {
  if (!src) return; const p = pixelAt(ev); if (!p) return;
  if (tool === "select") {
    const e = elementAt(...p); if (!e) return; e.checked = !e.checked; renderList(); drawSrc(); renderSticker(); return;
  }
  if (busy) return toast("Please wait for the current step to finish.");
  await samRefine(p[0], p[1], tool === "add");
});
async function samRefine(x, y, add) {
  setBusy(true); stagePrefix = "";
  const img = { data: src.data.slice(), width: src.width, height: src.height }, key = srcKey;
  try {
    status("Preparing click-select…", undefined, { cancel: true });
    const r = await call("sam", { img, key, x, y }, [img.data.buffer]);
    if (key !== srcKey) return;
    const n = src.width * src.height, A = new Float32Array(n); let s = 0;
    for (let i = 0; i < n; i++) { A[i] = r.mask[i] / 255; s += A[i]; }
    if (s < 4) { statusDone("Nothing found at that spot — try clicking a bit inside the part."); return; }
    const cnt = elements.filter(e => e.kind === (add ? "add" : "rem")).length + 1;
    const e = finishEl({ name: add ? `Added area ${cnt}` : `Removed area ${cnt}`, sub: "click-select", color: add ? PALETTE[(elements.length + 2) % PALETTE.length] : "#dc2626", kind: add ? "add" : "rem", alpha: A, checked: true });
    elements.push(e); renderList(); drawSrc(); renderSticker();
    statusDone(add ? "Area added." : "Area removed from the sticker.");
  } catch (e) {
    if (e.message !== "Cancelled") status("Click-select failed: " + e.message, null, { err: true });
  } finally { setBusy(false); }
}
$("toolSeg").onclick = ev => {
  const b = ev.target.closest("button"); if (!b) return; tool = b.dataset.tool;
  for (const x of $("toolSeg").children) x.classList.toggle("on", x === b);
  $("srcView").className = "view " + (tool === "select" ? "" : tool);
  $("toolHint").textContent = { select: "Click an element on the image to toggle it.", add: "Click any part of the image to add it as its own element (first click downloads a ~14 MB model).", rem: "Click any part that should not be in the sticker — it is cut out." }[tool];
};

// ---------------- enhance ----------------
function checkSrLimits() {
  if (!src) return; const s = +$("srScale").value, big = Math.max(src.width, src.height) * s > MAX_UPSCALED;
  $("srInfo").textContent = big ? `This image is too large for ${s}× (max ${MAX_UPSCALED}px result).` : enhanced ? `Enhanced ${enhanced}× (${src.width}×${src.height}px). You can enhance again.` : "AI super-resolution runs before cutting, for sharper edges. Can take a while without a GPU.";
  $("srBtn").disabled = busy || big;
}
$("srScale").onchange = checkSrLimits;
$("srBtn").onclick = async () => {
  if (!src || busy) return; const s = +$("srScale").value;
  setBusy(true); stagePrefix = "";
  const img = { data: src.data.slice(), width: src.width, height: src.height }, key = srcKey;
  try {
    status("Starting the upscaler…", undefined, { cancel: true });
    const r = await call("upscale", { img, scale: s, gpu: $("srGpu").checked }, [img.data.buffer]);
    if (key !== srcKey) return;
    src = new ImageData(new Uint8ClampedArray(r.img.data.buffer), r.img.width, r.img.height); srcKey++;
    enhanced = (enhanced || 1) * s;
    $("imgInfo").textContent = `${src.width}×${src.height}px (enhanced ${enhanced}× on ${r.device === "webgpu" ? "GPU" : "CPU"})`; $("srBanner").hidden = true;
    bdAuto(); drawSrc(); renderSticker();
  } catch (e) {
    setBusy(false);
    if (e.message !== "Cancelled") status("Enhance failed: " + e.message, null, { err: true });
    return;
  }
  setBusy(false); checkSrLimits();
  stagePrefix = "Re-cutting at the new size: ";
  await analyze(false);
  checkSrLimits();
};

// ---------------- style / border ----------------
$("styleSeg").onclick = ev => { const b = ev.target.closest("button"); if (!b) return; style = b.dataset.style; for (const x of $("styleSeg").children) x.classList.toggle("on", x === b); renderSticker(); };
let bdTouched = false;
function bdAuto() { if (bdTouched || !src) return; const v = Math.max(2, Math.min(60, Math.round(Math.max(src.width, src.height) * 0.03))); $("bdW").value = v; $("bdWv").textContent = v + " px"; }
$("bdW").oninput = () => { bdTouched = true; $("bdWv").textContent = $("bdW").value + " px"; renderSticker(); };
for (const id of ["bdOn", "bdC", "shOn", "keepText"]) $(id).oninput = () => renderSticker();

// ---------------- sticker rendering ----------------
let styleCache = { key: "", img: null };
function styled() {
  const keepText = $("keepText").checked, textEl = elements.find(e => e.kind === "text");
  const k = srcKey + ":" + style + ":" + (keepText && textEl ? textEl.id : ""); if (styleCache.key === k) return styleCache.img;
  let img = style === "original" ? src : applyStyle(src, style);
  if (img !== src && keepText && textEl) { // captions stay readable: blend original pixels back under the text mask
    const d = img.data, s = src.data, a = textEl.alpha;
    for (let i = 0; i < a.length; i++) if (a[i] > 0) { const t = a[i]; for (let c = 0; c < 3; c++) d[i * 4 + c] = d[i * 4 + c] * (1 - t) + s[i * 4 + c] * t; }
  }
  styleCache = { key: k, img }; return img;
}
let renderQueued = false;
function renderSticker() { if (!renderQueued) { renderQueued = true; requestAnimationFrame(() => { renderQueued = false; try { doRender(); } catch (e) { console.error(e); status("Rendering failed: " + e.message, null, { err: true }); } }); } }
function selectionAlpha() {
  const n = src.width * src.height, A = new Float32Array(n);
  for (const e of elements) if (e.checked && e.kind !== "rem") for (let i = 0; i < n; i++) A[i] += e.alpha[i];
  for (let i = 0; i < n; i++) A[i] = Math.min(1, A[i]);
  fillHoles(A, src.width, src.height, n * 0.02);
  for (const e of elements) if (e.checked && e.kind === "rem") for (let i = 0; i < n; i++) A[i] *= 1 - e.alpha[i];
  for (let i = 0; i < n; i++) A[i] *= src.data[i * 4 + 3] / 255;
  return A;
}
const oc = $("outCanvas");
function doRender() {
  const ph = $("outPh");
  if (!src || !elements.length) { oc.hidden = true; ph.hidden = false; ph.textContent = src ? (busy ? "Your sticker appears here after the scan." : "Nothing to show yet.") : "Your sticker appears here."; $("saveBtn").disabled = $("copyBtn").disabled = true; $("outInfo").textContent = ""; return; }
  const W = src.width, H = src.height, A = selectionAlpha();
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (A[y * W + x] > 0.03) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) { oc.hidden = true; ph.hidden = false; ph.textContent = "Select at least one element (tick a checkbox or click the image)."; $("saveBtn").disabled = $("copyBtn").disabled = true; $("outInfo").textContent = ""; return; }
  oc.hidden = false; ph.hidden = true;
  const bdOn = $("bdOn").checked, bw = bdOn ? +$("bdW").value : 0, shOn = $("shOn").checked;
  const shB = shOn ? Math.max(4, Math.round(Math.max(x1 - x0, y1 - y0) * 0.025) + bw * 0.4) : 0;
  const pad = Math.ceil(bw + (shOn ? shB * 1.6 : 0) + 2);
  const cw = x1 - x0 + 1 + 2 * pad, ch = y1 - y0 + 1 + 2 * pad;
  const st = styled(), sd = st.data;
  // subject layer
  const subj = new ImageData(cw, ch), s = subj.data;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x, o = ((y - y0 + pad) * cw + (x - x0 + pad)) * 4;
    s[o] = sd[i * 4]; s[o + 1] = sd[i * 4 + 1]; s[o + 2] = sd[i * 4 + 2]; s[o + 3] = A[i] * 255;
  }
  const layer = OffscreenCanvasOr(cw, ch); layer.getContext("2d").putImageData(subj, 0, 0);
  let shape = layer;
  if (bdOn) {
    // border = everything within `bw` px of the (binarised) cutout, anti-aliased with the true distance
    const bin = new Uint8Array(cw * ch); for (let i = 0; i < cw * ch; i++) bin[i] = s[i * 4 + 3] > 110 ? 1 : 0;
    const d2 = edt(bin, cw, ch), bd = new ImageData(cw, ch), b = bd.data;
    const col = parseInt($("bdC").value.slice(1), 16), cr = col >> 16, cg = (col >> 8) & 255, cb = col & 255;
    for (let i = 0; i < cw * ch; i++) { const a = Math.max(0, Math.min(1, bw + 0.5 - Math.sqrt(d2[i]))); b[i * 4] = cr; b[i * 4 + 1] = cg; b[i * 4 + 2] = cb; b[i * 4 + 3] = a * 255; }
    shape = OffscreenCanvasOr(cw, ch); shape.getContext("2d").putImageData(bd, 0, 0);
  }
  oc.width = cw; oc.height = ch; const ctx = oc.getContext("2d", { willReadFrequently: true }); ctx.clearRect(0, 0, cw, ch);
  if (shOn) { ctx.save(); ctx.shadowColor = "rgba(0,0,0,.38)"; ctx.shadowBlur = shB; ctx.shadowOffsetY = shB * 0.45; ctx.drawImage(shape, 0, 0); ctx.restore(); }
  if (bdOn || !shOn) ctx.drawImage(shape, 0, 0);
  if (bdOn) ctx.drawImage(layer, 0, 0);
  fit(oc); $("outInfo").textContent = `${cw}×${ch}px PNG`;
  $("saveBtn").disabled = $("copyBtn").disabled = false;
  oc.dataset.rendered = String(Date.now());
}

// ---------------- export ----------------
const pngBlob = () => new Promise((res, rej) => oc.toBlob(b => b ? res(b) : rej(new Error("PNG export failed")), "image/png"));
$("saveBtn").onclick = async () => {
  try { const b = await pngBlob(), a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "sticker.png"; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); toast("Saved sticker.png"); }
  catch (e) { toast("Download failed: " + e.message); }
};
$("copyBtn").onclick = async () => {
  try {
    if (!window.ClipboardItem || !navigator.clipboard?.write) throw new Error("this browser can't copy images — use Download");
    await navigator.clipboard.write([new ClipboardItem({ "image/png": pngBlob() })]);
    toast("Sticker copied — paste it anywhere (Ctrl+V).");
  } catch (e) { toast("Couldn't copy: " + (e.message || e)); }
};

// tests / debugging hooks
window.__3sticker.state = () => ({ w: src?.width, h: src?.height, busy, enhanced, elements: elements.map(e => ({ id: e.id, name: e.name, kind: e.kind, checked: e.checked, area: e.alpha.reduce((s, v) => s + (v > 0.5), 0) })) });
window.__3sticker.setChecked = (names) => { for (const e of elements) e.checked = names.includes(e.name); renderList(); drawSrc(); renderSticker(); };
window.__3sticker.loadFile = loadFile;
renderSticker();
document.body.dataset.ready = "1";
