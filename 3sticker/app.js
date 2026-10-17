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
let hovered = null, style = "original";
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
  zoom = 1; panX = panY = 0; applyView(); brushAuto(); history.length = 0; future.length = 0;
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
    lastAnalysis = { sal: r.sal, things, text: r.text, subj: r.subj, prevS: full || !lastThings?.S ? null : resizeMask(lastThings.S, lastThings.w, lastThings.h, src.width, src.height) };
    rebuild();
    if (full && lastThings) lastThings.S = lastS;
    statusDone(r.warn ? "Done (object detection failed: " + r.warn + ")" : `Found ${elements.length} element${elements.length === 1 ? "" : "s"}.`);
  } catch (e) {
    if (e.message !== "Cancelled") status("Analysis failed: " + e.message + " — you can still use “+ Add area” or reload to try again.", null, { err: true });
  } finally { setBusy(false); }
}

let lastAnalysis = null;
function rebuild() { const a = lastAnalysis; if (a) buildElements(a.sal, a.things, a.text, a.subj, a.prevS); }
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
function fillHoles(A, W, H, maxArea, protect) {
  const n = W * H, lab = new Int32Array(n), st = []; let id = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] || A[i] > 0.5) continue; id++; const px = []; let edge = false; st.push(i); lab[i] = id;
    while (st.length) { const j = st.pop(); px.push(j); const x = j % W, y = (j / W) | 0; if (x === 0 || y === 0 || x === W - 1 || y === H - 1) edge = true; for (const k of [j - 1, j + 1, j - W, j + W]) { if (k < 0 || k >= n || lab[k] || A[k] > 0.5) continue; if ((k === j - 1 && x === 0) || (k === j + 1 && x === W - 1)) continue; lab[k] = id; st.push(k); } }
    if (edge || px.length > maxArea) continue;
    let pr = 0; if (protect) for (const j of px) pr += protect[j];
    if (pr <= px.length * 0.1) for (const j of px) A[j] = 1;
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
function buildElements(sal, things, text, subj, prevS) {
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
  // text first, one element per detected region. Regions printed on a confident subject (a jersey number…) keep
  // their full pixels inside the subject mask instead of the colour-based letter extraction.
  const textEls = [], fullOnSubject = $("textFull").checked;
  for (const r of text.regions || []) {
    const bx0 = Math.max(0, Math.floor(r.x0)), by0 = Math.max(0, Math.floor(r.y0)), bx1 = Math.min(W - 1, Math.ceil(r.x1)), by1 = Math.min(H - 1, Math.ceil(r.y1));
    let on = 0, tot = 0; for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) { tot++; if (S[y * W + x] > 0.6) on++; }
    const onSubject = tot && on / tot > 0.55;
    let A;
    if (onSubject && fullOnSubject) {
      A = new Float32Array(n);
      for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) { const i = y * W + x; A[i] = S[i]; }
    } else {
      const tb = Float32Array.from(r.mask, v => v ? 1 : 0);
      A = boxBlurF(tb, W, H, Math.max(0, Math.round(Math.min(W, H) / 400)));
      // between the letters, only keep clearly-salient pixels for the subject (avoids ghost letters on "person only")
      if (!onSubject) for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) { const i = y * W + x; if (S[i] < 0.85) S[i] = 0; }
    }
    const a = take(A); if (area(a) < 0.0002) continue;
    // isolated, upright blobs away from the subject are rarely real text: listed, but off by default
    const tall = !onSubject && (by1 - by0) > 1.3 * (bx1 - bx0);
    textEls.push({ name: "Text " + (textEls.length + 1), sub: onSubject ? (fullOnSubject ? "on subject · full pixels" : "on subject") : tall ? "unlikely text" : `${Math.round(r.score * 100)}% sure`, color: TEXT_COLOR, kind: "text", group: "text", alpha: a, checked: !tall, box: [bx0, by0, bx1, by1] });
  }
  if (textEls.length === 1) textEls[0].name = "Text";
  // objects: big & confident only; the rest stays part of the background
  const minA = n * 0.012, R = Math.max(2, Math.round(Math.max(W, H) * 0.04));
  let cand = things.filter(t => t.score >= 0.6 && t.area >= minA).map(t => {
    let s = 0, c = 0; for (let i = 0; i < n; i++) if (t.mask[i]) { s += S[i]; if (S[i] > 0.6) c++; }
    const sal = Math.max(s / t.area, c / t.area > 0.3 ? 0.36 : 0); return { ...t, sal, meanS: s / t.area };
  });
  // when people are the subject, other things (a car between them…) need to be clearly salient on their own
  if (cand.some(t => t.label === "person" && t.sal > 0.35)) for (const t of cand) if (t.label !== "person" && t.meanS < 0.55) t.sal = Math.min(t.sal, 0.3);
  cand.sort((a, b) => (b.sal > 0.35) - (a.sal > 0.35) || b.area - a.area);
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
    objEls.push({ name: label, sub: t.sal > 0.35 ? "main subject" : `${Math.round(t.score * 100)}% sure · part of background`, color: PALETTE[objEls.length % PALETTE.length], kind: "obj", alpha: a, checked: t.sal > 0.35, label, group: t.sal > 0.35 ? null : "bg" });
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
  els.push(...textEls);
  els.unshift(...objEls.filter(e => e.group !== "bg"));
  els.push({ name: "Background", sub: "everything else", color: BG_COLOR, kind: "bg", alpha: remaining, checked: false });
  els.push(...objEls.filter(e => e.group === "bg"));
  // carry over user refinements (rescaled if the image size changed)
  for (const u of keepUser) {
    if (u.w !== W || u.h !== H) { const m8 = new Uint8Array(u.w * u.h); for (let i = 0; i < m8.length; i++) m8[i] = u.alpha[i] * 255; const r = resizeMask(m8, u.w, u.h, W, H); u.alpha = new Float32Array(n); for (let i = 0; i < n; i++) u.alpha[i] = r[i] / 255; u.w = W; u.h = H; }
    els.push(u);
  }
  const prev = new Map(elements.map(e => [e.name, e.checked]));
  for (const e of els) if (prev.has(e.name) && e.kind !== "add" && e.kind !== "rem") e.checked = prev.get(e.name);
  elements = els.map((e, i) => finishEl(e, i));
  noFill = new Uint8Array(n); history.length = 0; future.length = 0; selGrow = 0; selFeather = 0;
  if (target !== "sel" && !elements.some(e => e.id === target)) target = "sel";
  hovered = null; renderList(); syncAdjust(); drawSrc(); renderSticker();
}
let elSeq = 0;
function finishEl(e) {
  const W = src.width, H = src.height; e.id = e.id || ++elSeq; e.w = W; e.h = H;
  e.ver = (e.ver || 0) + 1; e.grow = e.grow || 0; e.feather = e.feather || 0;
  const rgb = parseInt(e.color.slice(1), 16); e.rgb = [rgb >> 16, (rgb >> 8) & 255, rgb & 255];
  return e;
}

// ---------------- effective masks (grow/shrink + feather) ----------------
function morph(a, W, H, g, f) {
  let out = a;
  if (g) {
    const n = W * H, bin = new Uint8Array(n);
    out = new Float32Array(n);
    if (g > 0) { for (let i = 0; i < n; i++) bin[i] = a[i] > 0.5 ? 1 : 0; const d = edt(bin, W, H); for (let i = 0; i < n; i++) out[i] = Math.max(a[i], Math.min(1, Math.max(0, g + 0.5 - Math.sqrt(d[i])))); }
    else { for (let i = 0; i < n; i++) bin[i] = a[i] > 0.5 ? 0 : 1; const d = edt(bin, W, H); for (let i = 0; i < n; i++) out[i] = Math.min(a[i], Math.min(1, Math.max(0, Math.sqrt(d[i]) + g + 0.5))); }
  }
  if (f > 0) out = boxBlurF(boxBlurF(out, W, H, Math.ceil(f / 2)), W, H, Math.ceil(f / 2));
  return out;
}
function effA(e) {
  if (!e.grow && !e.feather) return e.alpha;
  const k = e.ver + ":" + e.grow + ":" + e.feather;
  if (e._ek !== k) { e._eff = morph(e.alpha, src.width, src.height, e.grow, e.feather); e._ek = k; }
  return e._eff;
}
function edgesOf(e) {
  const k = e.ver + ":" + e.grow + ":" + e.feather;
  if (e._edk !== k) { e._edge = new Uint8Array(src.width * src.height); edgeRect(e, [0, 0, src.width - 1, src.height - 1]); e._edk = k; }
  return e._edge;
}
function edgeRect(e, [x0, y0, x1, y1]) { // (re)compute outline pixels inside a rect
  const W = src.width, H = src.height, a = effA(e), edge = e._edge;
  x0 = Math.max(0, x0 - 1); y0 = Math.max(0, y0 - 1); x1 = Math.min(W - 1, x1 + 1); y1 = Math.min(H - 1, y1 + 1);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x;
    edge[i] = a[i] > 0.5 && (x === 0 || y === 0 || x === W - 1 || y === H - 1 || a[i - 1] <= 0.5 || a[i + 1] <= 0.5 || a[i - W] <= 0.5 || a[i + W] <= 0.5) ? 1 : 0;
  }
}
const bump = e => { e.ver++; e._ek = e._edk = null; };

// ---------------- element list ----------------
let target = "sel";      // "sel" = the whole sticker, or an element id
function setChecked(e, v) {
  e.checked = v;
  if (e.kind === "bg") for (const c of elements) if (c.group === "bg") c.checked = v; // background objects follow the background
}
function thumb(e) {
  if (e._thumb) return e._thumb;
  const [x0, y0, x1, y1] = e.box, w = x1 - x0 + 1, h = y1 - y0 + 1, k = Math.min(1, 64 / w, 22 / h);
  const c = OffscreenCanvasOr(Math.max(1, Math.round(w * k)), Math.max(1, Math.round(h * k))), t = OffscreenCanvasOr(w, h);
  t.getContext("2d").putImageData(src, -x0, -y0, x0, y0, w, h); c.getContext("2d").drawImage(t, 0, 0, c.width, c.height);
  c.className = "th"; return (e._thumb = c);
}
function renderList() {
  const ul = $("elList"); ul.textContent = "";
  $("elEmpty").hidden = elements.length > 0; $("elActs").hidden = !elements.length;
  if (src && !elements.length) $("elEmpty").textContent = busy ? "Scanning the image…" : "No elements yet.";
  const texts = elements.filter(e => e.kind === "text");
  const row = (e, indent) => {
    const li = document.createElement("li"); li.dataset.id = e.id; if (indent) li.classList.add("child");
    if (hovered === e) li.classList.add("hov"); if (target === e.id && tool !== "select") li.classList.add("active");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = e.checked; cb.setAttribute("aria-label", e.name);
    cb.onclick = ev => ev.stopPropagation();
    cb.onchange = () => { setChecked(e, cb.checked); renderList(); drawSrc(); renderSticker(); };
    const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = e.color;
    if (e.kind === "rem") sw.style.background = `repeating-linear-gradient(45deg,${e.color} 0 3px,transparent 3px 6px)`;
    const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = e.name + " ";
    const sub = document.createElement("span"); sub.className = "sub"; sub.textContent = e.sub || ""; nm.append(sub);
    li.append(cb, sw, nm);
    if (e.kind === "text" && e.box) li.append(thumb(e));
    if (e.kind === "add" || e.kind === "rem") { const d = document.createElement("button"); d.className = "del"; d.textContent = "✕"; d.title = "Delete"; d.onclick = ev => { ev.stopPropagation(); elements = elements.filter(x => x !== e); if (hovered === e) hovered = null; if (target === e.id) target = "sel"; renderList(); drawSrc(); renderSticker(); }; li.append(d); }
    li.onclick = () => {
      if (tool !== "select") { target = e.id; renderList(); syncAdjust(); return; } // edit mode: pick what the tools change
      setChecked(e, !e.checked); renderList(); drawSrc(); renderSticker();
    };
    li.onmouseenter = () => setHover(e); li.onmouseleave = () => setHover(null);
    ul.append(li);
  };
  let textDone = false;
  for (const e of elements) {
    if (e.group === "bg") continue;
    if (e.kind === "text") {
      if (textDone) continue; textDone = true;
      if (texts.length > 1) { // group header for the text regions
        const li = document.createElement("li"); li.className = "grp";
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.setAttribute("aria-label", "All text");
        const on = texts.filter(t => t.checked).length; cb.checked = on === texts.length; cb.indeterminate = on > 0 && on < texts.length;
        cb.onclick = ev => ev.stopPropagation();
        cb.onchange = () => { texts.forEach(t => t.checked = cb.checked); renderList(); drawSrc(); renderSticker(); };
        const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = TEXT_COLOR;
        const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = `Text `; const sub = document.createElement("span"); sub.className = "sub"; sub.textContent = `${texts.length} regions`; nm.append(sub);
        li.append(cb, sw, nm); li.onclick = () => { const v = !cb.checked; texts.forEach(t => t.checked = v); renderList(); drawSrc(); renderSticker(); };
        ul.append(li); texts.forEach(t => row(t, true));
      } else row(e, false);
      continue;
    }
    row(e, false);
    if (e.kind === "bg") for (const c of elements) if (c.group === "bg") row(c, true);
  }
  // "apply to" choices for the edit tools
  const sel = $("target"), cur = String(target); sel.textContent = "";
  const opt = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.append(o); };
  opt("sel", "Whole sticker (final cutout)");
  for (const e of elements) opt(e.id, (e.group ? "  " : "") + e.name);
  sel.value = elements.some(e => String(e.id) === cur) ? cur : "sel";
  $("undoBtn").disabled = !history.length; $("redoBtn").disabled = !future.length;
}
function setHover(e) {
  if (hovered === e) return; hovered = e;
  for (const li of $("elList").children) li.classList.toggle("hov", +li.dataset.id === e?.id);
  drawSrc();
}
$("allBtn").onclick = () => { elements.forEach(e => e.checked = e.kind !== "rem" ? true : e.checked); renderList(); drawSrc(); renderSticker(); };
$("noneBtn").onclick = () => { elements.forEach(e => { if (e.kind !== "rem") e.checked = false; }); renderList(); drawSrc(); renderSticker(); };
$("textFull").onchange = () => { if (lastAnalysis) { rebuild(); toast($("textFull").checked ? "Text on clothing keeps its full pixels." : "Text on clothing uses letter shapes."); } };

// ---------------- view: fit, zoom (wheel) and pan (space+drag) ----------------
const sc = $("srcCanvas"), stage = $("stage"), svg = $("ovSvg");
let zoom = 1, panX = 0, panY = 0;
function fit(c) {
  const box = c === sc ? stage : c, v = box.parentElement, cs = getComputedStyle(v), pw = v.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight), ph = v.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  if (pw <= 0 || ph <= 0) return; const k = Math.min(pw / c.width, ph / c.height, c === $("outCanvas") ? 4 : 8);
  box.style.width = Math.floor(c.width * k) + "px"; box.style.height = Math.floor(c.height * k) + "px";
  if (c === sc) { svg.setAttribute("viewBox", `0 0 ${c.width} ${c.height}`); applyView(); }
}
function applyView() { stage.style.transform = `translate(${panX}px,${panY}px) scale(${zoom})`; $("zVal").textContent = Math.round(zoom * 100) + "%"; }
function zoomAt(f, cx, cy) {
  const r = stage.getBoundingClientRect(), nz = Math.min(16, Math.max(1, zoom * f));
  if (cx === undefined) { cx = r.left + r.width / 2; cy = r.top + r.height / 2; }
  const ux = (cx - r.left) / zoom, uy = (cy - r.top) / zoom, bx = r.left - panX, by = r.top - panY;
  zoom = nz; panX = nz === 1 ? 0 : cx - bx - ux * nz; panY = nz === 1 ? 0 : cy - by - uy * nz; applyView(); updateCursor();
}
new ResizeObserver(() => { fit(sc); fit($("outCanvas")); }).observe($("desk"));
$("srcView").addEventListener("wheel", ev => { if (!src) return; ev.preventDefault(); zoomAt(ev.deltaY < 0 ? 1.2 : 1 / 1.2, ev.clientX, ev.clientY); }, { passive: false });
$("zIn").onclick = () => zoomAt(1.5); $("zOut").onclick = () => zoomAt(1 / 1.5); $("zFit").onclick = () => { zoom = 1; panX = panY = 0; applyView(); };
let spaceDown = false, altDown = false;
const typing = () => /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName) && document.activeElement.type !== "range" && document.activeElement.type !== "checkbox";
addEventListener("keydown", ev => {
  if (ev.code === "Space" && !typing()) { spaceDown = true; $("srcView").classList.add("panning"); if (ev.target === document.body || ev.target.closest?.("#srcView")) ev.preventDefault(); }
  if (ev.key === "Alt") { altDown = true; ev.preventDefault(); updateCursor(); }
  if (!ev.ctrlKey && !ev.metaKey && !ev.altKey && !typing() && src) { const t = { v: "select", b: "brush", l: "lasso", w: "wand" }[ev.key.toLowerCase()]; if (t) setTool(t); if (ev.key === "x") setMode(editMode === "add" ? "erase" : "add"); }
  if ((ev.ctrlKey || ev.metaKey) && !typing()) {
    if (ev.key.toLowerCase() === "z" && !ev.shiftKey) { ev.preventDefault(); undo(); }
    else if (ev.key.toLowerCase() === "y" || (ev.key.toLowerCase() === "z" && ev.shiftKey)) { ev.preventDefault(); redo(); }
  }
});
addEventListener("keyup", ev => { if (ev.code === "Space") { spaceDown = false; $("srcView").classList.remove("panning"); } if (ev.key === "Alt") { altDown = false; ev.preventDefault(); updateCursor(); } });
addEventListener("blur", () => { spaceDown = altDown = false; $("srcView").classList.remove("panning"); });
$("ovOp").oninput = () => drawSrc();

// ---------------- image view with overlays ----------------
let ovImg = null;
function drawSrc(rect) {
  if (!src) return;
  const W = src.width, H = src.height; if (sc.width !== W || sc.height !== H) { sc.width = W; sc.height = H; ovImg = null; fit(sc); }
  if (!ovImg || ovImg.width !== W || ovImg.height !== H) { ovImg = new ImageData(W, H); rect = null; }
  const [x0, y0, x1, y1] = rect ? [Math.max(0, rect[0] - 1), Math.max(0, rect[1] - 1), Math.min(W - 1, rect[2] + 1), Math.min(H - 1, rect[3] + 1)] : [0, 0, W - 1, H - 1];
  const d = ovImg.data, sd = src.data, op = +$("ovOp").value / 100;
  const els = elements.map(e => [e, effA(e), edgesOf(e)]);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x; let r = sd[i * 4], g = sd[i * 4 + 1], b = sd[i * 4 + 2];
    for (const [e, A, E] of els) {
      const a = A[i] * op; if (a < 0.02 && !E[i]) continue;
      if (hovered) {
        if (e === hovered) { const k = .55 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; }
        else if (e.kind !== "rem") { const k = .45 * a; r -= r * k; g -= g * k; b -= b * k; }
      } else if (e.kind === "rem") { if (e.checked && ((x + y) & 4)) { const k = .6 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; } }
      else if (e.checked) { const k = .32 * a; r += (e.rgb[0] - r) * k; g += (e.rgb[1] - g) * k; b += (e.rgb[2] - b) * k; }
      else { const k = .5 * a; r -= r * k; g -= g * k; b -= b * k; }
      if (E[i] && op > 0 && (e === hovered || (!hovered && e.checked))) { r += (e.rgb[0] - r) * op; g += (e.rgb[1] - g) * op; b += (e.rgb[2] - b) * op; }
    }
    d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
  }
  sc.getContext("2d").putImageData(ovImg, 0, 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}
function pixelAt(ev, clamp) {
  const r = sc.getBoundingClientRect();
  let x = (ev.clientX - r.left) / r.width * sc.width, y = (ev.clientY - r.top) / r.height * sc.height;
  if (clamp) return [Math.max(0, Math.min(sc.width - 1, x)), Math.max(0, Math.min(sc.height - 1, y))];
  x = Math.floor(x); y = Math.floor(y);
  return x >= 0 && y >= 0 && x < sc.width && y < sc.height ? [x, y] : null;
}
function elementAt(x, y) {
  const i = y * src.width + x; let best = null, bv = 0.15;
  for (const e of elements) { const v = effA(e)[i]; if (e.kind !== "rem" && v > bv) { bv = v; best = e; } }
  return best;
}

// ---------------- tools ----------------
let tool = "select", editMode = "add";
const curMode = () => (altDown ? (editMode === "add" ? "erase" : "add") : editMode);
const TOOL_HINT = {
  select: "Click an element on the image to toggle it.",
  brush: "Paint to add or erase. Alt switches Add/Erase. Edge-aware only paints colours like the one under the brush centre.",
  lasso: "Draw around an area to add or erase it.",
  wand: "Click to add or erase a connected area of similar colour.",
  ai: "Click a part and AI selects it as its own element (first click downloads ~14 MB).",
};
function setTool(t) {
  tool = t; for (const x of $("toolSeg").children) x.classList.toggle("on", x.dataset.tool === t);
  $("editOpts").hidden = t === "select";
  for (const el of document.querySelectorAll("[data-for]")) el.hidden = !el.dataset.for.split(" ").includes(t);
  $("toolHint").textContent = TOOL_HINT[t]; $("srcView").dataset.tool = t; updateCursor(); renderList();
}
function setMode(m) { editMode = m; for (const x of $("modeSeg").children) x.classList.toggle("on", x.dataset.mode === m); updateCursor(); }
$("toolSeg").onclick = ev => { const b = ev.target.closest("button"); if (b) setTool(b.dataset.tool); };
$("modeSeg").onclick = ev => { const b = ev.target.closest("button"); if (b) setMode(b.dataset.mode); };
$("target").onchange = () => { target = $("target").value === "sel" ? "sel" : +$("target").value; renderList(); syncAdjust(); };
const brushSize = () => +$("brSize").value;
$("brSize").oninput = () => { $("brSizeV").textContent = $("brSize").value + " px"; updateCursor(); };
$("brHard").oninput = () => { $("brHardV").textContent = $("brHard").value + "%"; };
$("tol").oninput = () => { $("tolV").textContent = $("tol").value; };
function brushAuto() { if (!src) return; const v = Math.max(4, Math.min(150, Math.round(Math.max(src.width, src.height) / 25))); $("brSize").value = v; $("brSizeV").textContent = v + " px"; }

let lastPtr = null;
function updateCursor() {
  const c = $("brCursor"), show = tool === "brush" && lastPtr && src;
  c.style.display = show ? "" : "none"; if (!show) return;
  c.setAttribute("cx", lastPtr[0]); c.setAttribute("cy", lastPtr[1]); c.setAttribute("r", brushSize() / 2);
  c.setAttribute("stroke", curMode() === "add" ? "#22c55e" : "#ef4444"); c.setAttribute("stroke-width", 1.5 / (sc.getBoundingClientRect().width / sc.width || 1));
}

// which mask a tool edits: an element, or for the whole sticker a "Painted in" / "Painted out" layer
function editTarget(mode) {
  if (target !== "sel") { const e = elements.find(x => x.id === target); if (e) return { e, op: e.kind === "rem" ? (mode === "add" ? "erase" : "add") : mode, comp: e.kind !== "add" && e.kind !== "rem" }; }
  const kind = mode === "add" ? "add" : "rem", name = mode === "add" ? "Painted in" : "Painted out";
  let e = elements.find(x => x.kind === kind && x.name === name);
  if (!e) { e = finishEl({ name, sub: "manual edits", color: mode === "add" ? "#22c55e" : "#dc2626", kind, alpha: new Float32Array(src.width * src.height), checked: true }); elements.push(e); renderList(); }
  e.checked = true;
  return { e, op: "add", comp: false };
}

// ---- history (only the changed rectangle of each mask is stored) ----
const history = [], future = [];
let selGrow = 0, selFeather = 0, noFill = null;
function rectOf(e, r) { const [x0, y0, x1, y1] = r, W = src.width, w = x1 - x0 + 1, o = new Float32Array(w * (y1 - y0 + 1)); for (let y = y0; y <= y1; y++) o.set(e.alpha.subarray(y * W + x0, y * W + x1 + 1), (y - y0) * w); return o; }
function putRect(e, r, data) { const [x0, y0, x1, y1] = r, W = src.width, w = x1 - x0 + 1; for (let y = y0; y <= y1; y++) e.alpha.set(data.subarray((y - y0) * w, (y - y0 + 1) * w), y * W + x0); bump(e); }
function pushHistory(h) { history.push(h); if (history.length > 40) history.shift(); future.length = 0; $("undoBtn").disabled = false; $("redoBtn").disabled = true; }
function applyHist(h, dir) {
  if (h.type === "mask") { for (const it of h.items) if (elements.includes(it.e)) putRect(it.e, h.rect, dir < 0 ? it.before : it.after); if (h.noFill) { const [x0, y0, x1, y1] = h.rect, W = src.width, w = x1 - x0 + 1; for (let y = y0; y <= y1; y++) noFill.set((dir < 0 ? h.noFill.before : h.noFill.after).subarray((y - y0) * w, (y - y0 + 1) * w), y * W + x0); } drawSrc(h.rect); }
  else { const v = dir < 0 ? h.before : h.after; if (h.e === "sel") { selGrow = v.grow; selFeather = v.feather; } else if (elements.includes(h.e)) { h.e.grow = v.grow; h.e.feather = v.feather; } syncAdjust(); drawSrc(); }
  renderSticker(); $("undoBtn").disabled = !history.length; $("redoBtn").disabled = !future.length;
}
function undo() { const h = history.pop(); if (!h) return toast("Nothing to undo."); future.push(h); applyHist(h, -1); toast("Undone."); }
function redo() { const h = future.pop(); if (!h) return toast("Nothing to redo."); history.push(h); applyHist(h, 1); toast("Redone."); }
$("undoBtn").onclick = undo; $("redoBtn").onclick = redo;

// ---- generic mask edit: M (0..1, rect-local) is added to / erased from the target ----
// For detected elements the partition is kept: added pixels are taken from the others, erased pixels go to the background.
function beginEdit(mode) {
  const t = editTarget(mode);
  return { ...t, base: t.e.alpha.slice() };
}
function applyStroke(ed, M, r) { // live: target alpha = f(base, M) inside rect r
  const [x0, y0, x1, y1] = r, W = src.width, w = x1 - x0 + 1, a = ed.e.alpha, b = ed.base;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { const i = y * W + x, m = M[(y - y0) * w + x - x0]; a[i] = ed.op === "add" ? b[i] + (1 - b[i]) * m : b[i] * (1 - m); }
  if (ed.e.grow || ed.e.feather) return; // effective mask is recomputed when the edit is committed
  if (!ed.e._edge || ed.e._edk !== ed.e.ver + ":0:0") edgesOf(ed.e); edgeRect(ed.e, r);
}
function commitEdit(ed, r) {
  if (!r) return;
  const W = src.width, [x0, y0, x1, y1] = r, w = x1 - x0 + 1;
  const items = [], tgt = ed.e;
  const before = new Float32Array(w * (y1 - y0 + 1)); for (let y = y0; y <= y1; y++) before.set(ed.base.subarray(y * W + x0, y * W + x1 + 1), (y - y0) * w);
  items.push({ e: tgt, before, after: rectOf(tgt, r) });
  let nf = null;
  if (ed.comp) {
    const others = elements.filter(e => e !== tgt && e.kind !== "add" && e.kind !== "rem"), bg = elements.find(e => e.kind === "bg");
    const snaps = others.map(e => rectOf(e, r));
    // text printed on a subject: erased letters go back to the subject underneath, not to the background
    let host = null;
    if (tgt.kind === "text" && tgt.box) { let best = 0; const [bx0, by0, bx1, by1] = tgt.box; for (const o of others) { if (o.kind !== "obj") continue; let c = 0; for (let y = Math.max(0, by0 - 3); y <= Math.min(src.height - 1, by1 + 3); y++) for (let x = Math.max(0, bx0 - 3); x <= Math.min(W - 1, bx1 + 3); x++) c += o.alpha[y * W + x]; if (c > best) { best = c; host = o; } } }
    const nfBefore = new Uint8Array(w * (y1 - y0 + 1)); for (let y = y0; y <= y1; y++) nfBefore.set(noFill.subarray(y * W + x0, y * W + x1 + 1), (y - y0) * w);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = y * W + x, d = tgt.alpha[i] - ed.base[i]; if (Math.abs(d) < 1e-4) continue;
      if (d > 0) { let s = 0; for (const o of others) s += o.alpha[i]; const room = 1 - tgt.alpha[i]; if (s > room) { const k = room / s; for (const o of others) o.alpha[i] *= k; } }
      else if (host && lastS && lastS[i] > 153) host.alpha[i] = Math.min(1, host.alpha[i] - d);
      else { if (bg && bg !== tgt) bg.alpha[i] = Math.min(1, bg.alpha[i] - d); if (-d > 0.3) noFill[i] = 1; }
    }
    others.forEach((e, k) => { const after = rectOf(e, r); let ch = false; for (let j = 0; j < after.length && !ch; j++) if (after[j] !== snaps[k][j]) ch = true; if (ch) { items.push({ e, before: snaps[k], after }); bump(e); } });
    const nfAfter = new Uint8Array(w * (y1 - y0 + 1)); for (let y = y0; y <= y1; y++) nfAfter.set(noFill.subarray(y * W + x0, y * W + x1 + 1), (y - y0) * w);
    nf = { before: nfBefore, after: nfAfter };
  }
  bump(tgt);
  pushHistory({ type: "mask", rect: r, items, noFill: nf });
  drawSrc(r); renderStickerSoon();
}
const clampRect = (x0, y0, x1, y1) => [Math.max(0, Math.floor(x0)), Math.max(0, Math.floor(y0)), Math.min(src.width - 1, Math.ceil(x1)), Math.min(src.height - 1, Math.ceil(y1))];
const unionRect = (a, b) => a ? [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])] : b;

// ---- brush ----
let stroke = null; // {ed, S:Float32 full-size stroke coverage, rect, last}
function dab(cx, cy) {
  const W = src.width, p = src.data, r = brushSize() / 2, hard = +$("brHard").value / 100, smart = $("brSmart").checked;
  const tol = +$("tol").value * 2.2, soft = tol * 0.5 + 10;
  const rr = clampRect(cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1);
  // reference colour = average 3×3 under the brush centre
  let R = 0, G = 0, B = 0, c = 0; const ix = Math.round(Math.min(W - 1, Math.max(0, cx))), iy = Math.round(Math.min(src.height - 1, Math.max(0, cy)));
  for (let y = iy - 1; y <= iy + 1; y++) for (let x = ix - 1; x <= ix + 1; x++) { if (x < 0 || y < 0 || x >= W || y >= src.height) continue; const i = (y * W + x) * 4; R += p[i]; G += p[i + 1]; B += p[i + 2]; c++; }
  R /= c; G /= c; B /= c;
  for (let y = rr[1]; y <= rr[3]; y++) for (let x = rr[0]; x <= rr[2]; x++) {
    const dd = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / r; if (dd >= 1) continue;
    let wgt = dd <= hard ? 1 : (1 - dd) / (1 - hard); wgt = wgt * wgt * (3 - 2 * wgt);
    if (smart) { const i = (y * W + x) * 4, cd = Math.hypot(p[i] - R, p[i + 1] - G, p[i + 2] - B); if (cd > tol) wgt *= Math.max(0, 1 - (cd - tol) / soft); }
    const i = y * W + x; if (wgt > stroke.S[i]) stroke.S[i] = wgt;
  }
  return rr;
}
function strokeTo(pt) {
  const [x, y] = pt, last = stroke.last || pt, dist = Math.hypot(x - last[0], y - last[1]), step = Math.max(1, brushSize() * 0.2);
  const nSteps = Math.max(1, Math.ceil(dist / step)); let rr = null;
  for (let k = stroke.last ? 1 : 0; k <= nSteps; k++) { const t = k / nSteps; rr = unionRect(rr, dab(last[0] + (x - last[0]) * t, last[1] + (y - last[1]) * t)); }
  stroke.last = pt; if (!rr) return;
  stroke.rect = unionRect(stroke.rect, rr);
  const w = rr[2] - rr[0] + 1, M = new Float32Array(w * (rr[3] - rr[1] + 1));
  for (let yy = rr[1]; yy <= rr[3]; yy++) M.set(stroke.S.subarray(yy * src.width + rr[0], yy * src.width + rr[2] + 1), (yy - rr[1]) * w);
  applyStroke(stroke.ed, M, rr); drawSrc(rr); renderStickerSoon();
}
// ---- lasso ----
let lasso = null;
function lassoMask(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity; for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const r = clampRect(x0 - 1, y0 - 1, x1 + 1, y1 + 1), w = r[2] - r[0] + 1, h = r[3] - r[1] + 1;
  const c = OffscreenCanvasOr(w, h), ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.beginPath(); pts.forEach(([x, y], k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, x - r[0], y - r[1])); ctx.closePath(); ctx.fillStyle = "#000"; ctx.fill();
  const d = ctx.getImageData(0, 0, w, h).data, M = new Float32Array(w * h); for (let i = 0; i < w * h; i++) M[i] = d[i * 4 + 3] / 255;
  return { M, r };
}
// ---- magic wand: contiguous region of similar colour ----
function wandMask(sx, sy) {
  const W = src.width, H = src.height, p = src.data, tol = +$("tol").value * 2.2, s0 = (sy * W + sx) * 4, R = p[s0], G = p[s0 + 1], B = p[s0 + 2];
  const seen = new Uint8Array(W * H), st = [sy * W + sx]; seen[sy * W + sx] = 1; let x0 = sx, y0 = sy, x1 = sx, y1 = sy;
  while (st.length) {
    const j = st.pop(), x = j % W, y = (j / W) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    for (const k of [x > 0 ? j - 1 : -1, x < W - 1 ? j + 1 : -1, j - W, j + W]) { if (k < 0 || k >= W * H || seen[k]) continue; const i = k * 4; if (Math.hypot(p[i] - R, p[i + 1] - G, p[i + 2] - B) <= tol) { seen[k] = 1; st.push(k); } }
  }
  const r = clampRect(x0 - 1, y0 - 1, x1 + 1, y1 + 1), w = r[2] - r[0] + 1, h = r[3] - r[1] + 1, M = new Float32Array(w * h);
  for (let y = r[1]; y <= r[3]; y++) for (let x = r[0]; x <= r[2]; x++) M[(y - r[1]) * w + x - r[0]] = seen[y * W + x];
  // soften the jaggies a little
  const Ms = boxBlurF(M, w, h, 1); for (let i = 0; i < M.length; i++) M[i] = Math.max(M[i], Ms[i] > 0.5 ? Ms[i] : 0);
  return { M, r };
}
function applyOnce(mode, M, r) { const ed = beginEdit(mode); applyStroke(ed, M, r); commitEdit(ed, r); }

// ---- pointer handling on the image ----
let pan = null;
sc.addEventListener("contextmenu", ev => { if (tool !== "select") ev.preventDefault(); });
sc.addEventListener("pointerdown", ev => {
  if (!src) return;
  if (spaceDown || ev.button === 1) { pan = { x: ev.clientX, y: ev.clientY, px: panX, py: panY }; sc.setPointerCapture(ev.pointerId); ev.preventDefault(); return; }
  if (ev.button !== 0 || busy && tool === "ai") return;
  if (ev.altKey !== altDown) { altDown = ev.altKey; }
  const p = pixelAt(ev, true); if (!p) return;
  if (tool === "brush") { stroke = { ed: beginEdit(curMode()), S: new Float32Array(src.width * src.height), rect: null, last: null }; sc.setPointerCapture(ev.pointerId); strokeTo(p); }
  else if (tool === "lasso") { lasso = { pts: [p], mode: curMode() }; sc.setPointerCapture(ev.pointerId); drawLasso(); }
});
sc.addEventListener("pointermove", ev => {
  if (pan) { panX = pan.px + ev.clientX - pan.x; panY = pan.py + ev.clientY - pan.y; if (zoom === 1) panX = panY = 0; applyView(); return; }
  if (!src) return;
  lastPtr = pixelAt(ev, true); updateCursor();
  if (stroke) { for (const e of (ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev])) strokeTo(pixelAt(e, true)); return; }
  if (lasso) { lasso.pts.push(lastPtr); drawLasso(); return; }
  const q = pixelAt(ev); const e = q && elements.length ? elementAt(...q) : null;
  setHover(tool === "select" ? e : null);
  const tip = $("tip"); if (!q) { tip.style.display = "none"; return; }
  const vr = $("srcView").getBoundingClientRect(), m = curMode() === "add" ? "add" : "erase";
  tip.textContent = tool === "select" ? (e ? `${e.name} — click to ${e.checked ? "remove" : "add"}` : "") : tool === "ai" ? (m === "add" ? "Click to add this part" : "Click to cut this part out") : tool === "wand" ? `Click to ${m} similar colours` : "";
  tip.style.display = tip.textContent ? "block" : "none"; tip.style.left = (ev.clientX - vr.left + 14) + "px"; tip.style.top = (ev.clientY - vr.top + 10) + "px";
});
const endPointer = ev => {
  if (pan) { pan = null; return; }
  if (stroke) { const s = stroke; stroke = null; commitEdit(s.ed, s.rect); }
  if (lasso) { const l = lasso; lasso = null; drawLasso(); if (l.pts.length > 2) { const { M, r } = lassoMask(l.pts); applyOnce(l.mode, M, r); } }
};
sc.addEventListener("pointerup", endPointer); sc.addEventListener("pointercancel", endPointer);
sc.addEventListener("pointerleave", () => { if (!stroke) { lastPtr = null; updateCursor(); } setHover(null); $("tip").style.display = "none"; });
function drawLasso() {
  const pl = $("lassoPath");
  if (!lasso) { pl.setAttribute("points", ""); return; }
  pl.setAttribute("points", lasso.pts.map(p => p.join(",")).join(" ")); pl.setAttribute("stroke", lasso.mode === "add" ? "#22c55e" : "#ef4444");
  pl.setAttribute("stroke-width", 2 / (sc.getBoundingClientRect().width / sc.width || 1));
}
sc.addEventListener("click", async ev => {
  if (!src || pan) return; const p = pixelAt(ev); if (!p) return;
  if (tool === "select") { const e = elementAt(...p); if (!e) return; setChecked(e, !e.checked); renderList(); drawSrc(); renderSticker(); return; }
  if (tool === "wand") { const { M, r } = wandMask(p[0], p[1]); applyOnce(ev.altKey ? (editMode === "add" ? "erase" : "add") : editMode, M, r); return; }
  if (tool === "ai") { if (busy) return toast("Please wait for the current step to finish."); await samRefine(p[0], p[1], curMode() === "add"); }
});

// ---- grow / shrink and feather for the edit target ----
function adjTarget() { return target === "sel" ? null : elements.find(e => e.id === target); }
function syncAdjust() {
  const e = adjTarget(), g = e ? e.grow : selGrow, f = e ? e.feather : selFeather;
  $("grow").value = g; $("growV").textContent = (g > 0 ? "+" : "") + g + " px"; $("feather").value = f; $("featherV").textContent = f + " px";
  $("adjFor").textContent = e ? e.name : "whole sticker";
}
let adjStart = null;
for (const id of ["grow", "feather"]) {
  $(id).addEventListener("pointerdown", () => { const e = adjTarget(); adjStart = e ? { grow: e.grow, feather: e.feather } : { grow: selGrow, feather: selFeather }; });
  $(id).addEventListener("keydown", () => { if (!adjStart) { const e = adjTarget(); adjStart = e ? { grow: e.grow, feather: e.feather } : { grow: selGrow, feather: selFeather }; } });
  $(id).oninput = () => {
    const e = adjTarget(), g = +$("grow").value, f = +$("feather").value;
    if (!adjStart) adjStart = e ? { grow: e.grow, feather: e.feather } : { grow: selGrow, feather: selFeather };
    if (e) { e.grow = g; e.feather = f; } else { selGrow = g; selFeather = f; }
    syncAdjust(); drawSrcSoon(); renderStickerSoon();
  };
  $(id).onchange = () => {
    const e = adjTarget(), after = e ? { grow: e.grow, feather: e.feather } : { grow: selGrow, feather: selFeather };
    if (adjStart && (adjStart.grow !== after.grow || adjStart.feather !== after.feather)) pushHistory({ type: "adj", e: e || "sel", before: adjStart, after });
    adjStart = null; drawSrc(); renderSticker();
  };
}
let dsT = 0; function drawSrcSoon() { clearTimeout(dsT); dsT = setTimeout(() => drawSrc(), 60); }

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
    const e = finishEl({ name: add ? `Added area ${cnt}` : `Removed area ${cnt}`, sub: "AI click", color: add ? PALETTE[(elements.length + 2) % PALETTE.length] : "#dc2626", kind: add ? "add" : "rem", alpha: A, checked: true });
    elements.push(e); renderList(); drawSrc(); renderSticker();
    statusDone(add ? "Area added." : "Area removed from the sticker.");
  } catch (e) {
    if (e.message !== "Cancelled") status("Click-select failed: " + e.message, null, { err: true });
  } finally { setBusy(false); }
}
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
  const keepText = $("keepText").checked, texts = keepText ? elements.filter(e => e.kind === "text") : [];
  const k = srcKey + ":" + style + ":" + texts.map(e => e.id + "." + e.ver).join(","); if (styleCache.key === k) return styleCache.img;
  let base = styleCache.baseKey === srcKey + ":" + style ? styleCache.base : (style === "original" ? src : applyStyle(src, style));
  let img = base;
  if (base !== src && texts.length) { // captions stay readable: blend original pixels back under the text masks
    img = new ImageData(new Uint8ClampedArray(base.data), base.width, base.height);
    const d = img.data, s = src.data;
    for (const e of texts) { const a = e.alpha; for (let i = 0; i < a.length; i++) if (a[i] > 0) { const t = a[i]; for (let c = 0; c < 3; c++) d[i * 4 + c] = d[i * 4 + c] * (1 - t) + s[i * 4 + c] * t; } }
  }
  styleCache = { key: k, img, base, baseKey: srcKey + ":" + style }; return img;
}
let rsT = 0; function renderStickerSoon() { clearTimeout(rsT); rsT = setTimeout(renderSticker, 180); }
let renderQueued = false;
function renderSticker() { if (!renderQueued) { renderQueued = true; requestAnimationFrame(() => { renderQueued = false; try { doRender(); } catch (e) { console.error(e); status("Rendering failed: " + e.message, null, { err: true }); } }); } }
function selectionAlpha() {
  const W = src.width, H = src.height, n = W * H, A = new Float32Array(n);
  for (const e of elements) if (e.checked && e.kind !== "rem") { const a = effA(e); for (let i = 0; i < n; i++) A[i] += a[i]; }
  for (let i = 0; i < n; i++) A[i] = Math.min(1, A[i]);
  // pixels the user deliberately left out (unticked objects/text, erased or removed areas) are never "filled"
  const protect = noFill ? noFill.slice() : new Uint8Array(n);
  for (const e of elements) if ((!e.checked && e.kind !== "bg" && e.kind !== "add") || (e.checked && e.kind === "rem")) { const a = effA(e); for (let i = 0; i < n; i++) if (a[i] > 0.5) protect[i] = 1; }
  fillHoles(A, W, H, n * 0.02, protect);
  for (const e of elements) if (e.checked && e.kind === "rem") { const a = effA(e); for (let i = 0; i < n; i++) A[i] *= 1 - a[i]; }
  let out = selGrow || selFeather ? morph(A, W, H, selGrow, selFeather) : A;
  for (let i = 0; i < n; i++) out[i] *= src.data[i * 4 + 3] / 255;
  cleanIslands(out, W, H);
  return out;
}
// Drop tiny disconnected bits (and faint haze far from the cutout) so the border isn't drawn around specks.
function cleanIslands(A, W, H) {
  const n = W * H, bin = new Uint8Array(n); let tot = 0;
  for (let i = 0; i < n; i++) if (A[i] > 0.43) { bin[i] = 1; tot++; }
  if (!tot) return;
  const minA = Math.max(12, tot * 0.004), lab = new Int32Array(n), st = []; let id = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] || !bin[i]) continue; id++; const px = []; st.push(i); lab[i] = id;
    while (st.length) { const j = st.pop(); px.push(j); const x = j % W; for (const k of [x > 0 ? j - 1 : -1, x < W - 1 ? j + 1 : -1, j - W, j + W]) { if (k < 0 || k >= n || lab[k] || !bin[k]) continue; lab[k] = id; st.push(k); } }
    if (px.length < minA) for (const j of px) bin[j] = 0;
  }
  const d = edt(bin, W, H);
  for (let i = 0; i < n; i++) if (!bin[i] && d[i] > 4) A[i] = 0; // nothing faint more than 2 px away from a solid part
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
window.__3sticker.state = () => ({ w: src?.width, h: src?.height, busy, enhanced, elements: elements.map(e => ({ id: e.id, name: e.name, kind: e.kind, checked: e.checked, area: effA(e).reduce((s, v) => s + (v > 0.5), 0) })), target, tool, history: history.length, future: future.length });
window.__3sticker.selectionArea = () => { const A = selectionAlpha(); let c = 0; for (const v of A) c += v > 0.5; return c; };
window.__3sticker.maskArea = (name, box) => { const e = elements.find(x => x.name === name); if (!e) return -1; const a = effA(e), W = src.width; let c = 0; const [x0, y0, x1, y1] = box || [0, 0, W - 1, src.height - 1]; for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) c += a[y * W + x] > 0.5; return c; };
window.__3sticker.setChecked = (names) => { for (const e of elements) e.checked = names.includes(e.name); renderList(); drawSrc(); renderSticker(); };
window.__3sticker.textBoxes = () => elements.filter(e => e.kind === "text").map(e => ({ name: e.name, sub: e.sub, box: e.box }));
window.__3sticker.pixelOf = name => { const e = elements.find(x => x.name === name), a = effA(e), W = src.width; let best = -1, bv = 0.5; const [x0, y0, x1, y1] = e.box; const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2; let bd = 1e9; for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { const i = y * W + x; let others = 0; for (const o of elements) if (o !== e && o.kind !== "bg" && effA(o)[i] > 0.3) others = 1; if (a[i] > 0.8 && !others) { const d = (x - cx) ** 2 + (y - cy) ** 2; if (d < bd) { bd = d; best = i; } } } return best < 0 ? null : [best % W, Math.floor(best / W)]; };
window.__3sticker.setTarget = name => { const e = elements.find(x => x.name === name); target = e ? e.id : "sel"; renderList(); syncAdjust(); };
window.__3sticker.toImage = (x, y) => { const r = sc.getBoundingClientRect(); return [r.left + (x + 0.5) * r.width / sc.width, r.top + (y + 0.5) * r.height / sc.height]; };
setTool("select"); setMode("add");
window.__3sticker.loadFile = loadFile;
renderSticker();
document.body.dataset.ready = "1";
