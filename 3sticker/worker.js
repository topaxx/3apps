// 3sticker ML worker: runs all models off the main thread (transformers.js + onnxruntime-web).
import * as T from "./vendor/transformers.min.js";
const { env, AutoModel, AutoProcessor, AutoImageProcessor, SamModel, RawImage, Tensor } = T;

const BASE = new URL("./", import.meta.url).href;
env.allowLocalModels = true;
env.allowRemoteModels = false;              // everything is self-hosted next to the app
env.localModelPath = BASE + "models/";
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = BASE + "vendor/";
if (self.crossOriginIsolated) env.backends.onnx.wasm.numThreads = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
else env.backends.onnx.wasm.numThreads = 1;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
let curJob = 0;
const progress = (stage, extra) => post({ type: "progress", job: curJob, stage, ...extra });

// Model download progress -> UI
function dlCb(stage) {
  const files = {};
  return p => {
    if (p.status === "progress" && p.total) {
      files[p.file] = [p.loaded, p.total];
      let l = 0, t = 0; for (const k in files) { l += files[k][0]; t += files[k][1]; }
      progress(stage, { phase: "download", loaded: l, total: t });
    }
  };
}

let gpuOk = null;
async function hasWebGPU() {
  if (gpuOk !== null) return gpuOk;
  try { gpuOk = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { gpuOk = false; }
  return gpuOk;
}

const SO = { enableCpuMemArena: false, enableMemPattern: false }; // keep wasm memory low so several models fit
const cache = {};
async function load(key, stage, fn) {
  if (!cache[key]) cache[key] = fn(dlCb(stage)).catch(e => { delete cache[key]; throw e; });
  return cache[key];
}
const getRMBG = () => load("rmbg", "rmbg", async cb => [
  await AutoModel.from_pretrained("briaai/RMBG-1.4", { dtype: "q8", device: "wasm", session_options: SO, progress_callback: cb }),
  await AutoProcessor.from_pretrained("briaai/RMBG-1.4")]);
const getDETR = () => load("detr", "detr", async cb => [
  await AutoModel.from_pretrained("Xenova/detr-resnet-50-panoptic", { dtype: "q8", device: "wasm", session_options: SO, progress_callback: cb }),
  await AutoImageProcessor.from_pretrained("Xenova/detr-resnet-50-panoptic")]);
const getOCR = () => load("ocr", "text", async cb => AutoModel.from_pretrained("ppocr-det", { dtype: "fp32", device: "wasm", session_options: SO, progress_callback: cb }));
const getSAM = () => load("sam", "sam", async cb => [
  await SamModel.from_pretrained("Xenova/slimsam-77-uniform", { dtype: "q8", device: "wasm", session_options: SO, progress_callback: cb }),
  await AutoProcessor.from_pretrained("Xenova/slimsam-77-uniform")]);
const SR = { 2: "Xenova/swin2SR-lightweight-x2-64", 4: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr" };
let srDevice = {};
const getSR = (scale, useGpu) => load("sr" + scale + (useGpu ? "g" : "c"), "upscale", async cb => {
  const device = useGpu && await hasWebGPU() ? "webgpu" : "wasm";
  const m = await AutoModel.from_pretrained(SR[scale], { dtype: "fp32", device, progress_callback: cb });
  srDevice[scale] = device; return m;
});

const toRaw = img => new RawImage(new Uint8ClampedArray(img.data), img.width, img.height, 4).rgb();

// ---------- background / salient object (RMBG-1.4) ----------
async function saliency(img) {
  progress("rmbg", { phase: "load" });
  const [model, proc] = await getRMBG();
  progress("rmbg", { phase: "run" });
  const raw = toRaw(img);
  const { pixel_values } = await proc(raw);
  const out = await model({ input: pixel_values });
  const t = out[Object.keys(out)[0]];
  const m = await RawImage.fromTensor(t[0].mul(255).to("uint8")).resize(img.width, img.height);
  return new Uint8Array(m.data.buffer.slice(0)).subarray(0, img.width * img.height);
}

// ---------- labelled objects (DETR panoptic) ----------
async function panoptic(img) {
  progress("detr", { phase: "load" });
  const [model, proc] = await getDETR();
  progress("detr", { phase: "run" });
  const raw = toRaw(img);
  proc.size = { shortest_edge: 512, longest_edge: 640 }; // DETR's mask head is memory-hungry; this is plenty for listing objects
  const inputs = await proc(raw);
  const out = await model(inputs);
  const res = proc.post_process_panoptic_segmentation(out, 0.5, 0.5, 0.8, null, [[img.height, img.width]])[0];
  const seg = res.segmentation.data, n = img.width * img.height;
  const things = [];
  for (const s of res.segments_info) {
    const label = model.config.id2label[s.label_id] || "";
    if (!label || label.startsWith("LABEL_")) continue; // "stuff" (sky, wall, …) is background
    const mask = new Uint8Array(n); let area = 0;
    for (let i = 0; i < n; i++) if (seg[i] === s.id) { mask[i] = 255; area++; }
    if (area) things.push({ label, score: s.score, area, mask });
  }
  return things;
}

// ---------- text (PP-OCR DB text detector + glyph colour clustering) ----------
async function textMask(img) {
  progress("text", { phase: "load" });
  const W = img.width, H = img.height, n = W * H, px = img.data;
  let boxes = [];
  try {
    const model = await getOCR();
    progress("text", { phase: "run" });
    const sc = Math.min(1600 / Math.max(W, H), Math.max(1, 960 / Math.max(W, H)));
    const w = Math.max(32, Math.round(W * sc / 32) * 32), h = Math.max(32, Math.round(H * sc / 32) * 32);
    const r = await toRaw(img).resize(w, h); const rgb = r.data;
    const d = new Float32Array(3 * w * h), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) d[c * w * h + i] = (rgb[i * 3 + 2 - c] / 255 - mean[c]) / std[c]; // BGR like PaddleOCR
    const out = await model({ x: new Tensor("float32", d, [1, 3, h, w]) });
    const prob = out[Object.keys(out)[0]].data;
    boxes = components(prob, w, h, 0.3).filter(c => c.mean > 0.5 && c.area > 12).map(c => {
      const per = 2 * ((c.x1 - c.x0 + 1) + (c.y1 - c.y0 + 1));
      const dd = c.area * 1.6 / per;     // DB "unclip"
      return { x0: Math.max(0, (c.x0 - dd) * W / w), y0: Math.max(0, (c.y0 - dd) * H / h), x1: Math.min(W - 1, (c.x1 + 1 + dd) * W / w), y1: Math.min(H - 1, (c.y1 + 1 + dd) * H / h) };
    }).filter(b => (b.x1 - b.x0) > 4 && (b.y1 - b.y0) > 3);
  } catch (e) {
    progress("text", { phase: "fallback", note: String(e && e.message || e) });
    boxes = heuristicBoxes(img);
  }
  const mask = new Uint8Array(n);
  for (const b of boxes) glyphs(img, b, mask);
  return { mask, boxes };
}

function components(prob, w, h, thr) {
  const lab = new Int32Array(w * h), out = []; let id = 0; const st = [];
  for (let i = 0; i < w * h; i++) {
    if (lab[i] || prob[i] <= thr) continue;
    id++; let c = { x0: w, y0: h, x1: 0, y1: 0, area: 0, sum: 0 }; st.push(i); lab[i] = id;
    while (st.length) {
      const j = st.pop(), x = j % w, y = (j / w) | 0;
      c.area++; c.sum += prob[j]; if (x < c.x0) c.x0 = x; if (x > c.x1) c.x1 = x; if (y < c.y0) c.y0 = y; if (y > c.y1) c.y1 = y;
      for (const k of [j - 1, j + 1, j - w, j + w]) { if (k < 0 || k >= w * h || lab[k] || prob[k] <= thr) continue; if ((k === j - 1 && x === 0) || (k === j + 1 && x === w - 1)) continue; lab[k] = id; st.push(k); }
    }
    c.mean = c.sum / c.area; c.px = i % w; c.py = (i / w) | 0; out.push(c);
  }
  return out;
}

// Fallback when the text model can't load: saturated/high-contrast caption bands near top/bottom.
function heuristicBoxes(img) {
  const W = img.width, H = img.height, p = img.data, m = new Float32Array(W * H);
  for (let y = 0; y < H; y++) { if (y > H * 0.3 && y < H * 0.7) continue; for (let x = 0; x < W; x++) { const i = (y * W + x) * 4, r = p[i], g = p[i + 1], b = p[i + 2]; const mx = Math.max(r, g, b), mn = Math.min(r, g, b); const sat = mx ? (mx - mn) / mx : 0; if ((mx > 200 && sat > 0.45) || (mn > 235)) m[y * W + x] = 1; } }
  const k = Math.max(2, Math.round(Math.min(W, H) / 40)); const cl = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let v = 0; for (let dx = -k * 2; dx <= k * 2 && !v; dx++) { const xx = x + dx; if (xx >= 0 && xx < W && m[y * W + xx]) v = 1; } cl[y * W + x] = v; }
  return components(cl, W, H, 0.5).filter(c => (c.x1 - c.x0) > W * 0.2 && (c.x1 - c.x0) > 2.5 * (c.y1 - c.y0) && c.area > W * H * 0.004).map(c => ({ x0: c.x0, y0: Math.max(0, c.y0 - k), x1: c.x1, y1: Math.min(H - 1, c.y1 + k) }));
}

// Inside a text box, cluster colours and keep the clusters that are rare on the box border (= glyphs).
function glyphs(img, b, mask) {
  const W = img.width, H = img.height, p = img.data;
  const x0 = Math.floor(b.x0), y0 = Math.floor(b.y0), x1 = Math.ceil(b.x1), y1 = Math.ceil(b.y1);
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, ring = Math.max(1, Math.round(Math.min(bw, bh) * 0.12));
  const idx = []; for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) idx.push(y * W + x);
  const K = 4, cent = [];
  for (let k = 0; k < K; k++) { const i = idx[Math.floor((k + 0.5) * idx.length / K * 0.999)] * 4; cent.push([p[i], p[i + 1], p[i + 2]]); }
  // k-means++-ish init: spread by luminance
  idx.slice().sort((a, c) => lum(p, a) - lum(p, c)).forEach((v, j, arr) => { if (j % Math.max(1, Math.floor(arr.length / K)) === 0) { const k = Math.min(K - 1, Math.floor(j / (arr.length / K))); const i = v * 4; cent[k] = [p[i], p[i + 1], p[i + 2]]; } });
  const asg = new Uint8Array(idx.length);
  for (let it = 0; it < 8; it++) {
    const s = cent.map(() => [0, 0, 0, 0]);
    idx.forEach((v, j) => { const i = v * 4; let bk = 0, bd = 1e9; for (let k = 0; k < K; k++) { const c = cent[k], d = (p[i] - c[0]) ** 2 + (p[i + 1] - c[1]) ** 2 + (p[i + 2] - c[2]) ** 2; if (d < bd) { bd = d; bk = k; } } asg[j] = bk; const q = s[bk]; q[0] += p[i]; q[1] += p[i + 1]; q[2] += p[i + 2]; q[3]++; });
    s.forEach((q, k) => { if (q[3]) cent[k] = [q[0] / q[3], q[1] / q[3], q[2] / q[3]]; });
  }
  const inC = new Array(K).fill(0), brC = new Array(K).fill(0); let inN = 0, brN = 0;
  idx.forEach((v, j) => { const x = v % W, y = (v / W) | 0; const border = x - x0 < ring || x1 - x < ring || y - y0 < ring || y1 - y < ring; if (border) { brC[asg[j]]++; brN++; } else { inC[asg[j]]++; inN++; } });
  let keep = [];
  for (let k = 0; k < K; k++) { const fi = inC[k] / Math.max(1, inN), fb = brC[k] / Math.max(1, brN); if (fi > 0.08 && fb < fi * 0.45) keep.push(k); }
  if (!keep.length) { let bk = 0, br = 1e9; for (let k = 0; k < K; k++) { const fi = inC[k] / Math.max(1, inN), fb = brC[k] / Math.max(1, brN); if (fi > 0.05 && fb / fi < br) { br = fb / fi; bk = k; } } keep = [bk]; }
  const local = new Uint8Array(bw * bh);
  idx.forEach((v, j) => { if (keep.includes(asg[j])) { const x = v % W - x0, y = ((v / W) | 0) - y0; local[y * bw + x] = 1; } });
  // grow a little to include outlines / anti-aliasing, then write back
  const g = Math.max(1, Math.round(Math.min(W, H) / 120));
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    let on = 0; for (let dy = -g; dy <= g && !on; dy++) for (let dx = -g; dx <= g; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < bw && yy < bh && local[yy * bw + xx] && dx * dx + dy * dy <= g * g + 1) { on = 1; break; } }
    if (on) mask[(y + y0) * W + x + x0] = 255;
  }
}
const lum = (p, v) => p[v * 4] * 0.3 + p[v * 4 + 1] * 0.59 + p[v * 4 + 2] * 0.11;

// ---------- SlimSAM click refinement ----------
let samState = null; // {key, emb, inputs}
async function samClick(img, key, x, y) {
  progress("sam", { phase: "load" });
  const [model, proc] = await getSAM();
  const raw = toRaw(img);
  if (!samState || samState.key !== key) {
    progress("sam", { phase: "run", note: "embedding" });
    const inputs = await proc(raw);
    samState = { key, emb: await model.get_image_embeddings(inputs), inputs };
  }
  progress("sam", { phase: "run" });
  const pts = await proc(raw, { input_points: [[[x, y]]], input_labels: [[1]] });
  const out = await model({ ...samState.emb, input_points: pts.input_points, input_labels: pts.input_labels });
  const masks = await proc.post_process_masks(out.pred_masks, pts.original_sizes, pts.reshaped_input_sizes);
  const iou = out.iou_scores.data, n = img.width * img.height;
  // prefer the best-scoring mask, but skip near-empty ones
  let best = 0, bs = -1;
  for (let k = 0; k < 3; k++) { let a = 0; for (let i = 0; i < n; i++) a += masks[0].data[k * n + i] ? 1 : 0; const s = iou[k] - (a < n * 0.0005 ? 1 : 0); if (s > bs) { bs = s; best = k; } }
  const m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = masks[0].data[best * n + i] ? 255 : 0;
  return m;
}

// ---------- subject completion: RMBG is often unsure about bodies/clothes; ask SlimSAM with points in those areas ----------
const dbg = {}; let lastPts = null;
async function subjectRefine(img, key, sal, text, hint) {
  const W = img.width, H = img.height, n = W * H;
  const inText = new Uint8Array(n);
  for (const b of text.boxes) for (let y = Math.floor(b.y0); y <= Math.ceil(b.y1) && y < H; y++) for (let x = Math.floor(b.x0); x <= Math.ceil(b.x1) && x < W; x++) inText[y * W + x] = 1;
  let cx = 0, cy = 0, cn = 0;
  for (let i = 0; i < n; i++) if (sal[i] > 204 && !inText[i]) { cx += i % W; cy += (i / W) | 0; cn++; }
  dbg.core = cn / n; if (cn < n * 0.01) return null;
  const hazy = new Float32Array(n); let hn = 0;
  for (let i = 0; i < n; i++) if (sal[i] > 45 && sal[i] < 190 && !inText[i]) { hazy[i] = 1; hn++; }
  dbg.hazy = hn / n;
  // erode away thin anti-aliased fringes, keep solid unsure regions
  const er = Math.max(2, Math.round(Math.min(W, H) / 60));
  const comps = hn < n * 0.03 ? [] : components(densityF(hazy, W, H, er), W, H, 0.55).filter(c => c.area > n * 0.008).sort((a, b) => b.area - a.area).slice(0, 3);
  dbg.comps = comps.map(c => [c.area / n, c.px, c.py]);
  let pts = [], labels = [], fromHint = false;
  if (comps.length) {
    // positive: most salient point near the core centroid
    cx /= cn; cy /= cn; let best = -1, bd = 1e18;
    for (let i = 0; i < n; i++) if (sal[i] > 204) { const d = (i % W - cx) ** 2 + (((i / W) | 0) - cy) ** 2; if (d < bd) { bd = d; best = i; } }
    pts.push([best % W, (best / W) | 0]); labels.push(1);
    for (const c of comps) { pts.push([c.px, c.py]); labels.push(1); }
    for (const b of text.boxes) { pts.push([(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]); labels.push(0); }
  } else if (hint && hint.pts.length) { pts = hint.pts; labels = hint.labels; fromHint = true; } // same points as before enhancing
  else return null;
  lastPts = { pts, labels };
  progress("sam", { phase: "load" });
  const [model, proc] = await getSAM();
  const raw = toRaw(img);
  if (!samState || samState.key !== key) { progress("sam", { phase: "run", note: "subject" }); const inputs = await proc(raw); samState = { key, emb: await model.get_image_embeddings(inputs), inputs }; }
  progress("sam", { phase: "run", note: "subject" });
  const q = await proc(raw, { input_points: [pts], input_labels: [labels] });
  const out = await model({ ...samState.emb, input_points: q.input_points, input_labels: q.input_labels });
  const masks = await proc.post_process_masks(out.pred_masks, q.original_sizes, q.reshaped_input_sizes);
  const iou = out.iou_scores.data; let bk = 0; for (let k = 1; k < 3; k++) if (iou[k] > iou[bk]) bk = k;
  const m = new Uint8Array(n); let a = 0;
  for (let i = 0; i < n; i++) if (masks[0].data[bk * n + i] && (fromHint || sal[i] > 20) && !text.mask[i]) { m[i] = 255; a++; }
  dbg.sam = a / n; dbg.pts = pts; return a > n * 0.01 && a < n * 0.9 ? m : null;
}
function densityF(a, W, H, r) { // local fraction of "on" pixels (box filter)
  const I = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) { let row = 0; for (let x = 0; x < W; x++) { row += a[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; } }
  const o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const x0 = Math.max(0, x - r), y0 = Math.max(0, y - r), x1 = Math.min(W, x + r + 1), y1 = Math.min(H, y + r + 1);
    o[y * W + x] = (I[y1 * (W + 1) + x1] - I[y0 * (W + 1) + x1] - I[y1 * (W + 1) + x0] + I[y0 * (W + 1) + x0]) / ((x1 - x0) * (y1 - y0)) * (a[y * W + x] ? 1 : 0.999);
  }
  return o;
}

// ---------- Swin2SR super-resolution, tiled ----------
async function upscale(img, scale, useGpu) {
  progress("upscale", { phase: "load" });
  let model;
  try { model = await getSR(scale, useGpu); }
  catch (e) { if (!useGpu) throw e; model = await getSR(scale, false); }
  const W = img.width, H = img.height, S = scale, p = img.data;
  const TILE = srDevice[scale] === "webgpu" ? 192 : 96, OV = 8;
  const OW = W * S, OH = H * S, out = new Uint8ClampedArray(OW * OH * 4);
  const xs = [], ys = [];
  for (let x = 0; x < W; x += TILE) xs.push(x); for (let y = 0; y < H; y += TILE) ys.push(y);
  const total = xs.length * ys.length; let done = 0;
  progress("upscale", { phase: "run", done, total, device: srDevice[scale] });
  for (const ty of ys) for (const tx of xs) {
    const ax = Math.max(0, tx - OV), ay = Math.max(0, ty - OV), bx = Math.min(W, tx + TILE + OV), by = Math.min(H, ty + TILE + OV);
    let tw = bx - ax, th = by - ay; const pw = Math.ceil(tw / 8) * 8, ph = Math.ceil(th / 8) * 8;
    const d = new Float32Array(3 * pw * ph);
    for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
      const sx = ax + Math.min(x, tw - 1), sy = ay + Math.min(y, th - 1), i = (sy * W + sx) * 4, o = y * pw + x;
      d[o] = p[i] / 255; d[pw * ph + o] = p[i + 1] / 255; d[2 * pw * ph + o] = p[i + 2] / 255;
    }
    let res;
    try { res = await model({ pixel_values: new Tensor("float32", d, [1, 3, ph, pw]) }); }
    catch (e) {
      if (srDevice[scale] !== "webgpu") throw e;
      model = await getSR(scale, false); res = await model({ pixel_values: new Tensor("float32", d, [1, 3, ph, pw]) });
    }
    const r = res.reconstruction || res[Object.keys(res)[0]], rd = r.data, rw = r.dims[3], rh = r.dims[2];
    // copy the non-overlap part
    for (let y = ty * S; y < Math.min(H, ty + TILE) * S; y++) for (let x = tx * S; x < Math.min(W, tx + TILE) * S; x++) {
      const lx = x - ax * S, ly = y - ay * S; if (lx >= rw || ly >= rh) continue;
      const o = ly * rw + lx, q = (y * OW + x) * 4;
      out[q] = rd[o] * 255; out[q + 1] = rd[rw * rh + o] * 255; out[q + 2] = rd[2 * rw * rh + o] * 255; out[q + 3] = 255;
    }
    done++; progress("upscale", { phase: "run", done, total, device: srDevice[scale] });
  }
  // keep any source transparency (nearest-neighbour)
  for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) out[(y * OW + x) * 4 + 3] = p[(((y / S) | 0) * W + ((x / S) | 0)) * 4 + 3];
  return { data: out, width: OW, height: OH, device: srDevice[scale] };
}

self.onmessage = async ({ data: m }) => {
  curJob = m.job;
  try {
    if (m.type === "analyze") {
      const img = m.img;
      const sal = await saliency(img);
      let things = [], tErr = null;
      try { if (!m.skipDetr) things = await panoptic(img); } catch (e) { tErr = String(e.message || e); }
      const text = await textMask(img);
      let subj = null;
      lastPts = null; try { subj = await subjectRefine(img, m.key, sal, text, m.hint); } catch (e) { console.warn("subject refine failed", e); }
      const tr = [sal.buffer, text.mask.buffer, ...things.map(t => t.mask.buffer)]; if (subj) tr.push(subj.buffer);
      post({ type: "result", job: m.job, sal, subj, subjPts: subj ? lastPts : null, things, text, warn: tErr, dbg }, tr);
    } else if (m.type === "sam") {
      const mask = await samClick(m.img, m.key, m.x, m.y);
      post({ type: "result", job: m.job, mask }, [mask.buffer]);
    } else if (m.type === "upscale") {
      const r = await upscale(m.img, m.scale, m.gpu);
      post({ type: "result", job: m.job, img: { data: r.data, width: r.width, height: r.height }, device: r.device }, [r.data.buffer]);
    } else if (m.type === "ping") {
      post({ type: "result", job: m.job, isolated: self.crossOriginIsolated, threads: env.backends.onnx.wasm.numThreads, gpu: await hasWebGPU() });
    }
  } catch (e) {
    post({ type: "error", job: m.job, message: String(e && e.message || e) });
  }
};
post({ type: "ready" });
