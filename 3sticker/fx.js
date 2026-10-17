// 3sticker image effects: distance transform + cartoon-like styles (pure canvas/JS, runs on the device).

// Squared Euclidean distance to the nearest pixel where bin[i] === 1 (Felzenszwalb & Huttenlocher).
export function edt(bin, W, H) {
  const INF = 1e20, f = new Float64Array(Math.max(W, H)), d = new Float64Array(Math.max(W, H)), v = new Int32Array(Math.max(W, H)), z = new Float64Array(Math.max(W, H) + 1);
  const out = new Float32Array(W * H);
  const pass = (n) => {
    let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < n; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0; for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
  };
  for (let x = 0; x < W; x++) { for (let y = 0; y < H; y++) f[y] = bin[y * W + x] ? 0 : INF; pass(H); for (let y = 0; y < H; y++) out[y * W + x] = d[y]; }
  for (let y = 0; y < H; y++) { for (let x = 0; x < W; x++) f[x] = out[y * W + x]; pass(W); for (let x = 0; x < W; x++) out[y * W + x] = d[x]; }
  return out;
}

const clamp = v => v < 0 ? 0 : v > 255 ? 255 : v;
function lumOf(d, n) { const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = d[i * 4] * 0.299 + d[i * 4 + 1] * 0.587 + d[i * 4 + 2] * 0.114; return L; }
function boxOnce(src, W, H, r) {
  const t = new Float32Array(W * H), o = new Float32Array(W * H), k = 2 * r + 1;
  for (let y = 0; y < H; y++) { const row = y * W; let s = 0; for (let x = -r; x <= r; x++) s += src[row + Math.min(W - 1, Math.max(0, x))]; for (let x = 0; x < W; x++) { t[row + x] = s / k; s += src[row + Math.min(W - 1, x + r + 1)] - src[row + Math.max(0, x - r)]; } }
  for (let x = 0; x < W; x++) { let s = 0; for (let y = -r; y <= r; y++) s += t[Math.min(H - 1, Math.max(0, y)) * W + x]; for (let y = 0; y < H; y++) { o[y * W + x] = s / k; s += t[Math.min(H - 1, y + r + 1) * W + x] - t[Math.max(0, y - r) * W + x]; } }
  return o;
}
function blur1(a, W, H, r) { // 3x box blur ~ gaussian
  if (r < 1) return a.slice();
  let cur = a; for (let p = 0; p < 3; p++) cur = boxOnce(cur, W, H, r); return cur;
}

// Edge-preserving smoothing (sparse bilateral filter).
function bilateral(d, W, H, r, sr, iters) {
  let cur = new Float32Array(W * H * 3); for (let i = 0; i < W * H; i++) { cur[i * 3] = d[i * 4]; cur[i * 3 + 1] = d[i * 4 + 1]; cur[i * 3 + 2] = d[i * 4 + 2]; }
  const step = Math.max(1, Math.round(r / 3)), taps = [];
  for (let dy = -r; dy <= r; dy += step) for (let dx = -r; dx <= r; dx += step) { const ds = dx * dx + dy * dy; if (ds <= r * r) taps.push([dx, dy, Math.exp(-ds / (2 * (r / 1.5) ** 2))]); }
  const inv = 1 / (2 * sr * sr);
  for (let it = 0; it < iters; it++) {
    const nx = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 3, r0 = cur[i], g0 = cur[i + 1], b0 = cur[i + 2]; let sw = 0, sR = 0, sG = 0, sB = 0;
      for (const [dx, dy, ws] of taps) {
        const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = (yy * W + xx) * 3, dr = cur[j] - r0, dg = cur[j + 1] - g0, db = cur[j + 2] - b0;
        const w = ws * Math.exp(-(dr * dr + dg * dg + db * db) * inv); sw += w; sR += cur[j] * w; sG += cur[j + 1] * w; sB += cur[j + 2] * w;
      }
      nx[i] = sR / sw; nx[i + 1] = sG / sw; nx[i + 2] = sB / sw;
    }
    cur = nx;
  }
  return cur;
}

// k-means palette on a sample, then map every pixel to its nearest colour.
function quantize(c3, n, k) {
  const sample = [], stride = Math.max(1, Math.floor(n / 6000));
  for (let i = 0; i < n; i += stride) sample.push(i);
  const sorted = sample.slice().sort((a, b) => (c3[a * 3] + c3[a * 3 + 1] + c3[a * 3 + 2]) - (c3[b * 3] + c3[b * 3 + 1] + c3[b * 3 + 2]));
  let cent = []; for (let j = 0; j < k; j++) { const i = sorted[Math.floor((j + .5) / k * sorted.length)]; cent.push([c3[i * 3], c3[i * 3 + 1], c3[i * 3 + 2]]); }
  const near = i => { let bk = 0, bd = 1e12; for (let j = 0; j < k; j++) { const c = cent[j], d = (c3[i * 3] - c[0]) ** 2 + (c3[i * 3 + 1] - c[1]) ** 2 + (c3[i * 3 + 2] - c[2]) ** 2; if (d < bd) { bd = d; bk = j; } } return bk; };
  for (let it = 0; it < 10; it++) {
    const s = cent.map(() => [0, 0, 0, 0]);
    for (const i of sample) { const q = s[near(i)]; q[0] += c3[i * 3]; q[1] += c3[i * 3 + 1]; q[2] += c3[i * 3 + 2]; q[3]++; }
    cent = cent.map((c, j) => s[j][3] ? [s[j][0] / s[j][3], s[j][1] / s[j][3], s[j][2] / s[j][3]] : c);
  }
  const out = new Float32Array(n * 3), idx = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const j = near(i); idx[i] = j; out[i * 3] = cent[j][0]; out[i * 3 + 1] = cent[j][1]; out[i * 3 + 2] = cent[j][2]; }
  return { out, idx, cent };
}

// Dark line mask from a luminance image: Sobel magnitude, adaptive (percentile) threshold, optional thickening.
function edges(L, W, H, pct, thick) {
  const n = W * H, g = new Float32Array(n);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    const gx = -L[i - W - 1] - 2 * L[i - 1] - L[i + W - 1] + L[i - W + 1] + 2 * L[i + 1] + L[i + W + 1];
    const gy = -L[i - W - 1] - 2 * L[i - W] - L[i - W + 1] + L[i + W - 1] + 2 * L[i + W] + L[i + W + 1];
    g[i] = Math.hypot(gx, gy);
  }
  const hist = new Uint32Array(1024); for (let i = 0; i < n; i++) hist[Math.min(1023, g[i] | 0)]++;
  let acc = 0, thr = 1023; for (let v = 1023; v >= 0; v--) { acc += hist[v]; if (acc > n * pct) { thr = v; break; } }
  thr = Math.max(thr, 40);
  const e = new Float32Array(n); for (let i = 0; i < n; i++) e[i] = Math.max(0, Math.min(1, (g[i] - thr) / (thr * 0.6)));
  if (thick > 0) { const b = new Uint8Array(n); for (let i = 0; i < n; i++) b[i] = e[i] > 0.5 ? 1 : 0; const d = edt(b, W, H); for (let i = 0; i < n; i++) e[i] = Math.max(e[i], Math.max(0, Math.min(1, thick + 0.5 - Math.sqrt(d[i])))); }
  return e;
}

export function applyStyle(img, style) {
  const W = img.width, H = img.height, n = W * H, d = img.data, out = new ImageData(W, H), o = out.data;
  const S = Math.max(W, H), r = Math.max(2, Math.round(S / 170)), thick = Math.max(0, Math.round(S / 450));
  for (let i = 0; i < n; i++) o[i * 4 + 3] = d[i * 4 + 3];
  if (style === "cartoon") {
    const sm = bilateral(d, W, H, r, 26, 3);
    const { out: q0 } = quantize(sm, n, 16);
    const q = new Float32Array(n * 3); for (let i = 0; i < n * 3; i++) q[i] = q0[i] * 0.78 + sm[i] * 0.22; // flat colours, a hint of shading
    const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = sm[i * 3] * 0.299 + sm[i * 3 + 1] * 0.587 + sm[i * 3 + 2] * 0.114;
    const e = edges(blur1(L, W, H, Math.max(1, r >> 1)), W, H, 0.09, thick);
    for (let i = 0; i < n; i++) { const k = 1 - e[i] * 0.9; for (let c = 0; c < 3; c++) o[i * 4 + c] = clamp(q[i * 3 + c] * 1.05 * k + 6 * k); }
  } else if (style === "comic") {
    const sm = bilateral(d, W, H, r, 30, 1);
    const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = sm[i * 3] * 0.299 + sm[i * 3 + 1] * 0.587 + sm[i * 3 + 2] * 0.114;
    const e = edges(blur1(L, W, H, 1), W, H, 0.1, thick + 0.5);
    const cell = Math.max(3, Math.round(S / 110)); // halftone dots
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, l = L[i] / 255; let v;
      if (l < 0.22) v = 0; else if (l > 0.78) v = 255;
      else { const cx = (Math.floor(x / cell) + .5) * cell, cy = (Math.floor(y / cell) + .5) * cell; const rad = Math.sqrt(1 - (l - 0.22) / 0.56) * cell * 0.62; v = Math.hypot(x - cx, y - cy) < rad ? 0 : 255; }
      v *= 1 - e[i];
      // a touch of warm paper tone so it reads as ink, not a mask
      o[i * 4] = clamp(v * 1.0 + (v > 128 ? 0 : 12)); o[i * 4 + 1] = clamp(v * 0.97 + (v > 128 ? 0 : 10)); o[i * 4 + 2] = clamp(v * 0.9 + (v > 128 ? 0 : 14));
    }
  } else if (style === "pop") {
    const sm = bilateral(d, W, H, r, 30, 1);
    const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = sm[i * 3] * 0.299 + sm[i * 3 + 1] * 0.587 + sm[i * 3 + 2] * 0.114;
    const Lb = blur1(L, W, H, 1);
    const hist = new Uint32Array(256); for (let i = 0; i < n; i++) hist[clamp(Lb[i] | 0)]++;
    const q = []; let acc = 0, k = 1; for (let v = 0; v < 256; v++) { acc += hist[v]; while (k < 4 && acc >= n * k / 4) { q.push(v); k++; } } while (q.length < 3) q.push(255);
    const pal = [[28, 20, 100], [230, 0, 126], [0, 160, 233], [255, 242, 0]];
    const e = edges(Lb, W, H, 0.07, thick + 0.5);
    const cell = Math.max(3, Math.round(S / 90));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, l = Lb[i]; let t = l < q[0] ? 0 : l < q[1] ? 1 : l < q[2] ? 2 : 3; let c = pal[t];
      if (t === 2) { const cx = (Math.floor(x / cell) + .5) * cell, cy = (Math.floor(y / cell) + .5) * cell; if (Math.hypot(x - cx, y - cy) < cell * 0.28) c = [255, 255, 255]; }
      const k2 = 1 - e[i]; o[i * 4] = c[0] * k2; o[i * 4 + 1] = c[1] * k2; o[i * 4 + 2] = c[2] * k2;
    }
  } else if (style === "sketch") {
    const L = lumOf(d, n), inv = new Float32Array(n); for (let i = 0; i < n; i++) inv[i] = 255 - L[i];
    const b = blur1(inv, W, H, Math.max(2, Math.round(S / 90)));
    const e = edges(blur1(L, W, H, 1), W, H, 0.06, 0);
    for (let i = 0; i < n; i++) {
      let v = b[i] >= 255 ? 255 : Math.min(255, L[i] * 255 / (255 - b[i])); // colour dodge
      v = 255 - (255 - v) * 1.35; v *= 1 - e[i] * 0.35; v = clamp(v);
      o[i * 4] = clamp(v * 0.98 + 4); o[i * 4 + 1] = clamp(v * 0.97 + 3); o[i * 4 + 2] = clamp(v * 0.95);
    }
  } else { o.set(d); }
  return out;
}
