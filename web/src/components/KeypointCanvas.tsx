import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import type { PoseFrames } from "../types";

// 50 joints: 0 nose, 1 neck, 2-4 right shoulder/elbow/wrist, 5-7 left, 8-28 left hand, 29-49 right hand (MediaPipe)
// Torso & arms (excluding wrist-to-hand connections which must be conditionally drawn)
const TORSO = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7]];
const HAND = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

/**
 * Checks if a 21-joint hand is genuinely active & tracked.
 * Rejects:
 * 1. Zero/missing hands (all [0, 0, 0]).
 * 2. Dummy collapsed points where all 21 joints are identical (spread < 0.02).
 */
export function isHandValid(f: number[][], base: number): boolean {
  if (!f || f.length < base + 21) return false;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let nonZeroCount = 0;
  for (let i = 0; i < 21; i++) {
    const pt = f[base + i];
    if (!pt) return false;
    const x = pt[0], y = pt[1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    if (Math.abs(x) > 1e-4 || Math.abs(y) > 1e-4) nonZeroCount++;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  if (nonZeroCount < 5) return false;
  const spread = Math.max(maxX - minX, maxY - minY);
  return spread >= 0.02;
}

export function poseBounds(frames: number[][][]) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const f of frames) {
    if (!f) continue;
    // Core body joints (0..7)
    for (let i = 0; i < Math.min(8, f.length); i++) {
      const pt = f[i];
      if (!pt) continue;
      const [x, y] = pt;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (i !== 1 && Math.abs(x) < 1e-4 && Math.abs(y) < 1e-4) continue;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    // Left hand (8..28) only if active
    if (isHandValid(f, 8)) {
      for (let i = 8; i < 29; i++) {
        const [x, y] = f[i];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
    // Right hand (29..49) only if active
    if (isHandValid(f, 29)) {
      for (let i = 29; i < 50; i++) {
        const [x, y] = f[i];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  if (maxX - minX < 1e-3) { minX -= 1; maxX += 1; }
  if (maxY - minY < 1e-3) { minY -= 1; maxY += 1; }
  return { minX, maxX, minY, maxY };
}

export function KeypointCanvas({
  frames,
  time,
  canvasRef,
  style,
}: {
  frames: PoseFrames | null;
  time: number;
  canvasRef?: RefObject<HTMLCanvasElement>;
  style?: React.CSSProperties;
}) {
  const own = useRef<HTMLCanvasElement>(null);
  const ref = canvasRef ?? own;
  const box = useMemo(() => (frames?.frames.length ? poseBounds(frames.frames) : null), [frames]);
  const [size, setSize] = useState(0);

  useEffect(() => {
    const c = ref.current;
    if (!c || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setSize((n) => n + 1));
    ro.observe(c);
    return () => ro.disconnect();
  }, [ref]);

  useEffect(() => {
    const c = ref.current;
    if (!c || !frames || !box || !frames.frames.length) return;
    const rect = c.getBoundingClientRect();
    const parentRect = c.parentElement?.getBoundingClientRect();
    const clientW = c.clientWidth || rect.width || (parentRect && parentRect.width) || c.parentElement?.clientWidth || 400;
    const clientH = c.clientHeight || rect.height || (parentRect && parentRect.height) || c.parentElement?.clientHeight || 300;
    const finalW = Math.max(120, clientW);
    const finalH = Math.max(120, clientH);

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(finalW * dpr), h = Math.round(finalH * dpr);
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = c.getContext("2d");
    if (!g) return;

    // 1. Draw solid dark holographic background
    g.fillStyle = "#04081a";
    g.fillRect(0, 0, w, h);

    // 2. Draw subtle holographic coordinate radar grid
    g.strokeStyle = "rgba(47, 216, 239, 0.08)";
    g.lineWidth = 1 * dpr;
    const gridStep = Math.round(28 * dpr);
    g.beginPath();
    for (let gx = 0; gx < w; gx += gridStep) {
      g.moveTo(gx, 0);
      g.lineTo(gx, h);
    }
    for (let gy = 0; gy < h; gy += gridStep) {
      g.moveTo(0, gy);
      g.lineTo(w, gy);
    }
    g.stroke();

    const f = frames.frames[Math.min(frames.frames.length - 1, Math.max(0, Math.round(time * frames.fps)))];
    if (!f) return;

    const pad = 0.12;
    const sx = (w * (1 - 2 * pad)) / (box.maxX - box.minX);
    const sy = (h * (1 - 2 * pad)) / (box.maxY - box.minY);
    const s = Math.min(sx, sy);
    const ox = w / 2 - ((box.minX + box.maxX) / 2) * s;
    const oy = h / 2 - ((box.minY + box.maxY) / 2) * s;

    const P = (i: number) => {
      const pt = f[i];
      if (!pt) return [ox, oy] as const;
      // If neck (index 1) is at exactly (0,0), derive it from shoulder midpoint if available
      if (i === 1) {
        const shR = f[2], shL = f[5];
        if (shR && shL && Number.isFinite(shR[0]) && Number.isFinite(shL[0])) {
          const midX = (shR[0] + shL[0]) / 2;
          const midY = (shR[1] + shL[1]) / 2;
          return [midX * s + ox, midY * s + oy] as const;
        }
      }
      return [pt[0] * s + ox, pt[1] * s + oy] as const;
    };

    const ok = (i: number) => {
      const pt = f[i];
      if (!pt) return false;
      const x = pt[0], y = pt[1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
      // Index 1 is the canonical neck (origin [0, 0, 0] in our coordinate system)
      if (i === 1) return true;
      // For other joints, ignore missing/untracked points at (0,0)
      if (Math.abs(x) < 1e-4 && Math.abs(y) < 1e-4) return false;
      return true;
    };

    g.lineCap = "round";
    g.shadowColor = "#7ef0d3";
    g.shadowBlur = 14 * dpr;

    const lines = (pairs: number[][], base: number, width: number, color: string) => {
      g.strokeStyle = color;
      g.lineWidth = width * dpr;
      g.beginPath();
      for (const [a, b] of pairs) {
        if (!ok(a + base) || !ok(b + base)) continue;
        const [x1, y1] = P(a + base), [x2, y2] = P(b + base);
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
      }
      g.stroke();
    };

    const lhValid = isHandValid(f, 8);
    const rhValid = isHandValid(f, 29);

    // Torso connections (head to neck, neck to shoulders, arms)
    lines(TORSO, 0, 4, "#7ef0d3");

    // Connect right shoulder to left shoulder directly across the chest
    if (ok(2) && ok(5)) {
      lines([[2, 5]], 0, 4, "#7ef0d3");
    }

    if (lhValid) {
      lines([[7, 8]], 0, 3, "#7ef0d3");
      lines(HAND, 8, 2, "#8fb8ff");
    }
    if (rhValid) {
      lines([[4, 29]], 0, 3, "#7ef0d3");
      lines(HAND, 29, 2, "#8fb8ff");
    }

    g.fillStyle = "#e8fffa";
    // Body joints (0..7)
    for (let i = 0; i < 8; i++) {
      if (ok(i)) {
        const [x, y] = P(i);
        g.beginPath();
        g.arc(x, y, (i === 0 ? 4 : 3.5) * dpr, 0, Math.PI * 2);
        g.fill();
      }
    }

    // Left hand joints (8..28) only if active
    if (lhValid) {
      for (let i = 8; i < 29; i++) {
        if (ok(i)) {
          const [x, y] = P(i);
          g.beginPath();
          g.arc(x, y, 2 * dpr, 0, Math.PI * 2);
          g.fill();
        }
      }
    }

    // Right hand joints (29..49) only if active
    if (rhValid) {
      for (let i = 29; i < 50; i++) {
        if (ok(i)) {
          const [x, y] = P(i);
          g.beginPath();
          g.arc(x, y, 2 * dpr, 0, Math.PI * 2);
          g.fill();
        }
      }
    }

    // The signer's face (lips, eyes, irises, brows, nose, outline), when this frame has one; else a head ring
    const fi = Math.min(frames.frames.length - 1, Math.max(0, Math.round(time * frames.fps)));
    const fp = frames.face?.[fi];
    const hasFace = !!fp && !!frames.face_edges && fp.some((q) => q && q[0] != null);
    if (hasFace) {
      const Q = (k: number) => {
        const q = fp![k];
        return q && q[0] != null && q[1] != null ? [q[0] * s + ox, q[1] * s + oy] as const : null;
      };
      g.shadowBlur = 6 * dpr;
      for (const [part, pairs] of Object.entries(frames.face_edges!)) {
        g.strokeStyle = part === "lips" ? "#ff9fb2" : part.endsWith("iris") ? "#ffffff" : "#7ef0d3";
        g.lineWidth = (part === "oval" ? 1.5 : 1.8) * dpr;
        g.beginPath();
        for (const [a, b] of pairs) {
          const qa = Q(a), qb = Q(b);
          if (!qa || !qb) continue;
          g.moveTo(qa[0], qa[1]);
          g.lineTo(qb[0], qb[1]);
        }
        g.stroke();
      }
    }

    if (ok(0) && !hasFace) {
      // Head ring centered around head center
      const [nx, ny] = P(0);
      const [kx, ky] = ok(1) ? P(1) : [nx, ny + 30 * dpr];
      const dist = Math.hypot(nx - kx, ny - ky);
      const hx = nx;
      const hy = ny - dist * 0.15;
      const headRadius = Math.max(16 * dpr, dist * 0.52);

      // Subtle holographic face glow
      g.fillStyle = "rgba(126, 240, 211, 0.12)";
      g.beginPath();
      g.arc(hx, hy, headRadius, 0, Math.PI * 2);
      g.fill();

      // Head outline ring
      g.strokeStyle = "#7ef0d3";
      g.lineWidth = 3 * dpr;
      g.beginPath();
      g.arc(hx, hy, headRadius, 0, Math.PI * 2);
      g.stroke();
    }
  }, [frames, time, box, ref, size]);

  return (
    <canvas
      ref={ref}
      aria-label="keypoints"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        display: "block",
        ...style,
      }}
    />
  );
}

