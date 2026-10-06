import { useEffect, useRef, useState } from "react";
import heroPose from "../assets/hero-pose.json";
import { KeypointCanvas } from "./KeypointCanvas";
import { useI18n } from "../i18n/i18n";
import type { PoseFrames } from "../types";

const pose = heroPose as PoseFrames;
const DWELL_MS = 3600;

const STEP_KEYS = [
  {
    num: "01",
    layerKey: "pipeline.step.1.layer",
    tagKey: "pipeline.step.1.tag",
    titleKey: "pipeline.step.1.title",
    descKey: "pipeline.step.1.desc",
    metaKey: "pipeline.step.1.meta",
  },
  {
    num: "02",
    layerKey: "pipeline.step.2.layer",
    tagKey: "pipeline.step.2.tag",
    titleKey: "pipeline.step.2.title",
    descKey: "pipeline.step.2.desc",
    metaKey: "pipeline.step.2.meta",
  },
  {
    num: "03",
    layerKey: "pipeline.step.3.layer",
    tagKey: "pipeline.step.3.tag",
    titleKey: "pipeline.step.3.title",
    descKey: "pipeline.step.3.desc",
    metaKey: "pipeline.step.3.meta",
  },
  {
    num: "04",
    layerKey: "pipeline.step.4.layer",
    tagKey: "pipeline.step.4.tag",
    titleKey: "pipeline.step.4.title",
    descKey: "pipeline.step.4.desc",
    metaKey: "pipeline.step.4.meta",
  },
];

export function PipelineStack() {
  const { t, dir } = useI18n();
  const [active, setActive] = useState(0);
  const [time, setTime] = useState(0);
  const audioCanvasRef = useRef<HTMLCanvasElement>(null);
  const ropeCanvasRef = useRef<HTMLCanvasElement>(null);

  // Auto-cycle through layers
  useEffect(() => {
    const id = setInterval(() => {
      setActive((prev) => (prev + 1) % STEP_KEYS.length);
    }, DWELL_MS);
    return () => clearInterval(id);
  }, []);

  // Drive animation time for the real keypoint trajectory (Layer 3)
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const duration = pose.frames.length / pose.fps;
    function loop(now: number) {
      const elapsed = (now - start) / 1000;
      setTime(elapsed % duration);
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Layer 1: Animated Audio Waveform Spectrum
  useEffect(() => {
    const canvas = audioCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let t = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.scale(dpr, dpr);

    function draw() {
      if (!ctx || !canvas) return;
      t += 0.04;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      // Background grid
      ctx.strokeStyle = "rgba(47, 216, 239, 0.08)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 24) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Draw frequency spectrum bars
      const numBars = 32;
      const barW = (w - 40) / numBars;
      for (let i = 0; i < numBars; i++) {
        const freq = Math.sin(t * 2 + i * 0.3) * 0.5 + 0.5;
        const barH = Math.max(6, freq * (h * 0.42));
        const x = 20 + i * barW;
        const y = h * 0.55 - barH / 2;

        const grad = ctx.createLinearGradient(0, y, 0, y + barH);
        grad.addColorStop(0, "#50ff4a");
        grad.addColorStop(1, "#2fd8ef");
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, barW - 3, barH);
      }

      // Continuous soundwave line
      ctx.beginPath();
      ctx.strokeStyle = "#e8fffa";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#2fd8ef";
      ctx.shadowBlur = 10;
      for (let x = 0; x < w; x += 4) {
        const y = h * 0.55 + Math.sin(x * 0.04 + t * 3) * 16 * Math.cos(x * 0.015);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Layer 2: Animated RoPE Attention Phasor Rotation
  useEffect(() => {
    const canvas = ropeCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let theta = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.scale(dpr, dpr);

    function draw() {
      if (!ctx || !canvas) return;
      theta += 0.025;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      const cx = w * 0.5;
      const cy = h * 0.5;
      const r = Math.min(w, h) * 0.36;

      // Concentric orbital rings
      ctx.strokeStyle = "rgba(160, 180, 225, 0.15)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.5, 0, Math.PI * 2);
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();

      // Rotating phasor vectors
      const numVectors = 8;
      for (let i = 0; i < numVectors; i++) {
        const angle = theta + (i * Math.PI * 2) / numVectors;
        const vx = cx + Math.cos(angle) * r;
        const vy = cy + Math.sin(angle) * r;

        ctx.strokeStyle = i % 2 === 0 ? "rgba(80, 255, 74, 0.45)" : "rgba(47, 216, 239, 0.45)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(vx, vy);
        ctx.stroke();

        ctx.fillStyle = i % 2 === 0 ? "#50ff4a" : "#2fd8ef";
        ctx.beginPath();
        ctx.arc(vx, vy, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // Attention connection arcs between adjacent nodes
      ctx.strokeStyle = "rgba(232, 214, 187, 0.5)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let i = 0; i < numVectors; i++) {
        const a1 = theta + (i * Math.PI * 2) / numVectors;
        const a2 = theta + (((i + 2) % numVectors) * Math.PI * 2) / numVectors;
        ctx.moveTo(cx + Math.cos(a1) * r, cy + Math.sin(a1) * r);
        ctx.lineTo(cx + Math.cos(a2) * r, cy + Math.sin(a2) * r);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Centre core
      ctx.fillStyle = "#0b1231";
      ctx.strokeStyle = "#50ff4a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("RoPE", cx, cy);

      raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="pipeline-stack-container" dir={dir}>
      {/* 3D Exploded Isometric Holographic Stage */}
      <div className="lp-scene">
        <div className="lp-planes">
          {/* Layer 01: Speech & Tokenization Live Spectrum */}
          <div
            className={`lp-plane ${active >= 0 ? "shown" : ""} ${active === 0 ? "live" : ""}`}
            style={{ "--i": 0 } as any}
          >
            <div className="lp-plane-tag">{t("pipeline.step.1.layer")}</div>
            <div className="layer-plane-inner" style={{ position: "relative", width: "100%", height: "100%" }}>
              <canvas
                ref={audioCanvasRef}
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: 12,
                  left: 12,
                  right: 12,
                  display: "flex",
                  gap: 8,
                  alignItems: "center",
                  flexWrap: "wrap",
                  zIndex: 2,
                }}
              >
                <span className="token-chip">«ٱلسَّلَامُ» <small>[0.0s - 0.4s]</small></span>
                <span className="token-chip">«عَلَيْكُمْ» <small>[0.4s - 0.9s]</small></span>
                <span className="hud-badge">99.8% CONF</span>
              </div>
            </div>
          </div>

          {/* Layer 02: RoPE Attention Phasor Rotation & Token Heatmap */}
          <div
            className={`lp-plane ${active >= 1 ? "shown" : ""} ${active === 1 ? "live" : ""}`}
            style={{ "--i": 1 } as any}
          >
            <div className="lp-plane-tag">{t("pipeline.step.2.layer")}</div>
            <div className="layer-plane-inner" style={{ position: "relative", width: "100%", height: "100%" }}>
              <canvas
                ref={ropeCanvasRef}
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 10,
                  right: 12,
                  fontSize: "10px",
                  fontFamily: "monospace",
                  color: "#50ff4a",
                  background: "rgba(4, 8, 26, 0.75)",
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid rgba(80, 255, 74, 0.3)",
                }}
              >
                cos(θ) · d=512 · h=8
              </div>
            </div>
          </div>

          {/* Layer 03: Actual Real 25 FPS 50-Joint Kinematic Keypoints Canvas */}
          <div
            className={`lp-plane ${active >= 2 ? "shown" : ""} ${active === 2 ? "live" : ""}`}
            style={{ "--i": 2 } as any}
          >
            <div className="lp-plane-tag">{t("pipeline.step.3.layer")}</div>
            <div className="layer-plane-inner" style={{ position: "relative", width: "100%", height: "100%" }}>
              <KeypointCanvas key={`kp-layer-${active === 2 ? "active" : "standby"}`} frames={pose} time={time} />
              <div
                style={{
                  position: "absolute",
                  top: 10,
                  right: 12,
                  fontSize: "10px",
                  fontFamily: "monospace",
                  color: "#7ef0d3",
                  background: "rgba(4, 8, 26, 0.75)",
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid rgba(126, 240, 211, 0.3)",
                  zIndex: 2,
                }}
              >
                FRAME: {(Math.round(time * 25) % 120) + 1}/120 · 25 FPS
              </div>
            </div>
          </div>

          {/* Layer 04: Real Rendered 3D Avatar Video / Blender Viewport */}
          <div
            className={`lp-plane ${active >= 3 ? "shown" : ""} ${active === 3 ? "live" : ""}`}
            style={{ "--i": 3 } as any}
          >
            <div className="lp-plane-tag">{t("pipeline.step.4.layer")}</div>
            <div className="layer-plane-inner" style={{ position: "relative", width: "100%", height: "100%" }}>
              <video
                src="/clips/rocketbox_male_21_ar.webm"
                autoPlay
                loop
                muted
                playsInline
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                  filter: "contrast(1.05) brightness(1.05)",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: 10,
                  right: 12,
                  fontSize: "10px",
                  fontFamily: "monospace",
                  color: "#fff",
                  background: "rgba(4, 8, 26, 0.75)",
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid rgba(160, 180, 225, 0.3)",
                  zIndex: 2,
                }}
              >
                SAUDI_THOBE · 3D VRM 1.0
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step Navigation Sidebar */}
      <div className="lp-steps-panel">
        <div className="lp-kicker">
          <i />
          <span>{t("pipeline.kicker")}</span>
        </div>
        <ul className="lp-steps" role="tablist">
          {STEP_KEYS.map((step, idx) => {
            const isCur = active === idx;
            return (
              <li key={idx}>
                <button
                  type="button"
                  className="lp-step"
                  aria-current={isCur}
                  onClick={() => setActive(idx)}
                >
                  <span className="n">{step.num}</span>
                  <div>
                    <h4>{t(step.titleKey)}</h4>
                    <p>{t(step.descKey)}</p>
                    <div className="src">{t(step.metaKey)}</div>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="lp-progress">
          <i key={active} style={{ animationDuration: `${DWELL_MS}ms` }} />
        </div>
      </div>
    </div>
  );
}
