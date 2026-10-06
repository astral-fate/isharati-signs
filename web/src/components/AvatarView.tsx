import { SignAvatar } from "@static/avatar.js";
import { applyOutfit } from "@static/outfits.js";
import { useEffect, useRef, useState, type RefObject } from "react";
import type { PoseFrames } from "../types";
import { outfitFor, type Outfit } from "./avatars";

/** The 3D avatar (three.js + VRM) driven by pose frames at time t. One canvas; a new model replaces the old one. */
export function AvatarView({ model, outfit = "hijab", frames, time, onError, onReady, canvasRef }: {
  model: string; outfit?: Outfit; frames: PoseFrames | null; time: number; onError: () => void; onReady?: () => void;
  canvasRef?: RefObject<HTMLCanvasElement>;
}) {
  const own = useRef<HTMLCanvasElement>(null);
  const ref = canvasRef ?? own;
  const [av, setAv] = useState<SignAvatar | null>(null);
  const last = useRef(performance.now());
  const current = useRef<SignAvatar | null>(null);  // the live instance; anything else is disposed and must not render
  const draw = (a: SignAvatar | null, t: number, dt?: number) => { if (a && current.current === a) a.render(t, dt); };
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let alive = true, disposed = false;
    const a = new SignAvatar(canvas);
    current.current = a;
    const dispose = () => { if (!disposed) { disposed = true; a.renderer.dispose(); } };
    (async () => {
      try {
        await a.load("/static/" + model);
        if (!alive) return;
        applyOutfit(a, outfitFor(model, outfit));
        setAv(a);
        onReady?.();
      } catch (e) {
        if (!alive) return;
        console.error("avatar", e);
        dispose();
        onError();
      }
    })();
    return () => {
      alive = false;
      dispose();
      if (current.current === a) current.current = null;
      setAv(null);
    };
  }, [model]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { av?.setFrames(frames); draw(av, time, 1 / 30); }, [av, frames]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (av && current.current === av) { applyOutfit(av, outfitFor(model, outfit)); draw(av, time); } }, [av, outfit]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!av) return;
    const now = performance.now();
    draw(av, time, Math.min(0.1, (now - last.current) / 1000));
    last.current = now;
  }, [av, time]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const c = ref.current;
    if (!c || !av) return;
    const ro = new ResizeObserver(() => { if (current.current === av) { av.resize(); draw(av, time); } });
    ro.observe(c);
    return () => ro.disconnect();
  }, [av]);  // eslint-disable-line react-hooks/exhaustive-deps
  return <canvas ref={ref} aria-label="avatar" />;
}
