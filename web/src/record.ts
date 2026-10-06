import { SignAvatar } from "@static/avatar.js";
import { applyOutfit } from "@static/outfits.js";
import { outfitFor } from "./components/avatars";
import type { PoseFrames } from "./types";

declare global { interface Window { __ready?: boolean; __count?: number; __error?: string; __frame?: (i: number) => string; __probe?: (i: number) => Record<string, unknown> } }

(async () => {
  try {
    const p = new URLSearchParams(location.search);
    const model = p.get("model")!, poseUrl = p.get("pose")!;
    const canvas = document.getElementById("c") as HTMLCanvasElement;
    const av = await new SignAvatar(canvas).load("/static/" + model);
    applyOutfit(av, outfitFor(model));
    av.renderer.setClearColor(0x0b1d33, 1);                  // the site's panel colour, not transparent
    const frames: PoseFrames = await (await fetch(poseUrl)).json();
    av.setFrames(frames);
    window.__count = frames.frames.length;
    window.__frame = (i: number) => { av.render(i / frames.fps, 1 / frames.fps); return canvas.toDataURL("image/jpeg", 0.9); };
    // bone positions after posing frame i, for scripts/avatars/retarget_eval.py (the pose is rendered once more first,
    // so the smoothing has settled; rendering twice per frame is what the recorder does at 2x playback anyway)
    window.__probe = (i: number) => { av.render(i / frames.fps, 1 / frames.fps); av.render(i / frames.fps, 1 / frames.fps); return av.probe(); };
    (window as unknown as { __avatar: unknown }).__avatar = av;  // diagnostics (scripts/avatars/*)
    window.__ready = true;
  } catch (e) {
    window.__error = String(e);
  }
})();
