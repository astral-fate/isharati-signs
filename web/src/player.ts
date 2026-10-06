import { useCallback, useEffect, useRef, useState } from "react";

/** A media clock for poses: the avatar, the keypoints and the gloss chips all read t (seconds). */
export function usePlayer(duration: number, opts: { loop?: boolean; autoplay?: boolean } = {}) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(!!opts.autoplay);
  const [speed, setSpeed] = useState(0.75);
  const tRef = useRef(0);
  useEffect(() => { tRef.current = 0; setT(0); setPlaying(!!opts.autoplay); }, [duration, opts.autoplay]);
  useEffect(() => {
    if (!playing || duration <= 0) return;
    let raf = 0, last = performance.now();
    const step = (now: number) => {
      let next = tRef.current + (Math.min(0.1, (now - last) / 1000)) * speed;
      last = now;
      if (next >= duration) {
        if (opts.loop) next %= duration;
        else { next = duration; setPlaying(false); }
      }
      tRef.current = next;
      setT(next);
      if (next < duration || opts.loop) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, duration, speed, opts.loop]);
  const seek = useCallback((s: number) => { if (!Number.isFinite(s)) return; tRef.current = Math.max(0, Math.min(duration, s)); setT(tRef.current); },
                           [duration]);
  const play = useCallback(() => { if (tRef.current >= duration) seek(0); setPlaying(true); }, [duration, seek]);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, play, pause]);
  return { t, playing, speed, play, pause, toggle, seek, setSpeed };
}
