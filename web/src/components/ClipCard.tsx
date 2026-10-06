import { useInView } from "framer-motion";
import { useEffect, useRef } from "react";
import type { Clip } from "../types";

const touch = typeof matchMedia !== "undefined" && matchMedia("(hover: none)").matches;

/** An avatar's signing clip: plays on hover, or when scrolled into view on touch screens. */
export function ClipCard({ clip }: { clip: Clip }) {
  const ref = useRef<HTMLVideoElement>(null);
  const inView = useInView(ref, { amount: 0.6 });
  useEffect(() => {
    if (!touch || !ref.current) return;
    if (inView) ref.current.play().catch(() => {}); else ref.current.pause();
  }, [inView]);
  return (
    <video ref={ref} muted loop playsInline preload="metadata" poster={clip.poster}
           onMouseEnter={(e) => e.currentTarget.play().catch(() => {})} onMouseLeave={(e) => e.currentTarget.pause()}>
      <source src={clip.webm} type="video/webm" /><source src={clip.mp4} type="video/mp4" />
    </video>
  );
}
