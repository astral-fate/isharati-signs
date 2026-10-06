import type { Segment } from "../types";

export function GlossChips({ segments, time, onSeek }: { segments: Segment[]; time: number; onSeek: (s: number) => void }) {
  return (
    <div className="chips">
      {segments.map((s, i) => {
        const now = time >= s.start_s && time < s.end_s;
        return (
          <button key={i} type="button" className={"chip" + (s.kind === "missing" ? " missing" : "") + (now ? " now" : "")}
                  onClick={() => onSeek(s.start_s)}>{s.label}</button>
        );
      })}
    </div>
  );
}
