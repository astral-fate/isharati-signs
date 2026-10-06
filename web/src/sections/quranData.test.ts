import { describe, expect, it } from "vitest";
import { AYAH_COUNTS, cellKind, embedUrl, wordAt } from "./quranData";
import type { QuranSurahRow } from "../types";

const none = { recitation: 0, tafsir: 0 };
const rec  = { recitation: 1, tafsir: 0 };
const taf  = { recitation: 0, tafsir: 1 };
const both = { recitation: 1, tafsir: 1 };

function row(kfc: typeof none, curriculum = none, mukhtasar = none): QuranSurahRow {
  return { surah: 1, name: "al-Fatiha", kfc, curriculum, mukhtasar };
}

describe("cellKind", () => {
  it("none when all zero", () => expect(cellKind(row(none))).toBe("none"));
  it("recitation from kfc", () => expect(cellKind(row(rec))).toBe("recitation"));
  it("tafsir from mukhtasar", () => expect(cellKind(row(none, none, taf))).toBe("tafsir"));
  it("both when kfc has both", () => expect(cellKind(row(both))).toBe("both"));
  it("both when mixed across sources", () => expect(cellKind(row(rec, none, taf))).toBe("both"));
});

describe("wordAt", () => {
  const words = [{ start_s: 0, end_s: 1 }, { start_s: 1, end_s: 2.5 }, { start_s: 2.5, end_s: 4 }];
  it("returns 0 at t=0", () => expect(wordAt(words, 0)).toBe(0));
  it("returns 1 at boundary 1", () => expect(wordAt(words, 1)).toBe(1));
  it("returns 1 mid", () => expect(wordAt(words, 1.9)).toBe(1));
  it("returns 2 at 2.5", () => expect(wordAt(words, 2.5)).toBe(2));
  it("returns -1 past end", () => expect(wordAt(words, 4)).toBe(-1));
  it("returns -1 for empty", () => expect(wordAt([], 0)).toBe(-1));
});

describe("embedUrl", () => {
  it("null without youtube_id", () => expect(embedUrl({})).toBeNull());
  it("null with empty string", () => expect(embedUrl({ youtube_id: "" })).toBeNull());
  it("builds url", () => {
    const u = embedUrl({ youtube_id: "abc123", start: 12.3, end: 45.7 });
    expect(u).toBe("https://www.youtube-nocookie.com/embed/abc123?start=12&end=46&rel=0");
  });
  it("defaults start/end to 0", () => {
    const u = embedUrl({ youtube_id: "abc" });
    expect(u).toBe("https://www.youtube-nocookie.com/embed/abc?start=0&end=0&rel=0");
  });
});

describe("AYAH_COUNTS", () => {
  it("has 114 entries", () => expect(AYAH_COUNTS).toHaveLength(114));
  it("sums to 6236", () => expect(AYAH_COUNTS.reduce((a, b) => a + b, 0)).toBe(6236));
  it("first surah is 7", () => expect(AYAH_COUNTS[0]).toBe(7));
  it("last surah (An-Nas) is 6", () => expect(AYAH_COUNTS[113]).toBe(6));
});
