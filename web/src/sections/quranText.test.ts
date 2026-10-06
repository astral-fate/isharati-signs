import { describe, expect, it } from "vitest";
import { getAyahText, getAyahTextSync } from "./quranText";

describe("quranText", () => {
  it("provides instant fallback for Al-Fatihah 1:1 and 1:2", () => {
    const a1 = getAyahTextSync(1, 1);
    expect(a1?.uthmani).toContain("بِسْمِ ٱللَّهِ");
    expect(a1?.surahName).toBe("سُورَةُ ٱلْفَاتِحَةِ");

    const a2 = getAyahTextSync(1, 2);
    expect(a2?.uthmani).toContain("ٱلْحَمْدُ لِلَّهِ");
  });

  it("provides instant fallback for Al-Ikhlas 112:1", () => {
    const a1 = getAyahTextSync(112, 1);
    expect(a1?.uthmani).toContain("قُلْ هُوَ ٱللَّهُ أَحَدٌ");
  });

  it("returns null for unknown ayah when db not loaded yet", () => {
    expect(getAyahTextSync(999, 1)).toBeNull();
  });
});
