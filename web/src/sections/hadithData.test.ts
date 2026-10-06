import { describe, expect, it } from "vitest";
import { glossFor, sacredPage, hadithHref, labelKind, readOffset, refFromHash, watchUrl } from "./hadithData";

describe("refFromHash", () => {
  it("reads the deep link", () => expect(refFromHash("#/hadith/nawawi:12")).toBe("nawawi:12"));
  it("reads the Academy alias", () => expect(refFromHash("#/academy/hadith/nawawi:12")).toBe("nawawi:12"));
  it("reads an encoded colon", () => expect(refFromHash("#/academy/hadith/bukhari%3A69")).toBe("bukhari:69"));
  it("null for the hadith page itself", () => expect(refFromHash("#/academy/hadith")).toBeNull());
  it("null for anything that is not a ref", () => expect(refFromHash("#/academy/hadith/../x")).toBeNull());
  it("round-trips hadithHref", () => expect(refFromHash(hadithHref("muslim:2495").slice(1))).toBe("muslim:2495"));
});

describe("sacredPage", () => {
  it("quran", () => expect(sacredPage("#/quran")).toBe("quran"));
  it("the Academy alias", () => expect(sacredPage("#/academy/quran")).toBe("quran"));
  it("hadith with a ref", () => expect(sacredPage("#/hadith/nawawi:12")).toBe("hadith"));
  it("the Academy home", () => expect(sacredPage("#/academy")).toBeNull());
  it("a landing anchor", () => expect(sacredPage("#quran")).toBeNull());
});

describe("watchUrl", () => {
  it("null without a url", () => expect(watchUrl({ youtube_url: null, start: 0 })).toBeNull());
  it("keeps the publisher timestamp", () =>
    expect(watchUrl({ youtube_url: "https://www.youtube.com/watch?v=abc&t=12s", start: 30 })).toBe(
      "https://www.youtube.com/watch?v=abc&t=12s"));
  it("adds the sample start", () =>
    expect(watchUrl({ youtube_url: "https://www.youtube.com/watch?v=abc", start: 42.7 })).toBe(
      "https://www.youtube.com/watch?v=abc&t=42s"));
});

describe("readOffset", () => {
  it("0 when unknown", () => expect(readOffset({ read_start: null, start: 0 })).toBe(0));
  it("relative to the clip start", () => expect(readOffset({ read_start: 65.5, start: 60 })).toBe(5.5));
  it("never negative", () => expect(readOffset({ read_start: 1, start: 5 })).toBe(0));
});

describe("glossFor", () => {
  const ar = [{ text: "خير", oov: false }];
  const tid = [{ text: "namaz", oov: true }];
  it("null without gloss", () => expect(glossFor({ gloss: null, sign_language: "ArSL" })).toBeNull());
  it("the sample's own sign language first", () =>
    expect(glossFor({ gloss: { ArSL: ar, "TİD": tid }, sign_language: "TİD" })).toEqual({ lang: "TİD", items: tid }));
  it("never shows another sign language's gloss", () =>
    expect(glossFor({ gloss: { ArSL: ar }, sign_language: "TİD" })).toBeNull());
});

describe("labelKind", () => {
  it("drops +title", () => expect(labelKind("speech_match+title")).toBe("speech_match"));
  it("keeps a plain label", () => expect(labelKind("translation_match")).toBe("translation_match"));
});
