import type { HadithGloss, HadithSample } from "../types";

/** English names of the collections the dataset labels with (the Arabic name comes from the server). */
export const COLLECTION_EN: Record<string, string> = {
  nawawi: "An-Nawawi's Forty",
  riyad: "Riyad as-Salihin",
  bukhari: "Sahih al-Bukhari",
  muslim: "Sahih Muslim",
  abudawud: "Sunan Abi Dawud",
  tirmidhi: "Jami' at-Tirmidhi",
  nasai: "Sunan an-Nasa'i",
  ibnmajah: "Sunan Ibn Majah",
  malik: "Muwatta Malik",
};

/** "#/hadith/nawawi:12" (or the alias "#/academy/hadith/nawawi:12") -> "nawawi:12"; null for any other hash */
export function refFromHash(hash: string): string | null {
  const m = /^#\/(?:academy\/)?hadith\/([a-z0-9_]+:[0-9]+[a-z]?)$/.exec(decodeURIComponent(hash || ""));
  return m ? m[1] : null;
}

export const hadithHref = (ref: string) => `/#/hadith/${ref}`;

/** The sacred-text pages: "#/quran" -> "quran", "#/hadith[/ref]" -> "hadith" (also under "#/academy/"), else null */
export function sacredPage(hash: string): "quran" | "hadith" | null {
  const m = /^#\/(?:academy\/)?(quran|hadith)(\/|$)/.exec(hash || "");
  return m ? (m[1] as "quran" | "hadith") : null;
}

/** The original video at the sample's own timestamp (youtube_url already carries &t= when the publisher set one). */
export function watchUrl(s: Pick<HadithSample, "youtube_url" | "start">): string | null {
  if (!s.youtube_url) return null;
  if (/[?&]t=/.test(s.youtube_url)) return s.youtube_url;
  const sep = s.youtube_url.includes("?") ? "&" : "?";
  return `${s.youtube_url}${sep}t=${Math.floor(s.start ?? 0)}s`;
}

/**
 * Seconds into the pose where the hadith text itself starts (read_start is in video time; the pose starts at the
 * clip's start). 0 when unknown.
 */
export function readOffset(s: Pick<HadithSample, "read_start" | "start">): number {
  if (s.read_start == null) return 0;
  return Math.max(0, s.read_start - (s.start ?? 0));
}

/** The gloss in the sample's own sign language, else the first one the dataset has (null when none). */
export function glossFor(s: Pick<HadithSample, "gloss" | "sign_language">): { lang: string; items: HadithGloss[] } | null {
  const g = s.gloss;
  if (!g) return null;
  // only the clip's own sign language: an ArSL gloss under a TİD signer would describe signs they never made
  return g[s.sign_language]?.length ? { lang: s.sign_language, items: g[s.sign_language] } : null;
}

/** "speech_match+title" -> "speech_match": the base of how a sample was labelled, for its i18n key */
export const labelKind = (label: string) => label.split("+")[0];
