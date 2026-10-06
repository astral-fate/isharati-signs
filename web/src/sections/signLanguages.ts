import { ApiError } from "../api";
import { RTL } from "../i18n/i18n";
import type { Lang } from "../types";

export const SIGN_LANGS: { code: Lang; label: string }[] = [
  { code: "en", label: "English → ASL" },
  { code: "ar", label: "العربية → ArSL" },
  { code: "tr", label: "Türkçe → TİD" },
  { code: "ur", label: "اردو → ISL" },
];

// written in the question's language; the English and Arabic ones are already signed and replay instantly
export const EXAMPLES: Record<Lang, string[]> = {
  en: ["What are the pillars of Islam?", "What is Ramadan?", "How many prayers a day?"],
  ar: ["ما هو الإسلام؟", "ما هي الزكاة؟", "كم عدد الصلوات في اليوم؟"],
  tr: ["İslam'ın şartları nelerdir?", "Zekât nedir?", "Hac nedir?"],
  ur: ["اسلام کے ارکان کیا ہیں؟", "نماز کیا ہے؟", "زکوٰۃ کیا ہے؟"],
};

export const textDir = (lang: Lang) => (RTL.has(lang) ? "rtl" : "ltr");

export function errorKey(e: unknown, kind: "ask" | "text" = "ask"): string {
  if (e instanceof ApiError && e.kind === "html") return "try.html";
  if (e instanceof ApiError && e.kind === "quota") return kind === "text" ? "try.quotaStudio" : "try.quota";
  return "try.error";
}
