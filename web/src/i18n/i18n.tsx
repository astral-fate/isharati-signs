import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import type { Lang } from "../types";
import ar from "./ar.json";
import en from "./en.json";
import tr from "./tr.json";
import ur from "./ur.json";

const STRINGS: Record<Lang, Record<string, string>> = { en, ar, tr, ur };
export const LANG_LIST: Lang[] = ["en", "ar", "tr", "ur"];
export const RTL = new Set<Lang>(["ar", "ur"]);
const KEY = "isharati.ui";

interface I18n { ui: Lang; setUi: (l: Lang) => void; t: (key: string) => string; dir: "ltr" | "rtl" }
const Ctx = createContext<I18n | null>(null);

function detect(): Lang {
  const isLang = (x: unknown): x is Lang => LANG_LIST.includes(x as Lang);
  const param = new URLSearchParams(location.search).get("ui");
  if (isLang(param)) return param;
  try { const saved = localStorage.getItem(KEY); if (isLang(saved)) return saved; } catch { /* private mode */ }
  const nav = (navigator.language || "en").slice(0, 2);
  return isLang(nav) ? nav : "en";
}

export function I18nProvider({ children, initial }: { children: ReactNode; initial?: Lang }) {
  const [ui, setUi] = useState<Lang>(initial ?? detect());
  const dir = RTL.has(ui) ? "rtl" : "ltr";
  useEffect(() => {
    document.documentElement.lang = ui;
    document.documentElement.dir = dir;
    try { localStorage.setItem(KEY, ui); } catch { /* private mode */ }
  }, [ui, dir]);
  const t = useCallback((key: string) => STRINGS[ui][key] ?? STRINGS.en[key] ?? key, [ui]);
  return <Ctx.Provider value={{ ui, setUi, t, dir }}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error("useI18n outside I18nProvider");
  return v;
}
