import { LANG_LIST, useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

const NAMES = { en: "EN", ar: "ع", tr: "TR", ur: "اردو" } as const;

export function Nav() {
  const { t, ui, setUi } = useI18n();
  const { theme, toggleTheme } = useAppState();
  // only separate pages: the landing page's sections are reached from the page itself
  const pages = [
    { href: "/#", key: "nav.home", on: !/^#\/(academy|quran|hadith|review)/.test(location.hash) && location.pathname !== "/review" },
    { href: "/#/academy", key: "nav.academy", on: location.hash.startsWith("#/academy") && !/^#\/academy\/(quran|hadith)/.test(location.hash) },
    { href: "/#/quran", key: "nav.quran", on: /^#\/(academy\/)?quran/.test(location.hash) },
    { href: "/#/hadith", key: "nav.hadith", on: /^#\/(academy\/)?hadith/.test(location.hash) },
    { href: "/#/review", key: "nav.review", on: location.hash.startsWith("#/review") || location.pathname === "/review" },
  ];
  return (
    <nav className="nav">
      <div className="wrap">
        <a className="logo" href="#top">Isharati <span className="grad">إشارتي</span></a>
        <div className="links">
          {pages.map((p) => (
            <a key={p.key} href={p.href} className={p.on ? "on" : undefined} aria-current={p.on ? "page" : undefined}>
              {t(p.key)}
            </a>
          ))}
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div className="seg" role="group" aria-label={t("nav.language")}>
            {LANG_LIST.map((l) => <button key={l} className={l === ui ? "on" : ""} onClick={() => setUi(l)}>{NAMES[l]}</button>)}
          </div>
          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "الوضع الفاتح" : "الوضع الداكن"}
            style={{
              background: "transparent",
              border: "1px solid var(--line)",
              borderRadius: "50%",
              width: 36,
              height: 36,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              fontSize: "16px",
              color: "var(--ink)",
              transition: "all 0.2s ease",
            }}
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
        </div>
      </div>
    </nav>
  );
}
