import { motion } from "framer-motion";
import { useI18n } from "../i18n/i18n";

// "What would you like to do?": one card per product, each with a small preview of the real screen and one button,
// so a first-time visitor sees in a glance everything Isharati offers and where to start.
const CARDS = [
  { id: "ask", href: "#try", poster: "/clips/rocketbox_female_06_en.jpg", badge: true },
  { id: "studio", href: "#studio", poster: "/clips/rocketbox_male_21_ar.jpg" },
  { id: "mushaf", href: "/#/quran" },
  { id: "academy", href: "/#/academy", badge: true },
] as const;

function Preview({ id, poster }: { id: string; poster?: string }) {
  const { t } = useI18n();
  if (id === "ask")
    return (
      <div className="ft-prev ft-ask">
        {poster && <img src={poster} alt="" loading="lazy" />}
        <div className="ft-bubbles">
          <span className="ft-b q">{t("ft.ask.q")}</span>
          <span className="ft-b a">{t("ft.ask.a")}<em>📖 {t("ft.ask.src")}</em></span>
        </div>
      </div>
    );
  if (id === "studio")
    return (
      <div className="ft-prev ft-studio">
        {poster && <img src={poster} alt="" loading="lazy" />}
        <div className="ft-chips"><span>🎙️ {t("ft.studio.c1")}</span><span>🎬 {t("ft.studio.c2")}</span><span>⌨️ {t("ft.studio.c3")}</span></div>
        <div className="ft-caption">{t("ft.studio.cap")}</div>
      </div>
    );
  if (id === "mushaf")
    return (
      <div className="ft-prev ft-mushaf">
        <div className="ft-ayah" dir="rtl">
          {["بِسْمِ", "اللَّهِ", "الرَّحْمَٰنِ", "الرَّحِيمِ"].map((w, i) => <span key={w} className={i === 1 ? "on" : ""}>{w}</span>)}
          <b>۝١</b>
        </div>
        <div className="ft-tafsir"><i>✋</i>{t("ft.mushaf.tafsir")}</div>
      </div>
    );
  return (
    <div className="ft-prev ft-acad">
      <div className="ft-path">
        {["👋", "🤲", "🕋", "📿"].map((e, i) => <span key={e} className={i === 0 ? "done" : i === 1 ? "now" : ""}>{e}</span>)}
      </div>
      <div className="ft-quiz">
        <small>{t("ft.academy.qq")}</small>
        <div><span className="ok">✓ {t("ft.academy.o1")}</span><span>{t("ft.academy.o2")}</span></div>
      </div>
      <div className="ft-xp">🔥 3 · ⭐ 120 XP</div>
    </div>
  );
}

export function Features() {
  const { t } = useI18n();
  return (
    <section id="features" className="ft">
      <div className="wrap">
        <h2 className="ft-h">{t("ft.title")}</h2>
        <p className="ft-lead">{t("ft.lead")}</p>
        <div className="ft-grid">
          {CARDS.map((c, i) => (
            <motion.a key={c.id} href={c.href} className="ft-card"
                      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                      transition={{ duration: 0.5, delay: i * 0.08 }}>
              {"badge" in c && c.badge && <span className="ft-ribbon">{t(`ft.${c.id}.badge`)}</span>}
              <Preview id={c.id} poster={"poster" in c ? c.poster : undefined} />
              <h3>{t(`ft.${c.id}.title`)}</h3>
              <p>{t(`ft.${c.id}.desc`)}</p>
              <ul>{[1, 2, 3].map((k) => <li key={k}>{t(`ft.${c.id}.p${k}`)}</li>)}</ul>
              <span className="btn btn-primary ft-cta">{t(`ft.${c.id}.cta`)} →</span>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}

// The Academy, shown on the landing page: how learning works, and one click into the first lesson
export function AcademyPromo() {
  const { t } = useI18n();
  const pillars = [
    { icon: "📚", k: "lessons" }, { icon: "🎯", k: "quizzes" }, { icon: "🃏", k: "cards" }, { icon: "🌍", k: "langs" },
  ];
  return (
    <section id="academy-promo" className="ap">
      <div className="wrap ap-in">
        <div className="ap-copy">
          <span className="ft-kicker">🎓 {t("ap.kicker")}</span>
          <h2>{t("ap.title")}</h2>
          <p>{t("ap.lead")}</p>
          <div className="ap-cta">
            <a className="btn btn-primary" href="/#/academy">{t("ap.cta")} →</a>
            <span className="ap-free">{t("ap.free")}</span>
          </div>
        </div>
        <div className="ap-grid">
          {pillars.map((p) => (
            <a key={p.k} href="/#/academy" className="ap-card">
              <i>{p.icon}</i>
              <b>{t(`ap.${p.k}.title`)}</b>
              <span>{t(`ap.${p.k}.desc`)}</span>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
