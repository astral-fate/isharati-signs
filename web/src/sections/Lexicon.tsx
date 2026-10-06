import { motion } from "framer-motion";
import { Counter } from "../components/Counter";
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Lexicon() {
  const { t } = useI18n();
  const { stats } = useAppState();
  return (
    <section id="lexicon">
      <div className="wrap">
        <h2>{t("lex.title")}</h2>
        <p className="lead">{t("lex.lead")}</p>
        <div className="langs">
          {stats.languages.map((l) => (
            <motion.div key={l.code} className="lang panel" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}>
              <p className="label"><bdi dir="ltr">{l.name} → {l.sign_language}</bdi></p>
              <div className="big"><Counter value={l.signs} /></div>
              <div>{t("lex.signs")} · <Counter value={l.passages} /> {t("lex.passages")}</div>
              <div className="bar">
                <motion.div initial={{ width: 0 }} whileInView={{ width: `${l.coverage * 100}%` }} viewport={{ once: true }}
                            transition={{ duration: 1.4, ease: "easeOut" }} />
              </div>
              <div><b><Counter value={l.coverage} format="pct" /></b> {t("lex.coverage")}</div>
            </motion.div>
          ))}
        </div>
        <p className="note">{t("lex.note")}</p>
      </div>
    </section>
  );
}
