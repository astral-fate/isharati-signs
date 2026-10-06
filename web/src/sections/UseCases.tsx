import { motion } from "framer-motion";
import { useI18n } from "../i18n/i18n";

export function UseCases() {
  const { t } = useI18n();

  const cases = [
    {
      id: "mosques",
      title: t("usecases.mosques.title"),
      desc: t("usecases.mosques.desc"),
      tag: t("usecases.mosques.tag"),
      icon: "🕌",
      gradient: "from-emerald-500/10 via-emerald-500/5 to-transparent",
    },
    {
      id: "schools",
      title: t("usecases.schools.title"),
      desc: t("usecases.schools.desc"),
      tag: t("usecases.schools.tag"),
      icon: "📚",
      gradient: "from-blue-500/10 via-blue-500/5 to-transparent",
    },
    {
      id: "dawah",
      title: t("usecases.dawah.title"),
      desc: t("usecases.dawah.desc"),
      tag: t("usecases.dawah.tag"),
      icon: "🌍",
      gradient: "from-purple-500/10 via-purple-500/5 to-transparent",
    },
    {
      id: "nonprofits",
      title: t("usecases.nonprofits.title"),
      desc: t("usecases.nonprofits.desc"),
      tag: t("usecases.nonprofits.tag"),
      icon: "🤝",
      gradient: "from-amber-500/10 via-amber-500/5 to-transparent",
    },
  ];

  return (
    <section id="usecases" className="usecases-section">
      <div className="wrap">
        <div className="section-head text-center">
          <span className="badge-pill">{t("usecases.pill")}</span>
          <h2>{t("usecases.title")}</h2>
          <p className="lead">{t("usecases.lead")}</p>
        </div>

        <div className="usecases-grid">
          {cases.map((c, i) => (
            <motion.div
              key={c.id}
              className={`usecase-card panel bg-gradient-to-b ${c.gradient}`}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <div className="card-top">
                <span className="card-icon">{c.icon}</span>
                <span className="card-tag">{c.tag}</span>
              </div>
              <h3>{c.title}</h3>
              <p>{c.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
