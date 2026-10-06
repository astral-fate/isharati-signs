import { motion } from "framer-motion";
import { PipelineStack } from "../components/PipelineStack";
import { useI18n } from "../i18n/i18n";

export function HowItWorks() {
  const { t } = useI18n();

  const modalities = [
    {
      id: "quran",
      title: t("modalities.quran.t"),
      body: t("modalities.quran.b"),
      tag: "114 Surahs",
      icon: "📖",
      gradient: "from-blue-500/10 via-blue-500/5 to-transparent",
      badge: "KFC Verified",
      href: "#quran",
    },
    {
      id: "tafsir",
      title: t("modalities.tafsir.t"),
      body: t("modalities.tafsir.b"),
      tag: "Juma'a & Lessons",
      icon: "🎙️",
      gradient: "from-emerald-500/10 via-emerald-500/5 to-transparent",
      badge: "Audio ASR + Sign",
      href: "#try",
    },
    {
      id: "studio",
      title: t("modalities.studio.t"),
      body: t("modalities.studio.b"),
      tag: "Root Morphology",
      icon: "✍️",
      gradient: "from-purple-500/10 via-purple-500/5 to-transparent",
      badge: "CAMeL Engine",
      href: "#studio",
    },
    {
      id: "rag",
      title: t("modalities.rag.t"),
      body: t("modalities.rag.b"),
      tag: "Hadith & Tafsir",
      icon: "🔍",
      gradient: "from-amber-500/10 via-amber-500/5 to-transparent",
      badge: "Dense + Sparse E5",
      href: "#try",
    },
  ];

  return (
    <section id="how" className="how-section">
      <div className="wrap">
        {/* Modality Cards (SignCaption inspired) */}
        <div className="section-head text-center" style={{ marginBottom: 40 }}>
          <span className="badge-pill">4 Modalities</span>
          <h2>{t("modalities.title")}</h2>
          <p className="lead">{t("modalities.lead")}</p>
        </div>

        <div className="modalities-grid" style={{ marginBottom: 64 }}>
          {modalities.map((m, i) => (
            <motion.a
              key={m.id}
              href={m.href}
              className={`modality-card panel bg-gradient-to-b ${m.gradient}`}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <div className="card-top">
                <span className="card-icon">{m.icon}</span>
                <span className="card-badge">{m.badge}</span>
              </div>
              <h3>{m.title}</h3>
              <p>{m.body}</p>
              <div className="card-foot">
                <span className="card-tag">{m.tag}</span>
                <span className="card-arrow">→</span>
              </div>
            </motion.a>
          ))}
        </div>

        {/* 3D Exploded AI Pipeline Layers Stack (Inspired by Mityaf) */}
        <div style={{ marginBottom: 64 }}>
          <div className="section-head text-center" style={{ marginBottom: 24 }}>
            <span className="badge-pill">3D Pipeline Layers</span>
            <h2>{t("pipeline.title")}</h2>
            <p className="lead">{t("pipeline.lead")}</p>
          </div>
          <PipelineStack />
        </div>

        {/* Technical Architecture Steps */}
        <div className="section-head text-center" style={{ marginBottom: 32 }}>
          <h2>{t("how.title")}</h2>
        </div>
        <div className="steps">
          {[1, 2, 3, 4, 5].map((n) => (
            <motion.div
              key={n}
              className="step panel"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.5, delay: n * 0.12 }}
            >
              <div className="n">{n}</div>
              <h3>{t(`how.${n}.t`)}</h3>
              <p>{t(`how.${n}.b`)}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
