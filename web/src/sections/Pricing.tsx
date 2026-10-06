import { motion } from "framer-motion";
import { useState } from "react";
import { useI18n } from "../i18n/i18n";

export function Pricing() {
  const { t } = useI18n();
  const [yearly, setYearly] = useState(false);

  const plans = [
    {
      id: "community",
      title: t("pricing.community.title"),
      price: t("pricing.community.price"),
      sub: t("pricing.community.sub"),
      desc: t("pricing.community.desc"),
      features: [
        t("pricing.feat.comm1"),
        t("pricing.feat.comm2"),
        t("pricing.feat.comm3"),
        t("pricing.feat.comm4"),
      ],
      cta: t("pricing.community.cta"),
      href: "#try",
      highlight: false,
    },
    {
      id: "pro",
      title: t("pricing.pro.title"),
      price: yearly ? "$24" : "$29",
      sub: yearly ? t("pricing.period.yr") : t("pricing.period.mo"),
      desc: t("pricing.pro.desc"),
      features: [
        t("pricing.feat.pro1"),
        t("pricing.feat.pro2"),
        t("pricing.feat.pro3"),
        t("pricing.feat.pro4"),
        t("pricing.feat.pro5"),
      ],
      cta: t("pricing.pro.cta"),
      href: "#studio",
      highlight: true,
      popular: t("pricing.popular"),
    },
    {
      id: "mosque",
      title: t("pricing.mosque.title"),
      price: yearly ? "$169" : "$199",
      sub: yearly ? t("pricing.period.yr") : t("pricing.period.mo"),
      desc: t("pricing.mosque.desc"),
      features: [
        t("pricing.feat.mosq1"),
        t("pricing.feat.mosq2"),
        t("pricing.feat.mosq3"),
        t("pricing.feat.mosq4"),
      ],
      cta: t("pricing.mosque.cta"),
      href: "mailto:mosques@isharati.org",
      highlight: false,
    },
    {
      id: "enterprise",
      title: t("pricing.enterprise.title"),
      price: t("pricing.enterprise.price"),
      sub: t("pricing.enterprise.sub"),
      desc: t("pricing.enterprise.desc"),
      features: [
        t("pricing.feat.ent1"),
        t("pricing.feat.ent2"),
        t("pricing.feat.ent3"),
        t("pricing.feat.ent4"),
        t("pricing.feat.ent5"),
      ],
      cta: t("pricing.enterprise.cta"),
      href: "mailto:contact@isharati.org",
      highlight: false,
    },
  ];

  return (
    <section id="pricing" className="pricing-section">
      <div className="wrap">
        <div className="section-head text-center">
          <span className="badge-pill">{t("pricing.pill")}</span>
          <h2>{t("pricing.title")}</h2>
          <p className="lead">{t("pricing.lead")}</p>

          <div className="billing-toggle">
            <span className={!yearly ? "active" : ""}>{t("pricing.monthly")}</span>
            <button
              type="button"
              role="switch"
              aria-checked={yearly}
              className={`toggle-switch ${yearly ? "on" : ""}`}
              onClick={() => setYearly(!yearly)}
              aria-label="Toggle annual billing"
            >
              <span className="switch-knob" />
            </button>
            <span className={yearly ? "active" : ""}>
              {t("pricing.yearly")}{" "}
              <span className="discount-pill">{t("pricing.save")}</span>
            </span>
          </div>
        </div>

        <div className="pricing-grid">
          {plans.map((p, i) => (
            <motion.div
              key={p.id}
              className={`pricing-card panel ${p.highlight ? "highlighted" : ""}`}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              {p.popular && <div className="popular-badge">{p.popular}</div>}
              <div className="pricing-header">
                <h3>{p.title}</h3>
                <div className="price-row">
                  <span className="price-val">{p.price}</span>
                  <span className="price-sub">{p.sub}</span>
                </div>
                <p className="price-desc">{p.desc}</p>
              </div>

              <ul className="feature-list">
                {p.features.map((f, j) => (
                  <li key={j}>
                    <svg className="check-icon" viewBox="0 0 20 20" fill="currentColor">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>

              <a
                href={p.href}
                className={`btn ${p.highlight ? "btn-primary" : "ghost"} card-cta`}
              >
                {p.cta}
              </a>
            </motion.div>
          ))}
        </div>

      </div>
    </section>
  );
}
