import { motion } from "framer-motion";
import { useI18n } from "../i18n/i18n";

interface Member {
  name: string;
  nameSub: string;
  role: string;
  avatar: string;
  qrImg: string;
  badge: string;
  tags: string[];
  bullets: string[];
  social: {
    platform: "LinkedIn" | "YouTube";
    handle: string;
    url: string;
  };
  extraLinks?: { label: string; url: string; icon: string }[];
}

export function Team() {
  const { t, ui } = useI18n();
  const isAr = ui === "ar" || ui === "ur";

  const members: Member[] = [
    {
      name: isAr ? "فاطمة عماد الدين" : "Fatimah Emad Eldin",
      nameSub: isAr ? "Fatimah Emad Eldin" : "فاطمة عماد الدين",
      role: t("team.fatimah.role"),
      avatar: "/team/fatimah.png",
      qrImg: "/team/qr_linkedin.png",
      badge: isAr ? "الذكاء الاصطناعي والأبحاث" : "AI & Computer Vision",
      tags: [
        "10+ Papers",
        "MenaML 2026 (KAUST)",
        "Hugging Face",
        "ARSL-GEN",
      ],
      bullets: [
        t("team.fatimah.b1"),
        t("team.fatimah.b2"),
        t("team.fatimah.b3"),
        t("team.fatimah.b4"),
      ],
      social: {
        platform: "LinkedIn",
        handle: "linkedin.com/in/astral-fate",
        url: "https://www.linkedin.com/in/astral-fate",
      },
      extraLinks: [
        { label: "Hugging Face", url: "https://huggingface.co/FatimahEmadEldin", icon: "🤗" },
        { label: "GitHub", url: "https://github.com/astral-fate", icon: "💻" },
      ],
    },
    {
      name: isAr ? "د. أسماء الميرغني" : "Dr. Asmaa Al-Mirghani",
      nameSub: isAr ? "Dr. Asmaa Al-Mirghani" : "د. أسماء الميرغني",
      role: t("team.asmaa.role"),
      avatar: "/team/asmaa.jpg",
      qrImg: "/team/qr_youtube.png",
      badge: isAr ? "المحتوى الشرعي واللغوي" : "Sharia & Pedagogical Review",
      tags: [
        "PhD Applied Statistics",
        "BA Islamic Sharia",
        "20+ Years Exp",
        "70k+ Community",
      ],
      bullets: [
        t("team.asmaa.b1"),
        t("team.asmaa.b2"),
        t("team.asmaa.b3"),
        t("team.asmaa.b4"),
      ],
      social: {
        platform: "YouTube",
        handle: "youtube.com/channel/UC0AyY5d…",
        url: "https://www.youtube.com/channel/UC0AyY5dK_0iL31BMiphTh3A",
      },
    },
  ];

  return (
    <section id="team" className="team-section">
      <div className="wrap">
        <div className="team-header">
          <span className="team-kicker">{t("team.kicker")}</span>
          <h2>{t("team.title")}</h2>
          <p className="lead">{t("team.lead")}</p>
        </div>

        <div className="deck-team-grid">
          {members.map((m, idx) => (
            <motion.div
              key={m.name}
              className="deck-team-card"
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.45, delay: idx * 0.12 }}
            >
              {/* Top Accent Stripe matching PPTX deck line(s, x, 2.55, 8.85, 0.12, TEALBR) */}
              <div className="deck-card-top-stripe" />

              {/* Main Profile Header */}
              <div className="deck-profile-row">
                {/* Image Frame Container matching PPTX deck image(s, m['img'], x + 0.35, 2.85, 2.3, 3.0) */}
                <div className="deck-image-frame-container">
                  <div className="deck-image-inner-frame">
                    <img
                      src={m.avatar}
                      alt={m.name}
                      className="deck-member-photo"
                      loading="lazy"
                    />
                  </div>
                  <div className="deck-verified-badge" title="Verified Track Record">
                    ✓
                  </div>
                </div>

                {/* Identity & Credentials Beside Frame */}
                <div className="deck-identity-col">
                  <div className="deck-sub-badge">{m.badge}</div>
                  <h3 className="deck-primary-name">{m.name}</h3>
                  <div className="deck-sub-name">{m.nameSub}</div>
                  <div className="deck-role-title">{m.role}</div>

                  {/* Quick Tag Pills */}
                  <div className="deck-chips-wrap">
                    {m.tags.map((tag) => (
                      <span key={tag} className="deck-chip-tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Bullet Points with Teal Accent Dots */}
              <div className="deck-bullets-list">
                {m.bullets.map((bullet, bIdx) => (
                  <div key={bIdx} className="deck-bullet-item">
                    <span className="deck-bullet-marker" />
                    <span className="deck-bullet-text">{bullet}</span>
                  </div>
                ))}
              </div>

              {/* Bottom Social Strip matching PPTX deck social strip */}
              <div className="deck-social-strip">
                <a
                  href={m.social.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="deck-qr-link"
                  title={`${m.social.platform} QR Code`}
                >
                  <img
                    src={m.qrImg}
                    alt={`${m.name} QR`}
                    className="deck-qr-image"
                  />
                </a>

                <div className="deck-social-info">
                  <div className="deck-social-topline">
                    {m.social.platform === "LinkedIn" ? (
                      <span className="deck-social-badge linkedin">in</span>
                    ) : (
                      <span className="deck-social-badge youtube">▶</span>
                    )}
                    <span className="deck-social-name">{m.social.platform}</span>
                  </div>
                  <a
                    href={m.social.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="deck-social-handle"
                  >
                    {m.social.handle} <span className="arrow-sym">↗</span>
                  </a>
                </div>

                {m.extraLinks && m.extraLinks.length > 0 && (
                  <div className="deck-extra-links">
                    {m.extraLinks.map((ex) => (
                      <a
                        key={ex.url}
                        href={ex.url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="deck-extra-btn"
                        title={ex.label}
                      >
                        <span className="deck-extra-icon">{ex.icon}</span>
                        <span>{ex.label}</span>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
