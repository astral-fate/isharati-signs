import { motion } from "framer-motion";
import { AVATAR_KEYS, STYLES, avatarsFor } from "../components/avatars";
import { ClipCard } from "../components/ClipCard";
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Avatars() {
  const { t } = useI18n();
  const { avatars, clips, setAvatar } = useAppState();
  return (
    <section id="avatars">
      <div className="wrap">
        <h2>{t("av.title")}</h2>
        <p className="lead">{t("av.lead")}</p>
        {STYLES.map((style) => (
          <div key={style} style={{ marginBottom: 40 }}>
            <h3>{t(`style.${style}`)}</h3>
            <p className="note" style={{ marginTop: 0 }}>{t(`style.${style}Note`)}</p>
            <div className="gallery">
              {avatarsFor(style, avatars).map((m, i) => {
                const clip = clips.find((c) => c.avatar === m);
                return (
                  <motion.div key={m} className="clip panel" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                              viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
                    {clip ? <ClipCard clip={clip} /> : <div className="poster" />}
                    <div className="meta">
                      <div>{t(AVATAR_KEYS[m] ?? m)}<br /><small>{clip ? `${clip.sign_language} · ${clip.glosses.join(" · ")}` : ""}</small></div>
                      <a className="btn ghost" href="#try" onClick={() => setAvatar(m)}>{t("av.use")}</a>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
