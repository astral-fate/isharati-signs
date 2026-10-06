import { motion } from "framer-motion";
import heroPose from "../assets/hero-pose.json";
import { Counter } from "../components/Counter";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { PoseFrames } from "../types";

const pose = heroPose as PoseFrames;

export function Hero() {
  const { t } = useI18n();
  const { stats, clips } = useAppState();
  const player = usePlayer(pose.frames.length / pose.fps, { loop: true, autoplay: true });
  const signs = stats.languages.reduce((n, l) => n + l.signs, 0);
  const passages = stats.distinct_passages ?? Math.max(...stats.languages.map((l) => l.passages));

  return (
    <header className="hero" id="top">
      <div className="wrap">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <div className="lp-kicker">
            <i />
            <span>{t("hero.kicker") || "منظومة إشارتي للذكاء الاصطناعي الإشاري"}</span>
          </div>

          <h1 className="grad">{t("hero.title")}</h1>
          <p className="lead">{t("hero.lead")}</p>

          <div className="hero-cta-group">
            <a className="btn btn-primary" href="#try">{t("hero.cta")} ↓</a>
            <a className="btn ghost" href="/#/academy">🎓 {t("hero.academy")}</a>
            <a className="btn ghost" href="#quran">📖 {t("nav.quran")}</a>
          </div>

          <div className="stats">
            <div><b><Counter value={stats.languages.length} /></b>{t("hero.languages")}</div>
            <div><b><Counter value={signs} /></b>{t("hero.signs")}</div>
            <div><b><Counter value={passages} /></b>{t("hero.passages")}</div>
          </div>
        </motion.div>

        {/* Right Stage: 3D Visualizer Box */}
        <motion.div initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 1, delay: 0.2 }}>
          <div className="stage feature-visual-box" style={{ borderRadius: 20, border: "1px solid rgba(160, 180, 225, 0.2)" }}>
            <KeypointCanvas frames={pose} time={player.t} />
          </div>
          {clips.length > 0 && (
            <div className="reel">
              {clips.slice(0, 3).map((c) => (
                <video key={c.webm} muted loop autoPlay playsInline poster={c.poster}>
                  <source src={c.webm} type="video/webm" /><source src={c.mp4} type="video/mp4" />
                </video>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </header>
  );
}
