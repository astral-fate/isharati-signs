import { useEffect, useState } from "react";
import { getHadithItem, getHadithList, getHadithPose, getQuranAyah, getQuranPose } from "../api";
import { AvatarView } from "../components/AvatarView";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { HadithListItem, PoseFrames } from "../types";
import { getAyahTextSync } from "./quranText";

// The landing page's taste of the sacred texts, laid out like the translator (cards on the left, the signer on the
// right): al-Fatiha 1:1 from the Sign Mushaf and a few of An-Nawawi's Forty from the signed hadith, each playable on
// the avatar, with "see all" links to the full pages (#/quran, #/hadith).

type Pick = { kind: "quran" } | { kind: "hadith"; ref: string };

export function SacredPreview() {
  const { t } = useI18n();
  const { avatar, outfit } = useAppState();
  const [hadith, setHadith] = useState<HadithListItem[]>([]);
  const [pick, setPick] = useState<Pick>({ kind: "quran" });
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [failed, setFailed] = useState(false);
  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { loop: true, autoplay: true });
  const fatiha = getAyahTextSync(1, 1)?.uthmani || "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ";

  useEffect(() => {
    getHadithList("nawawi")
      .then((l) => (l.length ? l : getHadithList()))
      .then((l) => setHadith(l.slice(0, 3)), () => setHadith([]));
  }, []);

  // the chosen sample's pose: the first signed segment of 1:1, or the first signer of the hadith
  useEffect(() => {
    let live = true;
    setFrames(null);
    const load = pick.kind === "quran"
      ? getQuranAyah(1, 1).then((a) => {
          const seg = a.segments.find((s) => s.isharati);
          return seg ? getQuranPose(seg.source, seg.id) : null;
        })
      : getHadithItem(pick.ref).then((h) => (h.samples[0] ? getHadithPose(h.samples[0].id) : null));
    load.then((f) => live && setFrames(f), () => live && setFrames(null));
    return () => { live = false; };
  }, [pick]);

  const on = (p: Pick) => p.kind === pick.kind && (p.kind === "quran" || (pick.kind === "hadith" && p.ref === pick.ref));
  return (
    <section id="sacred">
      <div className="wrap">
        <h2>{t("sacred.title")}</h2>
        <p className="lead">{t("sacred.lead")}</p>
        <div className="demo">
          <div className="demo-input">
            <span className="demo-badge">📖 {t("nav.quran")}</span>
            <button type="button" className={`hadith-card sp-card${on({ kind: "quran" }) ? " selected" : ""}`}
                    onClick={() => setPick({ kind: "quran" })}>
              <span className="hadith-card-num">الفاتحة · 1:1</span>
              <span className="hadith-card-text" dir="rtl" lang="ar">{fatiha}</span>
            </button>
            <a className="sp-all" href="/#/quran">{t("sacred.allQuran")} →</a>

            <span className="demo-badge" style={{ marginTop: 12 }}>📜 {t("nav.hadith")}</span>
            {hadith.map((h) => (
              <button key={h.ref} type="button" className={`hadith-card sp-card${on({ kind: "hadith", ref: h.ref }) ? " selected" : ""}`}
                      onClick={() => setPick({ kind: "hadith", ref: h.ref })}>
                <span className="hadith-card-num">{h.collection_name} · {h.number} · {h.sign_languages.join(" · ")}</span>
                <span className="hadith-card-text" dir="rtl" lang="ar">{h.text}</span>
              </button>
            ))}
            <a className="sp-all" href="/#/hadith">{t("sacred.allHadith")} →</a>
          </div>
          <div className="demo-output">
            <div className="viewer">
              {!frames ? <p className="status">{t("hadith.loading")}</p>
                : failed ? <KeypointCanvas frames={frames} time={player.t} />
                : <AvatarView model={avatar} outfit={outfit} frames={frames} time={player.t} onError={() => setFailed(true)} />}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
