import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { getHadithCoverage, getHadithItem, getHadithList, getHadithPose } from "../api";
import { AvatarView } from "../components/AvatarView";
import { Counter } from "../components/Counter";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { HadithCoverage, HadithItem, HadithListItem, HadithSample, PoseFrames } from "../types";
import { COLLECTION_EN, glossFor, labelKind, readOffset, refFromHash, watchUrl } from "./hadithData";
import { embedUrl } from "./quranData";

// Signed Hadith: hadith signed by deaf-community channels (private dataset hadith-sign), browsed by collection
// (Nawawi's Forty first). Each hadith shows its text as printed in the collection, the signers who signed it, the
// signing replayed on the avatar, the original clip at its timestamp, and the Isharati machine gloss of the text.
// A nested Academy page (#/academy/hadith[/<ref>]), shown for the Academy's chosen sign language (signLanguage).

export function Hadith({ signLanguage = "" }: { signLanguage?: string }) {
  const { t, ui } = useI18n();
  const isRtl = ui === "ar" || ui === "ur";
  const { avatar, outfit } = useAppState();

  const [coverage, setCoverage] = useState<HadithCoverage | null>(null);
  const [collection, setCollection] = useState<string>(() => refFromHash(location.hash)?.split(":")[0] ?? "");
  const [signLang, setSignLang] = useState(signLanguage);
  useEffect(() => { setRef((cur) => (refFromHash(location.hash) === cur ? cur : null)); setSignLang(signLanguage); }, [signLanguage]);
  const [list, setList] = useState<HadithListItem[] | null>(null);
  const [ref, setRef] = useState<string | null>(() => refFromHash(location.hash));
  const [item, setItem] = useState<HadithItem | null>(null);
  const [sample, setSample] = useState<HadithSample | null>(null);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [poseLoading, setPoseLoading] = useState(false);
  const [error, setError] = useState(false);

  const listReq = useRef(0);
  const itemReq = useRef(0);
  const poseReq = useRef(0);
  const sectionRef = useRef<HTMLElement>(null);
  const avatarCanvas = useRef<HTMLCanvasElement>(null);
  const keypointCanvas = useRef<HTMLCanvasElement>(null);

  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { autoplay: true });

  // coverage: the collection chips; with no deep link, the first collection (the server lists Nawawi's Forty first)
  useEffect(() => {
    getHadithCoverage()
      .then((c) => {
        setCoverage(c);
        setCollection((cur) => cur || c.collections[0]?.collection || "");
      })
      .catch(() => setError(true));
  }, []);

  // the Academy's links (#hadith/<ref>) open that hadith, also when the page is already showing
  useEffect(() => {
    const onHash = () => {
      const r = refFromHash(location.hash);
      if (!r) return;
      setCollection(r.split(":")[0]);
      setRef(r);
      sectionRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    };
    if (refFromHash(location.hash)) setTimeout(onHash, 0);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // the hadith of the chosen collection (and sign language); the first one opens when none is chosen
  useEffect(() => {
    if (!collection) return;
    const my = ++listReq.current;
    setList(null);
    getHadithList(collection, signLang)
      .then((l) => {
        if (my !== listReq.current) return;
        setList(l);
        setRef((cur) => cur ?? l[0]?.ref ?? null);
      })
      .catch(() => my === listReq.current && setList([]));
  }, [collection, signLang]);

  useEffect(() => {
    if (!ref) return;
    const my = ++itemReq.current;
    setLoading(true);
    setItem(null);
    setSample(null);
    setFrames(null);
    getHadithItem(ref)
      .then((d) => {
        if (my !== itemReq.current) return;
        setItem(d);
        setLoading(false);
        const first = d.samples.find((s) => !signLang || s.sign_language === signLang) ?? d.samples[0];
        if (first) selectSample(first);
      })
      .catch(() => {
        if (my === itemReq.current) setLoading(false);
      });
  }, [ref]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setAvatarFailed(false);
  }, [avatar, outfit, sample]);

  function selectSample(s: HadithSample) {
    setSample(s);
    setFrames(null);
    const my = ++poseReq.current;
    setPoseLoading(true);
    getHadithPose(s.id)
      .then((f) => {
        if (my === poseReq.current) {
          setFrames(f);
          setPoseLoading(false);
        }
      })
      .catch(() => {
        if (my === poseReq.current) setPoseLoading(false);
      });
  }

  function selectCollection(c: string) {
    if (c === collection) return;
    setRef(null);
    setCollection(c);
  }

  function selectSignLang(l: string) {
    setRef(null);
    setSignLang(l);
  }

  const collName = (c: { collection: string; name?: string; collection_name?: string }) =>
    (isRtl ? c.name ?? c.collection_name : COLLECTION_EN[c.collection]) || c.name || c.collection_name || c.collection;
  const showAvatar = mode === "avatar" && !avatarFailed;
  const gloss = sample ? glossFor(sample) : null;
  const watch = sample ? watchUrl(sample) : null;
  const embed = sample?.video_id ? embedUrl({ youtube_id: sample.video_id, start: sample.start ?? 0, end: sample.end ?? 0 }) : null;
  const offset = sample ? readOffset(sample) : 0;
  const sameSource = (s: HadithSample) => (item?.samples.filter((x) => x.source === s.source).length ?? 0) > 1;
  const translation = ui === "tr" && item?.text_tr ? item.text_tr : item?.text_en;

  return (
    <section id="hadith" className="quran-dashboard hadith-section" ref={sectionRef}>
      <div className="wrap">
        <h2>{t("hadith.title")}</h2>
        <p className="lead">{t("hadith.lead")}</p>
        <p className="note">{t("hadith.private")}</p>

        {/* Source cards */}
        {coverage && (
          <div className="langs" style={{ marginBottom: 32 }}>
            {coverage.sources.map((src) => (
              <motion.div
                key={src.key}
                className="lang panel"
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <p className="label">{src.name} · {src.sign_language}</p>
                <div className="big"><Counter value={src.samples} /></div>
                <div>{t("hadith.samples")} · <Counter value={src.hadith} /> {t("hadith.distinct")}</div>
                <div><Counter value={src.hours} format="fixed1" /> h</div>
              </motion.div>
            ))}
          </div>
        )}
        {error && <p className="status err">{t("hadith.error")}</p>}

        {/* Collections (Nawawi's Forty first) and the sign-language filter */}
        {coverage && coverage.collections.length > 0 && (
          <div className="quick-surahs-bar">
            <span className="quick-label">{t("hadith.collections")}</span>
            <div className="quick-chips-scroll">
              {coverage.collections.map((c) => (
                <button
                  key={c.collection}
                  type="button"
                  className={`quick-surah-chip ${collection === c.collection ? "active" : ""}`}
                  onClick={() => selectCollection(c.collection)}
                >
                  <span className="chip-name">{collName(c)}</span>
                  <span className="chip-en">({c.hadith})</span>
                </button>
              ))}
            </div>
            {!signLanguage && <div className="seg" role="group" aria-label={t("hadith.signLanguage")}>
              {["", ...coverage.sign_languages.filter((s) => s.samples > 0).map((s) => s.sign_language)].map((l) => (
                <button key={l || "all"} type="button" className={signLang === l ? "on" : ""} onClick={() => selectSignLang(l)}>
                  {l || t("hadith.allLangs")}
                </button>
              ))}
            </div>}
          </div>
        )}

        <div className="quran-viewer-box">
          {/* The hadith of this collection */}
          {list === null && collection && <p className="status">{t("hadith.loading")}</p>}
          {list && list.length === 0 && <div className="panel empty-segments-box"><p>{t("hadith.none")}</p></div>}
          {list && list.length > 0 && (
            <div className="hadith-list" role="list">
              {list.map((h) => (
                <button
                  key={h.ref}
                  type="button"
                  role="listitem"
                  className={`hadith-card${ref === h.ref ? " selected" : ""}`}
                  onClick={() => setRef(h.ref)}
                >
                  <span className="hadith-card-num">{t("hadith.number")} {h.number}</span>
                  <span className="hadith-card-text" dir="rtl" lang="ar">{h.text}</span>
                  <span className="hadith-card-meta">{h.sign_languages.join(" · ")} · {h.samples} {t("hadith.samples")}</span>
                </button>
              ))}
            </div>
          )}

          {loading && <p className="status">{t("hadith.loading")}</p>}

          {/* The hadith text, as printed in the collection */}
          {item && (
            <div className="quran-ayah-display-card hadith-text-card">
              <div className="ayah-display-header">
                <div className="ayah-callout-badge">
                  <span className="ornament-ico">۞</span>
                  <span>{item.collection_name || collName(item)}</span>
                  <span className="ayah-badge-dot">·</span>
                  <span>{t("hadith.number")} {item.number}</span>
                </div>
              </div>
              <p className="hadith-text-ar" dir="rtl" lang="ar">{item.text_ar}</p>
              {translation && (
                <p className="hadith-text-tr" dir="ltr" lang={ui === "tr" && item.text_tr ? "tr" : "en"}>{translation}</p>
              )}
              {item.also_in.length > 0 && <p className="note">{t("hadith.also")} {item.also_in.join(" · ")}</p>}
            </div>
          )}

          {/* The signers of this hadith */}
          {item && item.samples.length > 0 && (
            <div className="segments-selector-row">
              <span className="seg-avail-label">{t("hadith.signers")}</span>
              <div className="seg-chips-list">
                {item.samples.map((s, k) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`chip ${sample?.id === s.id ? "on" : ""}`}
                    onClick={() => selectSample(s)}
                  >
                    {s.source_name} · {s.sign_language}{sameSource(s) ? ` #${k + 1}` : ""}
                  </button>
                ))}
              </div>
            </div>
          )}

          <AnimatePresence>
            {item && sample && (
              <motion.div
                className="quran-result-grid"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {/* Left column: who signed it, how it was labelled, the gloss of the text */}
                <div className="panel quran-text-detail-panel">
                  <div className="detail-panel-header">
                    <span className="panel-badge-kicker">✋ {t("hadith.signedBy")} {sample.channel || sample.source_name}</span>
                    <span className="source-credit">{sample.sign_language}</span>
                  </div>
                  <p className="note">{t(`hadith.label.${labelKind(sample.label_source)}`)}</p>

                  {gloss && (
                    <div className="detail-tafsir-box">
                      <p className="detail-label">{t("hadith.gloss")} · {gloss.lang}</p>
                      <div className="chips" dir={gloss.lang === "ArSL" ? "rtl" : "ltr"}>
                        {gloss.items.map((g, i) => (
                          <span key={i} className={`chip${g.oov ? " missing" : ""}`}>{g.text}</span>
                        ))}
                      </div>
                      {gloss.items.some((g) => g.oov) && <p className="note">{t("hadith.oovNote")}</p>}
                    </div>
                  )}

                  {watch && (
                    <p style={{ marginTop: 14 }}>
                      <a className="btn ghost" href={watch} target="_blank" rel="noopener">▶ {t("hadith.watch")}</a>
                    </p>
                  )}
                </div>

                {/* Right column: the signing on the avatar, then the original clip */}
                <div className="quran-media-column">
                  <div className="panel avatar-player-panel">
                    <div className="player-panel-top">
                      <p className="label" style={{ margin: 0 }}>{t("try.video")}</p>
                      <div className="seg">
                        <button type="button" className={mode === "avatar" ? "on" : ""} onClick={() => setMode("avatar")}>
                          {t("try.avatar")}
                        </button>
                        <button type="button" className={mode === "keypoints" ? "on" : ""} onClick={() => setMode("keypoints")}>
                          {t("try.keypoints")}
                        </button>
                      </div>
                    </div>

                    <div className="viewer">
                      {poseLoading && <p className="status">{t("hadith.loading")}</p>}
                      {!poseLoading && showAvatar ? (
                        <AvatarView
                          model={avatar}
                          outfit={outfit}
                          frames={frames}
                          time={player.t}
                          canvasRef={avatarCanvas}
                          onError={() => setAvatarFailed(true)}
                        />
                      ) : (
                        !poseLoading && <KeypointCanvas frames={frames} time={player.t} canvasRef={keypointCanvas} />
                      )}
                      {!poseLoading && !frames && <p className="status">{t("hadith.noPose")}</p>}
                    </div>

                    {frames && (
                      <div className="transport">
                        <button className="btn ghost" onClick={player.toggle}>
                          {player.playing ? t("try.pause") : t("try.play")}
                        </button>
                        <input
                          type="range"
                          min={0}
                          max={1000}
                          value={duration ? Math.round((player.t / duration) * 1000) : 0}
                          onChange={(e) => player.seek((+e.target.value / 1000) * duration)}
                          aria-label="position"
                        />
                        <select value={player.speed} onChange={(e) => player.setSpeed(+e.target.value)} aria-label={t("try.speed")}>
                          {[0.5, 0.75, 1].map((s) => (
                            <option key={s} value={s}>{s}×</option>
                          ))}
                        </select>
                      </div>
                    )}
                    {frames && offset > 0.5 && offset < duration && (
                      <button type="button" className="btn ghost" style={{ marginTop: 8 }} onClick={() => { player.seek(offset); player.play(); }}>
                        ⏩ {t("hadith.jump")} ({Math.round(offset)} s)
                      </button>
                    )}

                    {mode === "avatar" && <StyleSwitch />}
                    {avatarFailed && mode === "avatar" && <p className="status">{t("try.avatarFail")}</p>}
                  </div>

                  {embed && (
                    <div className="panel youtube-embed-panel">
                      <div className="embed-header">
                        <span className="embed-source-tag">📹 {t("hadith.original")}</span>
                      </div>
                      <iframe
                        src={embed}
                        title={`${t("hadith.title")} ${item.ref}`}
                        allow="encrypted-media; picture-in-picture"
                        loading="lazy"
                        style={{ width: "100%", aspectRatio: "16/9", border: "none", borderRadius: 8 }}
                      />
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}
