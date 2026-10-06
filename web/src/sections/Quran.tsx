import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { getQuranAyah, getQuranCoverage, getQuranPose } from "../api";
import { AvatarView } from "../components/AvatarView";
import { Counter } from "../components/Counter";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { PoseFrames, QuranAyah, QuranCoverage, QuranSegment, QuranSurahRow } from "../types";
import { AYAH_COUNTS, cellKind, embedUrl, QUICK_SURAHS, SURAHS_META } from "./quranData";
import { getAyahText, getAyahTextSync, QuranAyahText } from "./quranText";

function toArabicNumerals(n: number): string {
  const digits = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
  return String(n).replace(/\d/g, (d) => digits[+d]);
}

export function Quran() {
  const { t, ui } = useI18n();
  const isRtl = ui === "ar" || ui === "ur";
  const { avatar, outfit } = useAppState();

  const [coverage, setCoverage] = useState<QuranCoverage | null>(null);
  const [surah, setSurah] = useState(1);
  const [ayah, setAyah] = useState(1);
  const [ayahData, setAyahData] = useState<QuranAyah | null>(null);
  const [selSeg, setSelSeg] = useState<QuranSegment | null>(null);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [avatarReady, setAvatarReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [poseLoading, setPoseLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ayahText, setAyahText] = useState<QuranAyahText | null>(() => getAyahTextSync(1, 1));
  const [copied, setCopied] = useState(false);

  const req = useRef(0);
  const poseReq = useRef(0);
  const ayahStripRef = useRef<HTMLDivElement>(null);
  const avatarCanvas = useRef<HTMLCanvasElement>(null);
  const keypointCanvas = useRef<HTMLCanvasElement>(null);

  const maxAyahs = AYAH_COUNTS[surah - 1] ?? 1;
  const currentSurahMeta = SURAHS_META[surah - 1] || { num: surah, nameAr: "", nameEn: "", type: "meccan" };

  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { autoplay: true });

  // Initial load
  useEffect(() => {
    getQuranCoverage().then(setCoverage).catch(() => {});
  }, []);

  // Fetch Ayah data + actual Quran text
  useEffect(() => {
    const my = ++req.current;
    setLoading(true);
    setAyahData(null);
    setSelSeg(null);
    setFrames(null);
    setError(null);

    // Immediate sync fallback text, then async full text
    setAyahText(getAyahTextSync(surah, ayah));
    getAyahText(surah, ayah).then((txt) => {
      if (my === req.current && txt) setAyahText(txt);
    });

    getQuranAyah(surah, ayah)
      .then((d) => {
        if (my === req.current) {
          setAyahData(d);
          setLoading(false);
          // Auto-select the first segment if available for immediate viewing
          if (d.segments && d.segments.length > 0) {
            selectSeg(d.segments[0]);
          }
        }
      })
      .catch(() => {
        if (my === req.current) {
          setLoading(false);
          setError("error");
        }
      });
  }, [surah, ayah]);

  // Scroll active ayah into view in the horizontal strip
  useEffect(() => {
    if (ayahStripRef.current) {
      const activeBtn = ayahStripRef.current.querySelector<HTMLButtonElement>(`button[data-ayah="${ayah}"]`);
      if (activeBtn && typeof activeBtn.scrollIntoView === "function") {
        activeBtn.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
      }
    }
  }, [ayah, surah]);

  useEffect(() => {
    setAvatarFailed(false);
  }, [avatar, outfit, selSeg]);

  useEffect(() => {
    setAvatarReady(false);
  }, [avatar, selSeg, mode]);

  function selectSeg(seg: QuranSegment) {
    setSelSeg(seg);
    setFrames(null);
    if (!seg.isharati) return;
    const my = ++poseReq.current;
    setPoseLoading(true);
    getQuranPose(seg.source, seg.id)
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

  function goNearest() {
    if (ayahData?.nearest) {
      setSurah(ayahData.nearest[0]);
      setAyah(ayahData.nearest[1]);
    }
  }

  function selectSurah(s: number) {
    setSurah(s);
    setAyah(1);
  }

  function nextAyah() {
    if (ayah < maxAyahs) {
      setAyah(ayah + 1);
    } else if (surah < 114) {
      setSurah(surah + 1);
      setAyah(1);
    }
  }

  function prevAyah() {
    if (ayah > 1) {
      setAyah(ayah - 1);
    } else if (surah > 1) {
      const prevSurahMax = AYAH_COUNTS[surah - 2] ?? 1;
      setSurah(surah - 1);
      setAyah(prevSurahMax);
    }
  }

  function copyAyahText() {
    if (!ayahText?.uthmani) return;
    navigator.clipboard.writeText(`${ayahText.uthmani} [${currentSurahMeta.nameAr} : ${ayah}]`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const showAvatar = mode === "avatar" && !avatarFailed;
  const embed = selSeg ? embedUrl(selSeg) : null;

  return (
    <section id="quran" className="quran-dashboard">
      <div className="wrap">
        <h2>{t("quran.title")}</h2>
        <p className="lead">{t("quran.lead")}</p>
        <p className="note">{t("quran.private")}</p>

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
                <p className="label">{t(`quran.source.${src.key}`)}</p>
                <div className="big"><Counter value={src.segments} /></div>
                <div>{t("quran.segments")} · <Counter value={src.ayahs} /> {t("quran.ayahs")}</div>
                <div><Counter value={src.surahs} /> {t("quran.surahs")} · <Counter value={src.hours} format="fixed1" /> h</div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Surah map */}
        {coverage && (
          <div style={{ marginBottom: 28 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 8 }}>
              <p className="label" style={{ margin: 0 }}>{t("quran.map")}</p>
              <div className="legend" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {(["both", "recitation", "tafsir", "none"] as const).map((k) => (
                  <span key={k} className={`legend-cell legend-${k}`}>{t(`quran.legend.${k}`)}</span>
                ))}
              </div>
            </div>

            <div className="surah-map">
              {coverage.surahs.map((row: QuranSurahRow) => (
                <button
                  key={row.surah}
                  className={`surah-cell surah-${cellKind(row)}${surah === row.surah ? " surah-selected" : ""}`}
                  title={`${row.surah}${row.name ? ` · ${row.name}` : ""}`}
                  onClick={() => selectSurah(row.surah)}
                >
                  {row.surah}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Quick Surahs Chips */}
        <div className="quick-surahs-bar">
          <span className="quick-label">
            {t("quran.quickPicks") || (isRtl ? "سور شائعة:" : "Quick Surahs:")}
          </span>
          <div className="quick-chips-scroll">
            {QUICK_SURAHS.map((sNum) => {
              const meta = SURAHS_META[sNum - 1];
              if (!meta) return null;
              const isSelected = surah === sNum;
              return (
                <button
                  key={sNum}
                  type="button"
                  className={`quick-surah-chip ${isSelected ? "active" : ""}`}
                  onClick={() => selectSurah(sNum)}
                >
                  <span className="chip-num">{isRtl ? toArabicNumerals(sNum) : sNum}.</span>
                  <span className="chip-name">{meta.nameAr}</span>
                  <span className="chip-en">({meta.nameEn})</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Quran Studio Viewer */}
        <div className="quran-viewer-box">
          {/* Top Control Bar: Surah Picker & Stepper */}
          <div className="quran-control-bar">
            {/* Surah Dropdown with Badge */}
            <div className="surah-selector-wrap">
              <span className="surah-badge-num">
                {isRtl ? `سورة ${toArabicNumerals(surah)}` : `Surah ${surah}`}
              </span>
              <select
                value={surah}
                onChange={(e) => selectSurah(+e.target.value)}
                className="quran-surah-select"
                aria-label={t("quran.surah")}
              >
                {SURAHS_META.map((s) => (
                  <option key={s.num} value={s.num}>
                    {s.num}. {s.nameAr} ({s.nameEn}) — {s.ayahs} {t("quran.ayahs")} {s.type === "meccan" ? "· مكية" : "· مدنية"}
                  </option>
                ))}
              </select>
            </div>

            {/* Direct Step Bar (Ayah Navigation Bar) */}
            <div className="direct-nav-bar">
              <button
                type="button"
                className="nav-step-btn prev"
                onClick={prevAyah}
                disabled={surah === 1 && ayah === 1}
                title={isRtl ? "الآية السابقة" : "Previous Ayah"}
              >
                {isRtl ? "التالية ❯" : "❮ Prev"}
              </button>

              <div className="nav-step-info">
                <span className="ayah-cur">{isRtl ? toArabicNumerals(ayah) : ayah}</span>
                <span className="ayah-sep">/</span>
                <span className="ayah-tot">{isRtl ? toArabicNumerals(maxAyahs) : maxAyahs}</span>
              </div>

              <button
                type="button"
                className="nav-step-btn next"
                onClick={nextAyah}
                disabled={surah === 114 && ayah === maxAyahs}
                title={isRtl ? "الآية التالية" : "Next Ayah"}
              >
                {isRtl ? "❮ السابقة" : "Next ❯"}
              </button>
            </div>
          </div>

          {/* Quick Horizontal Ayah Numbers Strip */}
          <div className="ayah-strip-container">
            <div className="ayah-strip-label">
              <span>{isRtl ? "اختر الآية:" : "Select Ayah:"}</span>
            </div>
            <div className="ayah-strip-scroll" ref={ayahStripRef}>
              {Array.from({ length: maxAyahs }, (_, i) => i + 1).map((aNum) => {
                const isSelected = aNum === ayah;
                return (
                  <button
                    key={aNum}
                    type="button"
                    data-ayah={aNum}
                    className={`ayah-pill ${isSelected ? "selected" : ""}`}
                    onClick={() => setAyah(aNum)}
                  >
                    {isRtl ? toArabicNumerals(aNum) : aNum}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Actual Quranic Ayah Text Banner */}
          <div className="quran-ayah-display-card">
            <div className="ayah-display-header">
              <div className="ayah-callout-badge">
                <span className="ornament-ico">۞</span>
                <span>{ayahText?.surahName || currentSurahMeta.nameAr}</span>
                <span className="ayah-badge-dot">·</span>
                <span>{isRtl ? `الآية ${toArabicNumerals(ayah)}` : `Ayah ${ayah}`}</span>
                <span className="ayah-badge-dot">·</span>
                <span className="revelation-pill">{currentSurahMeta.type === "meccan" ? "مكية" : "مدنية"}</span>
              </div>

              <div className="ayah-actions-tools">
                <button
                  type="button"
                  className="copy-ayah-btn"
                  onClick={copyAyahText}
                  title={isRtl ? "نسخ الآية" : "Copy Ayah Text"}
                >
                  {copied ? (isRtl ? "✓ تم النسخ" : "✓ Copied") : (isRtl ? "📋 نسخ الآية" : "📋 Copy")}
                </button>
              </div>
            </div>

            {/* Large Quranic Text in Uthmani Script */}
            <div className="quran-uthmani-text" dir="rtl" lang="ar">
              {ayahText?.uthmani ? (
                <>
                  <span className="uthmani-string">{ayahText.uthmani}</span>
                  <span className="ayah-end-ornament">
                    ۝<span className="ayah-num-inside">{toArabicNumerals(ayah)}</span>
                  </span>
                </>
              ) : (
                <span className="ayah-loading-ph">
                  {loading ? "جارٍ تحميل نص الآية الكريمة..." : `${currentSurahMeta.nameAr} — ${surah}:${ayah}`}
                </span>
              )}
            </div>
          </div>

          {/* Ayah Loading or Error States */}
          {loading && <p className="status">{t("try.wait")}</p>}
          {error && <p className="status err">{t("try.error")}</p>}

          {/* No segments state: friendly helper with nearest covered button */}
          {!loading && ayahData && ayahData.segments.length === 0 && (
            <div className="panel empty-segments-box">
              <p>{t("quran.none")}</p>
              {ayahData.nearest && (
                <button className="btn ghost" style={{ marginTop: 8 }} onClick={goNearest}>
                  {t("quran.goNearest")} {ayahData.nearest[0]}:{ayahData.nearest[1]}
                </button>
              )}
            </div>
          )}

          {/* Sign Modality Selection Chips */}
          {!loading && ayahData && ayahData.segments.length > 0 && (
            <div className="segments-selector-row">
              <span className="seg-avail-label">{isRtl ? "الإشارات المتاحة لهذه الآية:" : "Available Signs:"}</span>
              <div className="seg-chips-list">
                {ayahData.segments.map((seg) => {
                  const isCurrent = selSeg?.id === seg.id && selSeg?.source === seg.source;
                  return (
                    <button
                      key={`${seg.source}-${seg.id}`}
                      className={`chip ${isCurrent ? "on" : ""}`}
                      onClick={() => selectSeg(seg)}
                    >
                      {t(`quran.source.${seg.source}`)} · {t(`quran.${seg.type}`)}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Selected segment detail: Sign Avatar & Tafsir/Recitation */}
          <AnimatePresence>
            {selSeg && (
              <motion.div
                className="quran-result-grid"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {/* Left Column: Tafsir & Word Details */}
                <div className="panel quran-text-detail-panel">
                  <div className="detail-panel-header">
                    <span className="panel-badge-kicker">
                      {selSeg.type === "tafsir" ? (isRtl ? "📖 التفسير بلغة الإشارة" : "📖 Tafsir Explanation") : (isRtl ? "🎙️ التلاوة بلغة الإشارة" : "🎙️ Recitation Sign")}
                    </span>
                    <span className="source-credit">{t(`quran.source.${selSeg.source}`)}</span>
                  </div>

                  {selSeg.arabic_text && (
                    <div className="detail-spoken-box">
                      <p className="detail-label">{isRtl ? "النص المنطوق / المُشار إليه:" : "Signed Text:"}</p>
                      <p dir="rtl" lang="ar" className="detail-spoken-text">
                        {selSeg.arabic_text}
                      </p>
                    </div>
                  )}

                  {selSeg.tafsir_text && (
                    <div className="detail-tafsir-box">
                      <p className="detail-label">{isRtl ? "شرح التفسير المعتمد:" : "Scholarly Tafsir:"}</p>
                      <p dir="rtl" lang="ar" className="detail-tafsir-text">
                        {selSeg.tafsir_text}
                      </p>
                    </div>
                  )}

                  {!selSeg.arabic_text && !selSeg.tafsir_text && (
                    <p className="note">{surah}:{ayah}</p>
                  )}

                  {!selSeg.isharati && (
                    <p className="note original-note" style={{ marginTop: 12 }}>
                      {t("quran.original")}
                    </p>
                  )}
                </div>

                {/* Right Column: 3D Avatar + YouTube Embed */}
                <div className="quran-media-column">
                  {selSeg.isharati && (
                    <div className="panel avatar-player-panel">
                      <div className="player-panel-top">
                        <p className="label" style={{ margin: 0 }}>{t("try.video")}</p>
                        <div className="seg">
                          <button
                            type="button"
                            className={mode === "avatar" ? "on" : ""}
                            onClick={() => setMode("avatar")}
                          >
                            {t("try.avatar")}
                          </button>
                          <button
                            type="button"
                            className={mode === "keypoints" ? "on" : ""}
                            onClick={() => setMode("keypoints")}
                          >
                            {t("try.keypoints")}
                          </button>
                        </div>
                      </div>

                      <div className="viewer">
                        {poseLoading && <p className="status">{t("try.wait")}</p>}
                        {!poseLoading && showAvatar ? (
                          <AvatarView
                            model={avatar}
                            outfit={outfit}
                            frames={frames}
                            time={player.t}
                            canvasRef={avatarCanvas}
                            onError={() => setAvatarFailed(true)}
                            onReady={() => setAvatarReady(true)}
                          />
                        ) : (
                          !poseLoading && (
                            <KeypointCanvas
                              frames={frames}
                              time={player.t}
                              canvasRef={keypointCanvas}
                            />
                          )
                        )}
                        {!poseLoading && !frames && selSeg.isharati && (
                          <p className="status">{t("quran.noPose")}</p>
                        )}
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
                          <select
                            value={player.speed}
                            onChange={(e) => player.setSpeed(+e.target.value)}
                            aria-label={t("try.speed")}
                          >
                            {[0.5, 0.75, 1].map((s) => (
                              <option key={s} value={s}>{s}×</option>
                            ))}
                          </select>
                        </div>
                      )}

                      {mode === "avatar" && <StyleSwitch />}
                      {avatarFailed && mode === "avatar" && <p className="status">{t("try.avatarFail")}</p>}
                    </div>
                  )}

                  {!embed && selSeg.video_url && (
                    <div className="panel youtube-embed-panel">
                      <div className="embed-header">
                        <span className="embed-source-tag">📹 {t("quran.originalRecording") || (isRtl ? "التسجيل الأصلي" : "Original Recording")}</span>
                      </div>
                      {/* Tebyan publishes each ayah as its own clip, not a YouTube video */}
                      <video src={selSeg.video_url} controls muted playsInline preload="metadata"
                             style={{ width: "100%", aspectRatio: "16/9", borderRadius: 8, background: "#000" }} />
                    </div>
                  )}

                  {embed && (
                    <div className="panel youtube-embed-panel">
                      <div className="embed-header">
                        <span className="embed-source-tag">📹 {t("quran.originalRecording") || (isRtl ? "التسجيل الأصلي" : "Original Recording")}</span>
                      </div>
                      <iframe
                        src={embed}
                        title={`${t("quran.title")} ${surah}:${ayah}`}
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
