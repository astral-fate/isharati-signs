import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ask, getPose, signText, transcribeAudio } from "../api";
import { AvatarView } from "../components/AvatarView";
import { GlossChips } from "../components/GlossChips";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { Lang, PoseFrames, Report } from "../types";
import { EXAMPLES, SIGN_LANGS, errorKey, textDir } from "./signLanguages";

type Status = { kind: "idle" } | { kind: "wait" } | { kind: "error"; text: string } | { kind: "info"; text: string };

export function Translator() {
  const { t, ui } = useI18n();
  const { avatar, outfit } = useAppState();
  const [lang, setLang] = useState<Lang>(ui);
  // "ask": a question answered from the sources; "text": Studio, the user's own text; "audio": speech/audio upload ASR
  const [kind, setKind] = useState<"ask" | "text" | "audio">(() => (location.hash === "#studio" ? "text" : "ask"));
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [report, setReport] = useState<Report | null>(null);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [avatarReady, setAvatarReady] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [isRecordingMic, setIsRecordingMic] = useState(false);
  const [micSeconds, setMicSeconds] = useState(0);
  const [isMorphing, setIsMorphing] = useState(false);
  const micMediaRecorder = useRef<MediaRecorder | null>(null);
  const micAudioChunks = useRef<Blob[]>([]);
  const micTimerRef = useRef<any>(null);
  const chosen = useRef(false);   // true once the user picks a sign-language tab or asks something
  const req = useRef(0);          // request token: a response is dropped if anything newer happened meanwhile
  const rec = useRef<{ r: MediaRecorder; discard: boolean } | null>(null);
  const avatarCanvas = useRef<HTMLCanvasElement>(null);
  const keypointCanvas = useRef<HTMLCanvasElement>(null);
  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { autoplay: true });

  function triggerMorph() {
    setIsMorphing(true);
    setTimeout(() => setIsMorphing(false), 600);
  }

  async function startMicRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micAudioChunks.current = [];
      const mr = new MediaRecorder(stream);
      micMediaRecorder.current = mr;
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) micAudioChunks.current.push(e.data);
      };
      mr.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        // the recorder's own format (webm/opus in Chrome and Firefox, mp4 in Safari), not an assumed one
        const blob = new Blob(micAudioChunks.current, { type: mr.mimeType || "audio/webm" });
        if (blob.size > 0) handleAudioUpload(blob);
        else setStatus({ kind: "error", text: t("try.error") + "no audio was recorded" });
      };
      mr.start(250);
      setIsRecordingMic(true);
      setMicSeconds(0);
      micTimerRef.current = setInterval(() => {
        setMicSeconds((s) => s + 1);
      }, 1000);
    } catch (err) {
      // no microphone or permission denied: say so (playing a sample here made a recording look transcribed)
      setStatus({ kind: "error", text: t("try.error") + `microphone unavailable (${(err as Error).name || err})` });
    }
  }

  function stopMicRecording() {
    if (micTimerRef.current) clearInterval(micTimerRef.current);
    if (micMediaRecorder.current && micMediaRecorder.current.state === "recording") {
      try { micMediaRecorder.current.stop(); } catch { /* ignore */ }
    }
    setIsRecordingMic(false);
  }

  useEffect(() => { if (!chosen.current) setLang(ui); }, [ui]);  // follow the interface until a language is chosen or asked
  useEffect(() => { setAvatarFailed(false); }, [avatar, outfit, report]);  // a new model, outfit or result gets a fresh try
  useEffect(() => { setAvatarReady(false); }, [avatar, report, mode]);     // the viewer remounts, so it is loading again
  useEffect(() => () => abortExport(), [report]);                          // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {                                                        // stop recording when the clip has played to its end
    if (!exporting || duration <= 0 || player.playing) return;
    if (player.t < duration) { abortExport(); return; }                    // paused midway: no truncated download
    const id = setTimeout(() => rec.current?.r.state === "recording" && rec.current.r.stop(), 150);
    return () => clearTimeout(id);
  }, [exporting, player.playing, player.t, duration]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {                                              // a shareable link: /?q=...&lang=tr#try
    const p = new URLSearchParams(location.search), q0 = p.get("q"), l0 = p.get("lang") as Lang | null;
    if (q0) { const l = SIGN_LANGS.some((s) => s.code === l0) ? l0! : ui; setLang(l); setQ(q0); run(q0, l, "ask"); }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {                                              // the nav's Studio link (#studio) opens the text tab
    const onHash = () => { if (location.hash === "#studio") { switchKind("text"); document.getElementById("try")?.scrollIntoView(); } };
    onHash();
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  function switchKind(k: "ask" | "text" | "audio") {
    req.current++;
    setKind(k); setQ(""); setReport(null); setFrames(null); setStatus({ kind: "idle" });
  }

  async function handleAudioUpload(file: File | Blob) {
    chosen.current = true;
    setStatus({ kind: "wait" });
    try {
      const res = await transcribeAudio(file);
      if (!res.text.trim()) throw new Error("no speech was heard in the recording");
      setQ(res.text);
      setStatus({ kind: "idle" });
      run(res.text, lang, "text");
    } catch (e) {
      // show the real failure: a fixed sample sentence here made every failed recording look like it worked
      setStatus({ kind: "error", text: t("try.error") + ((e as Error).message || String(e)) });
    }
  }

  async function selectSampleAudio(key: string) {
    chosen.current = true;
    let url = "";
    if (key === "khutbah") {
      url = "/audio/samples/hajj_rituals.mp3";
    } else if (key === "quran") {
      url = "/audio/samples/dawah_intro.mp3";
    } else {
      url = "/audio/samples/prophetic_seerah.mp3";
    }
    setStatus({ kind: "wait" });
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("Audio sample not found");
      const blob = await resp.blob();
      await handleAudioUpload(new File([blob], url.split("/").pop() || "sample.mp3", { type: blob.type || "audio/mpeg" }));
    } catch (e) {
      setStatus({ kind: "error", text: t("try.error") + ((e as Error).message || String(e)) });
    }
  }

  function abortExport() {
    const cur = rec.current;
    rec.current = null;
    if (cur) { cur.discard = true; if (cur.r.state !== "inactive") try { cur.r.stop(); } catch { /* already stopped */ } }
    setExporting(false);
  }

  async function run(question: string, l: Lang, as: "ask" | "text" = (kind === "text" ? "text" : "ask")) {
    if (!question.trim()) return;
    chosen.current = true;
    const my = ++req.current;
    setStatus({ kind: "wait" }); setReport(null); setFrames(null);
    try {
      const r = as === "text" ? await signText(question.trim(), l) : await ask(question.trim(), l);
      if (my !== req.current) return;
      if (r.status === "referred") return setStatus({ kind: "info", text: t("try.referred") });
      if (r.status === "unanswered") return setStatus({ kind: "info", text: t("try.unanswered") + (r.reason ?? "") });
      setReport(r);
      setStatus({ kind: "idle" });
      if (r.id) {
        const f = await getPose(r.id).catch(() => null);
        if (my !== req.current) return;
        if (f) setFrames(f); else setStatus({ kind: "error", text: t("try.noPose") });
      }
    } catch (e) {
      if (my !== req.current) return;
      setStatus({ kind: "error", text: t(errorKey(e, as)) + (errorKey(e, as) === "try.error" ? String((e as Error).message) : "") });
    }
  }

  function exportVideo() {
    const canvas = (mode === "avatar" && !avatarFailed ? avatarCanvas : keypointCanvas).current;
    if (!canvas || !duration || rec.current) return;
    try {
      if (typeof MediaRecorder === "undefined") throw new Error("MediaRecorder");
      const type = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"].find((x) => MediaRecorder.isTypeSupported(x));
      if (!type) throw new Error("mime");
      const r = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 6e6 });
      const cur = { r, discard: false };
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      r.onstop = () => {
        if (rec.current === cur) rec.current = null;
        setExporting(false);
        if (cur.discard) return;
        const url = URL.createObjectURL(new Blob(chunks, { type }));
        const a = document.createElement("a");
        a.href = url;
        a.download = `isharati_${lang}.${type.startsWith("video/mp4") ? "mp4" : "webm"}`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      };
      rec.current = cur;
      setExporting(true);
      player.seek(0); player.play(); r.start();
    } catch {
      rec.current = null;
      setExporting(false);
      setStatus({ kind: "error", text: t("try.error") });
    }
  }

  const showAvatar = mode === "avatar" && !avatarFailed;
  const dir = textDir(lang);
  const busy = status.kind === "wait";
  const TASKS = [
    { k: "ask" as const, icon: "💬", title: t("try.modeAsk"), desc: t("try.taskAskDesc") },
    { k: "text" as const, icon: "📝", title: t("try.modeText"), desc: t("try.taskTextDesc") },
    { k: "audio" as const, icon: "🎙️", title: t("try.audio"), desc: t("try.taskAudioDesc") },
  ];
  const micButton = (
    <button type="button" className={`btn-mic ${isRecordingMic ? "recording" : ""}`}
            onClick={isRecordingMic ? stopMicRecording : startMicRecording}
            title={isRecordingMic ? t("try.micStop") : t("try.micStart")} aria-label={isRecordingMic ? t("try.micStop") : t("try.micStart")}>
      <span>{isRecordingMic ? "⏹️" : "🎙️"}</span>
      {isRecordingMic && <span style={{ fontSize: "12px" }}>00:{micSeconds < 10 ? `0${micSeconds}` : micSeconds}</span>}
    </button>
  );
  return (
    <section id="try">
      <div className="wrap">
        <h2>{t("try.title")}</h2>
        <p className="lead">{t("try.demoLead")}</p>

        <div className="demo">
          {/* the input side: 1 task, 2 sign language, 3 input, then one obvious action */}
          <form className="demo-input" onSubmit={(e) => { e.preventDefault(); if (kind === "audio") return; triggerMorph(); run(q, lang); }}>
            <span className="demo-badge">Isharati Engine</span>

            <p className="demo-step"><b>1</b>{t("try.stepTask")}</p>
            <div className="task-cards" role="tablist">
              {TASKS.map((x) => (
                <button key={x.k} type="button" role="tab" aria-selected={kind === x.k}
                        className={"task-card" + (kind === x.k ? " on" : "")} onClick={() => switchKind(x.k)}>
                  <span className="task-icon" aria-hidden>{x.icon}</span>
                  <span className="task-title">{x.title}</span>
                  <span className="task-desc">{x.desc}</span>
                </button>
              ))}
            </div>

            <p className="demo-step"><b>2</b>{t("try.stepLang")}</p>
            <div className="tabs" role="tablist">
              {SIGN_LANGS.map((s) => (
                <button key={s.code} type="button" role="tab" aria-selected={s.code === lang} className={s.code === lang ? "on" : ""}
                        onClick={() => { chosen.current = true; req.current++; setLang(s.code); setQ(""); setReport(null); setFrames(null); setStatus({ kind: "idle" }); }}>
                  <bdi dir="ltr">{s.label}</bdi>
                </button>
              ))}
            </div>

            <p className="demo-step"><b>3</b>{t(kind === "ask" ? "try.stepAsk" : kind === "text" ? "try.stepText" : "try.stepAudio")}</p>
            {kind === "audio" ? (
              <div className="demo-audio">
                <button type="button" className={`btn-mic big ${isRecordingMic ? "recording" : ""}`} onClick={isRecordingMic ? stopMicRecording : startMicRecording}>
                  {isRecordingMic
                    ? <><span>⏹️ {t("try.micStop")}</span><span>(00:{micSeconds < 10 ? `0${micSeconds}` : micSeconds})</span>
                        <div className="mic-waveform"><div className="mic-wave-bar" /><div className="mic-wave-bar" /><div className="mic-wave-bar" /><div className="mic-wave-bar" /></div></>
                    : <><span style={{ fontSize: "1.3rem" }}>🎙️</span><span>{t("try.micStart")}</span></>}
                </button>
                <div className="dropzone" role="button" tabIndex={0}
                     onDragOver={(e) => e.preventDefault()}
                     onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleAudioUpload(f); }}
                     onClick={() => {
                       const input = document.createElement("input");
                       input.type = "file"; input.accept = "audio/*,video/*";
                       input.onchange = (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleAudioUpload(f); };
                       input.click();
                     }}>
                  <span aria-hidden>📁</span> {t("audio.drop")}
                </div>
                {q && (
                  <>
                    <p className="label" style={{ marginTop: 12 }}>{t("audio.transcript_label")}</p>
                    <textarea className="demo-text" value={q} dir={dir} lang={lang} rows={3} onChange={(e) => setQ(e.target.value)} />
                  </>
                )}
              </div>
            ) : (
              <div className={"demo-field" + (isMorphing ? " keyboard-morph-active" : "")}>
                <textarea className="demo-text" value={q} dir={dir} lang={lang} maxLength={kind === "text" ? 1000 : 300}
                          rows={kind === "text" ? 6 : 3} placeholder={t(kind === "text" ? "try.textPlaceholder" : "try.placeholder")}
                          aria-label={t(kind === "text" ? "try.textPlaceholder" : "try.placeholder")}
                          onChange={(e) => setQ(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && kind === "ask") { e.preventDefault(); triggerMorph(); run(q, lang); } }} />
                {micButton}
              </div>
            )}
            {kind === "text" && <p className="note">{q.length} / 1000</p>}

            <div className="demo-chips" dir={dir}>
              {kind === "ask" && EXAMPLES[lang].map((x) => (
                <button key={x} type="button" disabled={busy} onClick={() => { setQ(x); run(x, lang); }}><span className="chip-kicker">{t("try.suggest")}</span> <span>{x}</span></button>
              ))}
              {kind === "audio" && [["khutbah", "audio.sample1"], ["quran", "audio.sample2"], ["dua", "audio.sample3"]].map(([key, label]) => (
                <button key={key} type="button" disabled={busy} onClick={() => selectSampleAudio(key)}>{t(label)}</button>
              ))}
            </div>

            {busy && <p className="status">{t(kind === "audio" && !q ? "audio.transcribing" : "try.wait")}</p>}
            {(status.kind === "error" || status.kind === "info") &&
              <p className={"status" + (status.kind === "error" ? " err" : "")}>{status.text}</p>}

            {kind === "audio"
              ? <button type="button" className="btn demo-go" disabled={busy || !q.trim()} onClick={() => { triggerMorph(); run(q, lang, "text"); }}>
                  ⚡ {t("try.generate")}
                </button>
              : <button type="submit" className="btn demo-go" disabled={busy || !q.trim()}>⚡ {t("try.generate")}</button>}
          </form>

          {/* the output side: the signer, idle until something is generated */}
          <div className="demo-output">
            {report ? (
              <>
                <div className="demo-output-bar">
                  <p className="label" style={{ margin: 0 }}>{t("try.video")}</p>
                  <div className="seg">
                    <button type="button" className={mode === "avatar" ? "on" : ""} onClick={() => setMode("avatar")}>{t("try.avatar")}</button>
                    <button type="button" className={mode === "keypoints" ? "on" : ""} onClick={() => setMode("keypoints")}>{t("try.keypoints")}</button>
                  </div>
                </div>
                <div className="viewer">
                  {showAvatar
                    ? <AvatarView model={avatar} outfit={outfit} frames={frames} time={player.t} canvasRef={avatarCanvas}
                                  onError={() => setAvatarFailed(true)} onReady={() => setAvatarReady(true)} />
                    : <KeypointCanvas frames={frames} time={player.t} canvasRef={keypointCanvas} />}
                </div>
                {avatarFailed && mode === "avatar" && <p className="status">{t("try.avatarFail")}</p>}
                <div className="transport">
                  <button type="button" className="btn ghost" onClick={player.toggle}>{player.playing ? t("try.pause") : t("try.play")}</button>
                  <input type="range" min={0} max={1000} value={duration ? Math.round((player.t / duration) * 1000) : 0}
                         onChange={(e) => player.seek((+e.target.value / 1000) * duration)} aria-label="position" />
                  <select value={player.speed} onChange={(e) => player.setSpeed(+e.target.value)} aria-label={t("try.speed")}>
                    {[0.5, 0.75, 1].map((s) => <option key={s} value={s}>{s}×</option>)}
                  </select>
                  <button type="button" className="btn" disabled={exporting || !duration || (showAvatar && !avatarReady)} onClick={exportVideo}>
                    {exporting ? t("try.exporting") : t("try.export")}
                  </button>
                </div>
                {mode === "avatar" && <StyleSwitch />}
              </>
            ) : (
              <div className="demo-idle">
                <span className="demo-idle-icon" aria-hidden>{busy ? "⏳" : "🧍"}</span>
                <p>{t(busy ? "try.idleBusy" : "try.idle")}</p>
                {!busy && <p className="note">{t("try.idleHint")}</p>}
              </div>
            )}
          </div>
        </div>

        <AnimatePresence>
          {report && (
            <motion.div className="demo-info" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="panel">
                <p className="label">{t(report.mode === "text" ? "try.yourText" : "try.answer")}</p>
                <p className="answer" dir={dir} lang={lang}>{report.answer}</p>
                {report.mode === "text"
                  ? <p className="note">{t("try.unchecked")}{report.glosser === "rule" ? ` · ${t("try.ruleGlosser")}` : ""}</p>
                  : <div className="sources">
                      {report.sources?.map((s) => /^https?:\/\//.test(s.url)
                        ? <a key={s.url} href={s.url} target="_blank" rel="noopener">{s.reference}</a>
                        : <span key={s.url}>{s.reference}</span>)}
                    </div>}
              </div>
              <div className="panel">
                <p className="label">{t("try.gloss")}</p>
                <div dir={dir} lang={lang}>
                  <GlossChips segments={report.segments ?? []} time={player.t} onSeek={(s) => { player.seek(s); player.play(); }} />
                </div>
                <p className="note">
                  {t("try.coverage")} {Math.round((report.coverage ?? 0) * 100)}% · {t("try.missingNote")}
                  {report.cached ? ` · ${t("try.cached")}` : ""}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
