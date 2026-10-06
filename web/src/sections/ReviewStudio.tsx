import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { getPendingReviews, getReviewPose, submitSignReview, uploadSignRecording } from "../api";
import { AvatarView } from "../components/AvatarView";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { PoseFrames, ReviewItem } from "../types";

export function ReviewStudio() {
  const { t } = useI18n();
  const { avatar, outfit } = useAppState();
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [avatarReady, setAvatarReady] = useState(false);
  const avatarCanvasRef = useRef<HTMLCanvasElement>(null);
  const keypointCanvasRef = useRef<HTMLCanvasElement>(null);

  const [datasetFilter, setDatasetFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [datasetsMap, setDatasetsMap] = useState<Record<string, number>>({});
  const [statusesMap, setStatusesMap] = useState<Record<string, number>>({});
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [total, setTotal] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [selected, setSelected] = useState<ReviewItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [poseLoading, setPoseLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [notes, setNotes] = useState("");
  const [notification, setNotification] = useState<string | null>(null);

  // Camera recording states
  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [uploadingRecording, setUploadingRecording] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { autoplay: true });

  useEffect(() => { setAvatarFailed(false); }, [avatar, outfit, selected]);
  useEffect(() => { setAvatarReady(false); }, [avatar, selected, mode]);

  const showAvatar = mode === "avatar" && !avatarFailed;

  useEffect(() => {
    loadList(datasetFilter, statusFilter);
  }, [datasetFilter, statusFilter]);

  async function loadList(currentDataset = datasetFilter, currentStatus = statusFilter) {
    setLoading(true);
    try {
      const res = await getPendingReviews(100, 0, currentDataset, currentStatus);
      setItems(res.items);
      setTotal(res.total);
      setPendingCount(res.pending);
      if (res.datasets) {
        setDatasetsMap(res.datasets);
      }
      if (res.statuses) {
        setStatusesMap(res.statuses);
      }
      if (res.items.length > 0) {
        selectSign(res.items[0]);
      } else {
        setSelected(null);
        setFrames(null);
      }
    } catch {
      // offline / mock fallback
    } finally {
      setLoading(false);
    }
  }

  async function selectSign(item: ReviewItem) {
    setSelected(item);
    setNotes(item.notes || "");
    setFrames(null);
    stopCamera();
    setRecordedBlob(null);
    setRecordedUrl(null);
    setPoseLoading(true);
    try {
      const poseData = await getReviewPose(item.sign_id);
      setFrames(poseData);
    } catch {
      setFrames(null);
    } finally {
      setPoseLoading(false);
    }
  }

  async function handleDecision(decision: "true" | "false" | "need_modification") {
    if (!selected) return;
    setSubmitting(true);
    try {
      await submitSignReview(selected.sign_id, {
        decision,
        notes,
        reviewer: "expert_reviewer",
      });
      setNotification(`Sign «${selected.gloss}» marked as ${decision}!`);
      setTimeout(() => setNotification(null), 4000);
      // Update local state
      setItems((prev) =>
        prev.map((i) =>
          i.sign_id === selected.sign_id
            ? { ...i, decision, status: decision === "true" ? "approved" : decision === "false" ? "rejected" : "needs_modification", notes }
            : i
        )
      );
      setPendingCount((c) => Math.max(0, c - 1));
      // Auto-advance to next pending
      const next = items.find((i) => i.sign_id !== selected.sign_id && i.status === "pending");
      if (next) selectSign(next);
    } catch (e) {
      alert(`Error submitting review: ${e}`);
    } finally {
      setSubmitting(false);
    }
  }

  // Camera Management
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err) {
      alert(`Cannot open camera: ${err}`);
    }
  }

  function stopCamera() {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setRecording(false);
  }

  function toggleRecord() {
    if (recording) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      setRecording(false);
    } else {
      // Start recording
      if (!mediaStreamRef.current) return;
      recordedChunksRef.current = [];
      const mr = new MediaRecorder(mediaStreamRef.current, { mimeType: "video/webm" });
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
        setRecordedBlob(blob);
        setRecordedUrl(URL.createObjectURL(blob));
      };
      mr.start(100);
      mediaRecorderRef.current = mr;
      setRecording(true);
    }
  }

  async function handleUploadRecording() {
    if (!selected || !recordedBlob) return;
    setUploadingRecording(true);
    try {
      await uploadSignRecording(selected.sign_id, recordedBlob);
      setNotification(`Webcam re-recording saved for «${selected.gloss}»!`);
      setTimeout(() => setNotification(null), 4000);
      stopCamera();
    } catch (e) {
      alert(`Failed uploading recording: ${e}`);
    } finally {
      setUploadingRecording(false);
    }
  }

  return (
    <section id="review" style={{ scrollMarginTop: 80 }}>
      <div className="wrap">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>{t("review.title") || "Pending Signs Review & Annotation Studio"}</h2>
            <p className="lead">
              {t("review.lead") ||
                "Review RoPE-segmented continuous signs and dictionary prototypes. Verify accuracy, submit decisions to Neon PostgreSQL, or open your camera to re-record corrected physical gestures."}
            </p>
          </div>
          <div className="panel" style={{ padding: "8px 16px", borderRadius: 8, fontSize: "0.9rem" }}>
            <span style={{ color: "var(--accent)" }}>● </span>
            <strong>{pendingCount}</strong> Pending Review / <strong>{total}</strong> Seeded in DB
          </div>
        </div>

        {notification && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel"
            style={{ background: "#064e3b", color: "#34d399", borderColor: "#059669", marginBottom: 16 }}
          >
            ✓ {notification}
          </motion.div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 24, marginTop: 16 }}>
          {/* Left: Queue List */}
          <div className="panel" style={{ maxHeight: 680, overflowY: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <p className="label" style={{ margin: 0 }}>Review Queue</p>
              <span style={{ fontSize: "0.75rem", opacity: 0.6 }}>{total} total</span>
            </div>

            {/* Filter Controls: Dataset & Revision Status */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, margin: "4px 0 10px 0" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <label htmlFor="review-status-select" style={{ fontSize: "0.75rem", color: "var(--fg2, #94a3b8)", fontWeight: 600 }}>
                  Review Status Filter:
                </label>
                <select
                  id="review-status-select"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: "1px solid var(--accent, #3b82f6)",
                    background: "var(--bg2, #0f172a)",
                    color: "inherit",
                    fontSize: "0.85rem",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  <option value="all">All Signs ({total})</option>
                  <option value="needs_revision">⚠️ Needs Revision / Pending ({statusesMap["pending"] || pendingCount})</option>
                  <option value="approved">✓ Approved Only ({statusesMap["approved"] || 0})</option>
                  <option value="needs_modification">✎ Needs Modification ({statusesMap["needs_modification"] || 0})</option>
                  <option value="rejected">✕ Rejected ({statusesMap["rejected"] || 0})</option>
                </select>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <label htmlFor="review-dataset-select" style={{ fontSize: "0.75rem", color: "var(--fg2, #94a3b8)" }}>
                  Collection / Dataset:
                </label>
                <select
                  id="review-dataset-select"
                  value={datasetFilter}
                  onChange={(e) => setDatasetFilter(e.target.value)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: "1px solid var(--border, #333)",
                    background: "var(--bg2, #0f172a)",
                    color: "inherit",
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                >
                  <option value="all">All Collections (6,112 Signs)</option>
                  <option value="new_arabic_sources">New Broadcast Sources ({datasetsMap["new_arabic_sources"] || 2409})</option>
                  <option value="unified_dictionary">Unified Arabic Dict ({datasetsMap["unified_dictionary"] || 1283})</option>
                  <option value="kuwaiti_dictionary">Kuwaiti Sign Dict ({datasetsMap["kuwaiti_dictionary"] || 975})</option>
                  <option value="karsl">KArSL Saudi Benchmark ({datasetsMap["karsl"] || 492})</option>
                  <option value="quran_curriculum">Qur'an Curriculum ({datasetsMap["quran_curriculum"] || 305})</option>
                  <option value="tawasol">Tawasol Islamic Center ({datasetsMap["tawasol"] || 196})</option>
                  <option value="scouts_dictionary">Scouts Sign Dict ({datasetsMap["scouts_dictionary"] || 139})</option>
                  <option value="jordan_shorts">Jordanian Shorts ({datasetsMap["jordan_shorts"] || 110})</option>
                  <option value="children_dictionary">Children's Sign Dict ({datasetsMap["children_dictionary"] || 87})</option>
                  <option value="arabic_dictionary_explained">Arabic Dict Explained ({datasetsMap["arabic_dictionary_explained"] || 43})</option>
                  <option value="saudi_dictionary_explained">Saudi Dict Explained ({datasetsMap["saudi_dictionary_explained"] || 39})</option>
                  <option value="isharah">Isharah Corpus Mined ({datasetsMap["isharah"] || 34})</option>
                </select>
              </div>
            </div>

            {loading && <p className="status">{t("try.wait") || "Loading queue..."}</p>}
            {!loading && items.map((item) => {
              const isSel = selected?.sign_id === item.sign_id;
              const badgeBg =
                item.status === "approved" ? "#065f46" : item.status === "rejected" ? "#7f1d1d" : item.status === "needs_modification" ? "#78350f" : "#374151";
              return (
                <button
                  key={item.sign_id}
                  type="button"
                  onClick={() => selectSign(item)}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "10px 12px",
                    borderRadius: 6,
                    border: isSel ? "1.5px solid var(--accent, #3b82f6)" : "1px solid var(--border, #333)",
                    background: isSel ? "var(--bg3, #1e293b)" : "transparent",
                    color: "inherit",
                    cursor: "pointer",
                    textAlign: "left",
                  }}
                >
                  <div>
                    <strong style={{ fontSize: "1.1rem" }}>{item.gloss}</strong>
                    <div style={{ fontSize: "0.75rem", opacity: 0.6 }}>{item.dataset}</div>
                  </div>
                  <span style={{ fontSize: "0.7rem", padding: "2px 6px", borderRadius: 4, background: badgeBg, color: "#fff" }}>
                    {item.decision || item.status}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Right: Inspection & Annotation & Re-recording */}
          {selected && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div className="panel" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                {/* Visualizer: 3D Avatar & Keypoint Skeleton */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                    <div className="seg" style={{ margin: 0 }}>
                      <button
                        type="button"
                        className={mode === "avatar" ? "on" : ""}
                        onClick={() => setMode("avatar")}
                      >
                        {t("try.avatar") || "Avatar (3D)"}
                      </button>
                      <button
                        type="button"
                        className={mode === "keypoints" ? "on" : ""}
                        onClick={() => setMode("keypoints")}
                      >
                        {t("try.keypoints") || "Keypoints"}
                      </button>
                    </div>
                    <span style={{ fontSize: "0.8rem", opacity: 0.7 }}>25 FPS · 50 Joints</span>
                  </div>
                  <div
                    className="viewer"
                    style={{
                      height: 330,
                      background: "#08121f",
                      borderRadius: 8,
                      border: "1px solid var(--border)",
                      position: "relative",
                      overflow: "hidden",
                    }}
                  >
                    {poseLoading && <p className="status">{t("try.wait") || "Loading pose trajectory..."}</p>}
                    {!poseLoading && frames && showAvatar && (
                      <AvatarView
                        model={avatar}
                        outfit={outfit}
                        frames={frames}
                        time={player.t}
                        canvasRef={avatarCanvasRef}
                        onError={() => setAvatarFailed(true)}
                        onReady={() => setAvatarReady(true)}
                      />
                    )}
                    {!poseLoading && frames && !showAvatar && (
                      <KeypointCanvas
                        frames={frames}
                        time={player.t}
                        canvasRef={keypointCanvasRef}
                      />
                    )}
                    {!poseLoading && !frames && (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.5 }}>
                        No pose trajectory found on disk
                      </div>
                    )}
                    {avatarFailed && mode === "avatar" && (
                      <p className="status" style={{ position: "absolute", bottom: 8, left: 8, right: 8, zIndex: 5 }}>
                        {t("try.avatarFail") || "3D Avatar unavailable on this device; switch to keypoints."}
                      </p>
                    )}
                  </div>
                  {frames && (
                    <div className="transport" style={{ marginTop: 8 }}>
                      <button className="btn ghost" onClick={player.toggle}>
                        {player.playing ? t("try.pause") || "Pause" : t("try.play") || "Play"}
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={1000}
                        value={duration ? Math.round((player.t / duration) * 1000) : 0}
                        onChange={(e) => player.seek((+e.target.value / 1000) * duration)}
                        aria-label="position"
                      />
                      <select value={player.speed} onChange={(e) => player.setSpeed(+e.target.value)}>
                        {[0.5, 0.75, 1, 1.25].map((s) => (
                          <option key={s} value={s}>{s}×</option>
                        ))}
                      </select>
                    </div>
                  )}
                  {mode === "avatar" && (
                    <div style={{ marginTop: 10 }}>
                      <StyleSwitch />
                    </div>
                  )}
                </div>

                {/* Annotation Form */}
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                  <div>
                    <h3 style={{ margin: "0 0 6px 0", fontSize: "1.4rem" }}>
                      Sign: «{selected.gloss}»
                    </h3>
                    <div style={{ fontSize: "0.85rem", opacity: 0.8, marginBottom: 12 }}>
                      <div><strong>Sign ID:</strong> <code>{selected.sign_id}</code></div>
                      <div><strong>Dataset Origin:</strong> {selected.dataset}</div>
                      <div><strong>Current Status:</strong> {selected.status}</div>
                    </div>

                    <label style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 16 }}>
                      <span className="label">Linguistic Feedback & Review Notes</span>
                      <textarea
                        rows={3}
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="e.g. Non-manual markers clear, gesture stroke accurate, or handshape needs modification..."
                        style={{ width: "100%", padding: 8, borderRadius: 6, background: "var(--bg)", border: "1px solid var(--border)", color: "inherit" }}
                      />
                    </label>
                  </div>

                  {/* Decision Action Buttons */}
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={() => handleDecision("true")}
                      className="btn"
                      style={{ background: "#059669", color: "#fff", flex: 1 }}
                    >
                      ✓ True (Approve)
                    </button>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={() => handleDecision("need_modification")}
                      className="btn"
                      style={{ background: "#d97706", color: "#fff", flex: 1 }}
                    >
                      ⚠ Needs Modification
                    </button>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={() => handleDecision("false")}
                      className="btn"
                      style={{ background: "#dc2626", color: "#fff", flex: 1 }}
                    >
                      ✕ False (Reject)
                    </button>
                  </div>
                </div>
              </div>

              {/* Camera Re-recording Station */}
              <div className="panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <div>
                    <p className="label" style={{ margin: 0 }}>Webcam Physical Re-recording Station</p>
                    <span style={{ fontSize: "0.85rem", opacity: 0.75 }}>
                      Capture the correct physical sign gesture in real time with your webcam to override or modify this sign.
                    </span>
                  </div>
                  {!cameraActive ? (
                    <button type="button" className="btn" onClick={startCamera}>
                      📷 Open Camera
                    </button>
                  ) : (
                    <button type="button" className="btn ghost" onClick={stopCamera}>
                      Close Camera
                    </button>
                  )}
                </div>

                {cameraActive && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
                    {/* Live Feed */}
                    <div style={{ position: "relative", background: "#000", borderRadius: 8, overflow: "hidden", height: 280 }}>
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                      {recording && (
                        <div
                          style={{
                            position: "absolute",
                            top: 12,
                            left: 12,
                            background: "rgba(220, 38, 38, 0.85)",
                            color: "#fff",
                            padding: "4px 8px",
                            borderRadius: 4,
                            fontSize: "0.75rem",
                            fontWeight: 600,
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                          }}
                        >
                          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#fff", display: "inline-block" }}></span>
                          RECORDING
                        </div>
                      )}
                      <div style={{ position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)" }}>
                        <button
                          type="button"
                          className="btn"
                          onClick={toggleRecord}
                          style={{ background: recording ? "#dc2626" : "#2563eb", color: "#fff" }}
                        >
                          {recording ? "⏹ Stop Recording" : "⏺ Record Sign"}
                        </button>
                      </div>
                    </div>

                    {/* Review Recorded Video & Save to DB */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      <p className="label" style={{ margin: 0 }}>Recorded Take Preview</p>
                      {recordedUrl ? (
                        <>
                          <video
                            src={recordedUrl}
                            controls
                            style={{ width: "100%", height: 200, background: "#000", borderRadius: 8 }}
                          />
                          <button
                            type="button"
                            disabled={uploadingRecording}
                            className="btn"
                            onClick={handleUploadRecording}
                            style={{ background: "#059669", color: "#fff" }}
                          >
                            {uploadingRecording ? "Uploading & Linking..." : "💾 Save Re-recording to Database"}
                          </button>
                        </>
                      ) : (
                        <div
                          style={{
                            height: 200,
                            border: "1px dashed var(--border)",
                            borderRadius: 8,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            opacity: 0.6,
                            fontSize: "0.9rem",
                          }}
                        >
                          No take recorded yet. Click 'Record Sign' to start.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
