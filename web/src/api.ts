import type {
  Clip,
  HadithCoverage,
  HadithItem,
  HadithListItem,
  Lang,
  PoseFrames,
  QuranAyah,
  QuranCoverage,
  QuranSegment,
  Report,
  ReviewListResponse,
  ReviewSubmitPayload,
  Stats,
} from "./types";

export type ApiErrorKind = "html" | "quota" | "server" | "network";

export class ApiError extends Error {
  constructor(public kind: ApiErrorKind, message: string) { super(message); this.name = "ApiError"; }
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try { r = await fetch(url, init); } catch (e) { throw new ApiError("network", String(e)); }
  // a private Space's sign-in page or a proxy error page arrives as HTML
  if (!(r.headers.get("content-type") || "").includes("json")) throw new ApiError("html", `HTTP ${r.status}`);
  let body: any;
  try { body = await r.json(); } catch (e) { throw new ApiError("server", `HTTP ${r.status}: unreadable reply`); }
  const d = body && typeof body === "object" ? (body as any).detail : undefined;
  const msg = typeof d === "string" ? d : d ? JSON.stringify(d) : "";
  if (r.status === 503) throw new ApiError("quota", msg || "quota");
  if (!r.ok) throw new ApiError("server", msg || r.statusText);
  return body as T;
}

export const ask = (question: string, lang: Lang) =>
  call<Report>("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ question, lang }) });
export const signText = (text: string, lang: Lang) =>  // Studio: the user's own text, no answer generation
  call<Report>("/api/sign", { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ text, lang }) });
export const getPose = (id: string) => call<PoseFrames>(`/pose/${id}.json`);
export const getStats = () => call<Stats>("/api/stats");
export const getAvatars = () => call<string[]>("/api/avatars");
export const getClips = () => call<Clip[]>("/clips/clips.json").then((c) => (Array.isArray(c) ? c : []), () => [] as Clip[]);
export const getQuranCoverage = () => call<QuranCoverage>("/api/quran/coverage");
export const getQuranAyah = (surah: number, ayah: number) =>
  call<QuranAyah>(`/api/quran/ayah?surah=${surah}&ayah=${ayah}`);
export const getQuranPose = (source: string, id: string) =>
  call<PoseFrames>(`/api/quran/pose/${source}/${id}.json`);
export const getHadithCoverage = () => call<HadithCoverage>("/api/hadith/coverage");
export const getHadithList = (collection = "", signLanguage = "") =>
  call<{ hadith: HadithListItem[] }>(
    `/api/hadith/list?collection=${encodeURIComponent(collection)}&sign_language=${encodeURIComponent(signLanguage)}`
  ).then((r) => r.hadith);
export const getHadithItem = (ref: string) => call<HadithItem>(`/api/hadith/item?ref=${encodeURIComponent(ref)}`);
export const getHadithPose = (id: string) => call<PoseFrames>(`/api/hadith/pose/${encodeURIComponent(id)}.json`);

export interface TranscribeResult {
  text: string;
  segments?: Array<{ text: string; start: number; end: number }>;
}

export const transcribeAudio = async (file: File | Blob): Promise<TranscribeResult> => {
  const fd = new FormData();
  // a recorded Blob has no name: give it one whose extension matches its format, so the server can decode it
  const type = (file.type || "").split(";")[0];
  const ext = ({ "audio/webm": "webm", "video/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "video/mp4": "mp4",
                 "audio/mpeg": "mp3", "audio/wav": "wav" } as Record<string, string>)[type] || "webm";
  fd.append("file", file, (file as File).name || `recording.${ext}`);
  return call<TranscribeResult>("/api/transcribe", { method: "POST", body: fd });
};

export const getPendingReviews = (limit = 50, offset = 0, dataset = "all", status = "all") =>
  call<ReviewListResponse>(
    `/api/reviews/pending?limit=${limit}&offset=${offset}&dataset=${encodeURIComponent(dataset)}&status=${encodeURIComponent(status)}`
  );

export const getReviewPose = (signId: string) =>
  call<PoseFrames>(`/api/reviews/pose/${encodeURIComponent(signId)}.json`);

export const submitSignReview = (signId: string, payload: ReviewSubmitPayload) =>
  call<{ ok: boolean; sign_id: string; status: string; decision: string }>(
    `/api/reviews/${encodeURIComponent(signId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );

export const uploadSignRecording = async (signId: string, videoBlob: Blob) => {
  const fd = new FormData();
  fd.append("video", videoBlob, `${signId}.webm`);
  return call<{ ok: boolean; sign_id: string; saved_path: string }>(
    `/api/reviews/record/${encodeURIComponent(signId)}`,
    { method: "POST", body: fd }
  );
};


// Isharati Academy (src/isharati/academy.py): the curriculum and one item's sign in one sign language
export interface AcademyItem { id: string; meaning: string; words: Record<string, string>; texts?: Record<string, string>; src?: Record<string, string>; langs: string[] }
export interface Curriculum {
  sign_languages: Record<string, string>;
  paths: { id: string; icon: string; title: string; lessons: { id: string; items: AcademyItem[] }[] }[];
}
export const getCurriculum = (ui: string) => call<Curriculum>(`/api/academy/curriculum?ui=${encodeURIComponent(ui)}`);
export const getAcademySign = (lang: string, id: string) => call<PoseFrames>(`/api/academy/sign/${lang}/${id}.json`);
