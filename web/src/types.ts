export type Lang = "en" | "ar" | "tr" | "ur";

export interface Segment { label: string; kind: "sign" | "missing" | "fingerspell"; start_s: number; end_s: number }
export interface Source { reference: string; url: string }
export interface Report {
  status: "signed" | "referred" | "unanswered";
  id?: string; question?: string; answer?: string; reason?: string;
  sources?: Source[]; glosses?: string[]; missing_signs?: string[]; segments?: Segment[];
  coverage?: number; llm?: string | null; cached?: boolean; dropped_unsupported?: string[];
  mode?: "answer" | "text"; glosser?: string;
}
// blend: per frame, the signer's face blendshape scores (ARKit names in blend_names; null where no face was seen)
// face: per frame, 160 face contour points [x, y] in the pose's frame (null where none); face_edges: how to join them
export interface PoseFrames {
  fps: number; frames: number[][][]; blend?: (number | null)[][]; blend_names?: string[];
  face?: ((number | null)[] | null)[][]; face_edges?: Record<string, number[][]>;
}
export interface LangStats {
  code: Lang; name: string; sign_language: string; signs: number; glosses: number; passages: number;
  coverage: number; signs_by_source?: Record<string, number>;
}
export interface Licence { source: string; licence: string; use: string }
export interface Stats { languages: LangStats[]; distinct_passages?: number; licences: Licence[]; created?: string; avatars?: number }
export interface Clip {
  avatar: string; lang: Lang; sign_language: string; glosses: string[];
  webm: string; mp4: string; poster: string; seconds: number;
}

export type QuranSourceKey = "kfc" | "mukhtasar" | "curriculum" | "tebyan" | "diyanet";
export interface QuranSourceCoverage {
  key: QuranSourceKey; label: string; segments: number; recitation: number; tafsir: number;
  ayahs: number; surahs: number; hours: number;
}
export interface QuranSurahRow {
  surah: number; name: string;
  kfc: { recitation: number; tafsir: number };
  curriculum: { recitation: number; tafsir: number };
  mukhtasar: { recitation: number; tafsir: number };
  tebyan?: { recitation: number; tafsir: number };
  diyanet?: { recitation: number; tafsir: number };
}
export interface QuranCoverage { sources: QuranSourceCoverage[]; surahs: QuranSurahRow[] }
export interface QuranSegment {
  id: string; source: QuranSourceKey; type: "recitation" | "tafsir";
  ayahs: [number, number][]; seconds: number; isharati: boolean;
  youtube_id?: string; start?: number; end?: number; video_url?: string;
  arabic_text?: string; tafsir_text?: string;
}
export interface QuranAyah { surah: number; ayah: number; segments: QuranSegment[]; nearest: [number, number] | null }

// Signed hadith (src/isharati/hadith_data.py): one sample = one signed video clip labelled with a collection's hadith
export type HadithSignLanguage = "ArSL" | "TİD";
export interface HadithCount { samples: number; hadith: number }
export interface HadithCoverage extends HadithCount {
  hours: number; skipped: Record<string, number>;
  sources: ({ key: string; name: string; sign_language: string; hours: number } & HadithCount)[];
  sign_languages: ({ sign_language: string } & HadithCount)[];
  collections: ({ collection: string; name: string } & HadithCount)[];
}
export interface HadithListItem {
  ref: string; collection: string; collection_name: string; number: string; text: string;
  samples: number; sign_languages: string[];
}
export interface HadithGloss { text: string; oov: boolean }
export interface HadithSample {
  id: string; source: string; source_name: string; sign_language: string; channel?: string | null; title?: string | null;
  youtube_url?: string | null; video_id?: string | null; start?: number | null; end?: number | null;
  label_source: string; read_start?: number | null; read_end?: number | null;
  gloss?: Record<string, HadithGloss[]> | null; isharati: boolean;
}
export interface HadithItem {
  ref: string; collection: string; collection_name: string; number: string; also_in: string[];
  text_ar: string; text_en?: string | null; text_tr?: string | null; samples: HadithSample[];
}

export interface ReviewItem {
  id: number;
  sign_id: string;
  gloss: string;
  dataset: string;
  status: "pending" | "approved" | "rejected" | "needs_modification";
  decision: "true" | "false" | "need_modification" | null;
  notes: string | null;
  reviewer: string | null;
  keypoints_path: string | null;
  video_url: string | null;
  updated_at?: string;
}

export interface ReviewListResponse {
  items: ReviewItem[];
  total: number;
  pending: number;
  datasets?: Record<string, number>;
  statuses?: Record<string, number>;
}

export interface ReviewSubmitPayload {
  decision: "true" | "false" | "need_modification";
  notes?: string;
  reviewer?: string;
}
