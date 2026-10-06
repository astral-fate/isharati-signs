import type { QuranSurahRow } from "../types";

/** "both" when both recitation and tafsir exist (across all sources), "recitation", "tafsir", or "none" */
export function cellKind(row: QuranSurahRow): "both" | "recitation" | "tafsir" | "none" {
  const keys = ["kfc", "curriculum", "mukhtasar", "tebyan", "diyanet"] as const;
  // a source the server has not loaded yet has no counts
  const has = (k: (typeof keys)[number]) => row[k] ?? { recitation: 0, tafsir: 0 };
  const rec = keys.some((k) => has(k).recitation > 0);
  const taf = keys.some((k) => has(k).tafsir > 0);
  if (rec && taf) return "both";
  if (rec) return "recitation";
  if (taf) return "tafsir";
  return "none";
}

/**
 * Index of the word whose time window [start_s, end_s) contains t (relative to segment start).
 * Returns -1 when no word matches or segments is empty.
 */
export function wordAt(words: { start_s: number; end_s: number }[], t: number): number {
  for (let i = 0; i < words.length; i++) {
    if (t >= words[i].start_s && t < words[i].end_s) return i;
  }
  return -1;
}

/** YouTube embed URL, null when youtube_id is missing */
export function embedUrl(seg: { youtube_id?: string; start?: number; end?: number }): string | null {
  if (!seg.youtube_id) return null;
  const s = Math.floor(seg.start ?? 0);
  const e = Math.ceil(seg.end ?? 0);
  return `https://www.youtube-nocookie.com/embed/${seg.youtube_id}?start=${s}&end=${e}&rel=0`;
}

/** Standard Hafs ayah counts for surahs 1–114 */
export const AYAH_COUNTS: number[] = [
  7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
  123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
  112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
  34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
  54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
  60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
  14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
  28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
  29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
  15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
  11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
  5, 4, 5, 6
];
// Note: 114 surahs total, sum = 6236

export interface SurahMeta {
  num: number;
  nameAr: string;
  nameEn: string;
  ayahs: number;
  type: "meccan" | "medinan";
}

export const QUICK_SURAHS = [1, 2, 18, 36, 55, 67, 112, 114];

export const SURAHS_META: SurahMeta[] = [
  { num: 1, nameAr: "الفاتحة", nameEn: "Al-Fatihah", ayahs: 7, type: "meccan" },
  { num: 2, nameAr: "البقرة", nameEn: "Al-Baqarah", ayahs: 286, type: "medinan" },
  { num: 3, nameAr: "آل عمران", nameEn: "Ali 'Imran", ayahs: 200, type: "medinan" },
  { num: 4, nameAr: "النساء", nameEn: "An-Nisa'", ayahs: 176, type: "medinan" },
  { num: 5, nameAr: "المائدة", nameEn: "Al-Ma'idah", ayahs: 120, type: "medinan" },
  { num: 6, nameAr: "الأنعام", nameEn: "Al-An'am", ayahs: 165, type: "meccan" },
  { num: 7, nameAr: "الأعراف", nameEn: "Al-A'raf", ayahs: 206, type: "meccan" },
  { num: 8, nameAr: "الأنفال", nameEn: "Al-Anfal", ayahs: 75, type: "medinan" },
  { num: 9, nameAr: "التوبة", nameEn: "At-Tawbah", ayahs: 129, type: "medinan" },
  { num: 10, nameAr: "يونس", nameEn: "Yunus", ayahs: 109, type: "meccan" },
  { num: 11, nameAr: "هود", nameEn: "Hud", ayahs: 123, type: "meccan" },
  { num: 12, nameAr: "يوسف", nameEn: "Yusuf", ayahs: 111, type: "meccan" },
  { num: 13, nameAr: "الرعد", nameEn: "Ar-Ra'd", ayahs: 43, type: "medinan" },
  { num: 14, nameAr: "إبراهيم", nameEn: "Ibrahim", ayahs: 52, type: "meccan" },
  { num: 15, nameAr: "الحجر", nameEn: "Al-Hijr", ayahs: 99, type: "meccan" },
  { num: 16, nameAr: "النحل", nameEn: "An-Nahl", ayahs: 128, type: "meccan" },
  { num: 17, nameAr: "الإسراء", nameEn: "Al-Isra'", ayahs: 111, type: "meccan" },
  { num: 18, nameAr: "الكهف", nameEn: "Al-Kahf", ayahs: 110, type: "meccan" },
  { num: 19, nameAr: "مريم", nameEn: "Maryam", ayahs: 98, type: "meccan" },
  { num: 20, nameAr: "طه", nameEn: "Ta-Ha", ayahs: 135, type: "meccan" },
  { num: 21, nameAr: "الأنبياء", nameEn: "Al-Anbiya'", ayahs: 112, type: "meccan" },
  { num: 22, nameAr: "الحج", nameEn: "Al-Hajj", ayahs: 78, type: "medinan" },
  { num: 23, nameAr: "المؤمنون", nameEn: "Al-Mu'minun", ayahs: 118, type: "meccan" },
  { num: 24, nameAr: "النور", nameEn: "An-Nur", ayahs: 64, type: "medinan" },
  { num: 25, nameAr: "الفرقان", nameEn: "Al-Furqan", ayahs: 77, type: "meccan" },
  { num: 26, nameAr: "الشعراء", nameEn: "Ash-Shu'ara'", ayahs: 227, type: "meccan" },
  { num: 27, nameAr: "النمل", nameEn: "An-Naml", ayahs: 93, type: "meccan" },
  { num: 28, nameAr: "القصص", nameEn: "Al-Qasas", ayahs: 88, type: "meccan" },
  { num: 29, nameAr: "العنكبوت", nameEn: "Al-'Ankabut", ayahs: 69, type: "meccan" },
  { num: 30, nameAr: "الروم", nameEn: "Ar-Rum", ayahs: 60, type: "meccan" },
  { num: 31, nameAr: "لقمان", nameEn: "Luqman", ayahs: 34, type: "meccan" },
  { num: 32, nameAr: "السجدة", nameEn: "As-Sajdah", ayahs: 30, type: "meccan" },
  { num: 33, nameAr: "الأحزاب", nameEn: "Al-Ahzab", ayahs: 73, type: "medinan" },
  { num: 34, nameAr: "سبأ", nameEn: "Saba'", ayahs: 54, type: "meccan" },
  { num: 35, nameAr: "فاطر", nameEn: "Fatir", ayahs: 45, type: "meccan" },
  { num: 36, nameAr: "يس", nameEn: "Ya-Sin", ayahs: 83, type: "meccan" },
  { num: 37, nameAr: "الصافات", nameEn: "As-Saffat", ayahs: 182, type: "meccan" },
  { num: 38, nameAr: "ص", nameEn: "Sad", ayahs: 88, type: "meccan" },
  { num: 39, nameAr: "الزمر", nameEn: "Az-Zumar", ayahs: 75, type: "meccan" },
  { num: 40, nameAr: "غافر", nameEn: "Ghafir", ayahs: 85, type: "meccan" },
  { num: 41, nameAr: "فصلت", nameEn: "Fussilat", ayahs: 54, type: "meccan" },
  { num: 42, nameAr: "الشورى", nameEn: "Ash-Shura", ayahs: 53, type: "meccan" },
  { num: 43, nameAr: "الزخرف", nameEn: "Az-Zukhruf", ayahs: 89, type: "meccan" },
  { num: 44, nameAr: "الدخان", nameEn: "Ad-Dukhan", ayahs: 59, type: "meccan" },
  { num: 45, nameAr: "الجاثية", nameEn: "Al-Jathiyah", ayahs: 37, type: "meccan" },
  { num: 46, nameAr: "الأحقاف", nameEn: "Al-Ahqaf", ayahs: 35, type: "meccan" },
  { num: 47, nameAr: "محمد", nameEn: "Muhammad", ayahs: 38, type: "medinan" },
  { num: 48, nameAr: "الفتح", nameEn: "Al-Fath", ayahs: 29, type: "medinan" },
  { num: 49, nameAr: "الحجرات", nameEn: "Al-Hujurat", ayahs: 18, type: "medinan" },
  { num: 50, nameAr: "ق", nameEn: "Qaf", ayahs: 45, type: "meccan" },
  { num: 51, nameAr: "الذاريات", nameEn: "Adh-Dhariyat", ayahs: 60, type: "meccan" },
  { num: 52, nameAr: "الطور", nameEn: "At-Tur", ayahs: 49, type: "meccan" },
  { num: 53, nameAr: "النجم", nameEn: "An-Najm", ayahs: 62, type: "meccan" },
  { num: 54, nameAr: "القمر", nameEn: "Al-Qamar", ayahs: 55, type: "meccan" },
  { num: 55, nameAr: "الرحمن", nameEn: "Ar-Rahman", ayahs: 78, type: "medinan" },
  { num: 56, nameAr: "الواقعة", nameEn: "Al-Waqi'ah", ayahs: 96, type: "meccan" },
  { num: 57, nameAr: "الحديد", nameEn: "Al-Hadid", ayahs: 29, type: "medinan" },
  { num: 58, nameAr: "المجادلة", nameEn: "Al-Mujadilah", ayahs: 22, type: "medinan" },
  { num: 59, nameAr: "الحشر", nameEn: "Al-Hashr", ayahs: 24, type: "medinan" },
  { num: 60, nameAr: "الممتحنة", nameEn: "Al-Mumtahanah", ayahs: 13, type: "medinan" },
  { num: 61, nameAr: "الصف", nameEn: "As-Saff", ayahs: 14, type: "medinan" },
  { num: 62, nameAr: "الجمعة", nameEn: "Al-Jumu'ah", ayahs: 11, type: "medinan" },
  { num: 63, nameAr: "المنافقون", nameEn: "Al-Munafiqun", ayahs: 11, type: "medinan" },
  { num: 64, nameAr: "التغابن", nameEn: "At-Taghabun", ayahs: 18, type: "medinan" },
  { num: 65, nameAr: "الطلاق", nameEn: "At-Talaq", ayahs: 12, type: "medinan" },
  { num: 66, nameAr: "التحريم", nameEn: "At-Tahrim", ayahs: 12, type: "medinan" },
  { num: 67, nameAr: "الملك", nameEn: "Al-Mulk", ayahs: 30, type: "meccan" },
  { num: 68, nameAr: "القلم", nameEn: "Al-Qalam", ayahs: 52, type: "meccan" },
  { num: 69, nameAr: "الحاقة", nameEn: "Al-Haqqah", ayahs: 52, type: "meccan" },
  { num: 70, nameAr: "المعارج", nameEn: "Al-Ma'arij", ayahs: 44, type: "meccan" },
  { num: 71, nameAr: "نوح", nameEn: "Nuh", ayahs: 28, type: "meccan" },
  { num: 72, nameAr: "الجن", nameEn: "Al-Jinn", ayahs: 28, type: "meccan" },
  { num: 73, nameAr: "المزمل", nameEn: "Al-Muzzammil", ayahs: 20, type: "meccan" },
  { num: 74, nameAr: "المدثر", nameEn: "Al-Muddaththir", ayahs: 56, type: "meccan" },
  { num: 75, nameAr: "القيامة", nameEn: "Al-Qiyamah", ayahs: 40, type: "meccan" },
  { num: 76, nameAr: "الإنسان", nameEn: "Al-Insan", ayahs: 31, type: "medinan" },
  { num: 77, nameAr: "المرسلات", nameEn: "Al-Mursalat", ayahs: 50, type: "meccan" },
  { num: 78, nameAr: "النبأ", nameEn: "An-Naba'", ayahs: 40, type: "meccan" },
  { num: 79, nameAr: "النازعات", nameEn: "An-Nazi'at", ayahs: 42, type: "meccan" },
  { num: 80, nameAr: "عبس", nameEn: "'Abasa", ayahs: 42, type: "meccan" },
  { num: 81, nameAr: "التكوير", nameEn: "At-Takwir", ayahs: 29, type: "meccan" },
  { num: 82, nameAr: "الانفطار", nameEn: "Al-Infitar", ayahs: 19, type: "meccan" },
  { num: 83, nameAr: "المطففين", nameEn: "Al-Mutaffifin", ayahs: 36, type: "meccan" },
  { num: 84, nameAr: "الانشقاق", nameEn: "Al-Inshiqaq", ayahs: 25, type: "meccan" },
  { num: 85, nameAr: "البروج", nameEn: "Al-Buruj", ayahs: 22, type: "meccan" },
  { num: 86, nameAr: "الطارق", nameEn: "At-Tariq", ayahs: 17, type: "meccan" },
  { num: 87, nameAr: "الأعلى", nameEn: "Al-A'la", ayahs: 19, type: "meccan" },
  { num: 88, nameAr: "الغاشية", nameEn: "Al-Ghashiyah", ayahs: 26, type: "meccan" },
  { num: 89, nameAr: "الفجر", nameEn: "Al-Fajr", ayahs: 30, type: "meccan" },
  { num: 90, nameAr: "البلد", nameEn: "Al-Balad", ayahs: 20, type: "meccan" },
  { num: 91, nameAr: "الشمس", nameEn: "Ash-Shams", ayahs: 15, type: "meccan" },
  { num: 92, nameAr: "الليل", nameEn: "Al-Layl", ayahs: 21, type: "meccan" },
  { num: 93, nameAr: "الضحى", nameEn: "Ad-Duha", ayahs: 11, type: "meccan" },
  { num: 94, nameAr: "الشرح", nameEn: "Ash-Sharh", ayahs: 8, type: "meccan" },
  { num: 95, nameAr: "التين", nameEn: "At-Tin", ayahs: 8, type: "meccan" },
  { num: 96, nameAr: "العلق", nameEn: "Al-'Alaq", ayahs: 19, type: "meccan" },
  { num: 97, nameAr: "القدر", nameEn: "Al-Qadr", ayahs: 5, type: "meccan" },
  { num: 98, nameAr: "البينة", nameEn: "Al-Bayyinah", ayahs: 8, type: "medinan" },
  { num: 99, nameAr: "الزلزلة", nameEn: "Az-Zalzalah", ayahs: 8, type: "medinan" },
  { num: 100, nameAr: "العاديات", nameEn: "Al-'Adiyat", ayahs: 11, type: "meccan" },
  { num: 101, nameAr: "القارعة", nameEn: "Al-Qari'ah", ayahs: 11, type: "meccan" },
  { num: 102, nameAr: "التكاثر", nameEn: "At-Takathur", ayahs: 8, type: "meccan" },
  { num: 103, nameAr: "العصر", nameEn: "Al-'Asr", ayahs: 3, type: "meccan" },
  { num: 104, nameAr: "الهمزة", nameEn: "Al-Humazah", ayahs: 9, type: "meccan" },
  { num: 105, nameAr: "الفيل", nameEn: "Al-Fil", ayahs: 5, type: "meccan" },
  { num: 106, nameAr: "قريش", nameEn: "Quraysh", ayahs: 4, type: "meccan" },
  { num: 107, nameAr: "الماعون", nameEn: "Al-Ma'un", ayahs: 7, type: "meccan" },
  { num: 108, nameAr: "الكوثر", nameEn: "Al-Kawthar", ayahs: 3, type: "meccan" },
  { num: 109, nameAr: "الكافرون", nameEn: "Al-Kafirun", ayahs: 6, type: "meccan" },
  { num: 110, nameAr: "النصر", nameEn: "An-Nasr", ayahs: 3, type: "medinan" },
  { num: 111, nameAr: "المسد", nameEn: "Al-Masad", ayahs: 5, type: "meccan" },
  { num: 112, nameAr: "الإخلاص", nameEn: "Al-Ikhlas", ayahs: 4, type: "meccan" },
  { num: 113, nameAr: "الفلق", nameEn: "Al-Falaq", ayahs: 5, type: "meccan" },
  { num: 114, nameAr: "الناس", nameEn: "An-Nas", ayahs: 6, type: "meccan" }
];
