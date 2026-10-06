/**
 * Quran text loader and cache for full Uthmani and clean Arabic Ayah text.
 * Reads /api/quran/text (structured by surah -> ayahs; served from the isharati-app-data dataset).
 */

export interface QuranAyahText {
  surah: number;
  ayah: number;
  surahName: string;
  uthmani: string;
  clean: string;
}

interface RawAyah {
  ayah: number;
  uthmani: string;
  clean: string;
}

interface RawSurah {
  name: string;
  ayahs: RawAyah[];
}

type QuranDatabase = Record<string, RawSurah>;

let dbCache: QuranDatabase | null = null;
let fetchPromise: Promise<QuranDatabase | null> | null = null;

// Embedded fallbacks for instant initial render or offline/testing
const FALLBACKS: Record<string, { surahName: string; ayahs: Record<number, { uthmani: string; clean: string }> }> = {
  "1": {
    surahName: "سُورَةُ ٱلْفَاتِحَةِ",
    ayahs: {
      1: { uthmani: "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", clean: "بسم الله الرحمن الرحيم" },
      2: { uthmani: "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ", clean: "الحمد لله رب العالمين" },
      3: { uthmani: "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", clean: "الرحمن الرحيم" },
      4: { uthmani: "مَٰلِكِ يَوْمِ ٱلدِّينِ", clean: "مالك يوم الدين" },
      5: { uthmani: "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ", clean: "إياك نعبد وإياك نستعين" },
      6: { uthmani: "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ", clean: "اهدنا الصراط المستقيم" },
      7: { uthmani: "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ", clean: "صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين" },
    },
  },
  "112": {
    surahName: "سُورَةُ ٱلْإِخْلَاصِ",
    ayahs: {
      1: { uthmani: "قُلْ هُوَ ٱللَّهُ أَحَدٌ", clean: "قل هو الله أحد" },
      2: { uthmani: "ٱللَّهُ ٱلصَّمَدُ", clean: "الله الصمد" },
      3: { uthmani: "لَمْ يَلِدْ وَلَمْ يُولَدْ", clean: "لم يلد ولم يولد" },
      4: { uthmani: "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ", clean: "ولم يكن له كفوا أحد" },
    },
  },
  "114": {
    surahName: "سُورَةُ ٱلنَّاسِ",
    ayahs: {
      1: { uthmani: "قُلْ أَعُوذُ بِرَبِّ ٱلنَّاسِ", clean: "قل أعوذ برب الناس" },
      2: { uthmani: "مَلِكِ ٱلنَّاسِ", clean: "ملك الناس" },
      3: { uthmani: "إِلَٰهِ ٱلنَّاسِ", clean: "إله الناس" },
      4: { uthmani: "مِن شَرِّ ٱلْوَسْوَاسِ ٱلْخَنَّاسِ", clean: "من شر الوسواس الخناس" },
      5: { uthmani: "ٱلَّذِى يُوَسْوِسُ فِى صُدُورِ ٱلنَّاسِ", clean: "الذي يوسوس في صدور الناس" },
      6: { uthmani: "مِنَ ٱلْجِنَّةِ وَٱلنَّاسِ", clean: "من الجنة والناس" },
    },
  },
};

/** Load the full Quran database into memory */
export async function loadQuranDatabase(): Promise<QuranDatabase | null> {
  if (dbCache) return dbCache;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      const res = await fetch("/api/quran/text");
      if (!res.ok) return null;
      const data: QuranDatabase = await res.json();
      dbCache = data;
      return data;
    } catch {
      return null;
    } finally {
      fetchPromise = null;
    }
  })();

  return fetchPromise;
}

/** Get the Uthmani & clean text for any Surah and Ayah */
export function getAyahTextSync(surah: number, ayah: number): QuranAyahText | null {
  const sStr = String(surah);

  if (dbCache && dbCache[sStr]) {
    const s = dbCache[sStr];
    const match = s.ayahs.find((a) => a.ayah === ayah);
    if (match) {
      return {
        surah,
        ayah,
        surahName: s.name,
        uthmani: match.uthmani,
        clean: match.clean,
      };
    }
  }

  // Check fallbacks
  if (FALLBACKS[sStr] && FALLBACKS[sStr].ayahs[ayah]) {
    const fb = FALLBACKS[sStr];
    const a = fb.ayahs[ayah];
    return {
      surah,
      ayah,
      surahName: fb.surahName,
      uthmani: a.uthmani,
      clean: a.clean,
    };
  }

  return null;
}

/** Async getter ensuring the database is loaded */
export async function getAyahText(surah: number, ayah: number): Promise<QuranAyahText | null> {
  const sync = getAyahTextSync(surah, ayah);
  if (sync) return sync;
  await loadQuranDatabase();
  return getAyahTextSync(surah, ayah);
}
