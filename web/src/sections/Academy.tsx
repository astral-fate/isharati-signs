import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { getAcademySign, getCurriculum, getHadithList, type AcademyItem, type Curriculum } from "../api";
import { AvatarView } from "../components/AvatarView";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { HadithListItem, PoseFrames } from "../types";
import { hadithHref } from "./hadithData";

// Isharati Academy: learning paths of faith-based signs, lessons (avatar demonstration, then a quiz), a practice hub
// (flashcards, matching) and a dictionary. Every sign is a real recording from the lexicons; progress, XP, streak and
// the signs a learner missed (spaced repetition) are kept in this browser.

type SL = "ar" | "en" | "tr" | "ur";
const SL_LABEL: Record<SL, string> = { ar: "ArSL", en: "ASL", tr: "TİD", ur: "ISL/PSL" };
const KEY = "isharati.academy";
const HADITH_SL: Partial<Record<SL, string>> = { ar: "ArSL", tr: "TİD" };  // the signed-hadith dataset's sign languages
const QURAN_LANGS: SL[] = ["ar", "tr"];                                      // Sign Mushaf sources: ArSL, and Diyanet (TİD)
const ModeCtx = createContext<"avatar" | "keypoints">("avatar");  // the signer as the avatar or as keypoints (reference / debugging)
const textDir = (l: string) => (l === "ar" || l === "ur" ? "rtl" : "ltr");
/** the item written in a sign language's spoken language (Turkish under TİD, Arabic under ArSL, ...) */
const tx = (i: AcademyItem, l: string) => i.texts?.[l] || i.meaning;
function Txt({ item, lang, big = false }: { item: AcademyItem; lang: string; big?: boolean }) {
  const text = tx(item, lang);
  return big
    ? <><h3 className="ac-word" dir={textDir(lang)} lang={lang}>{text}</h3>{text !== item.meaning && <p className="ac-meaning">{item.meaning}</p>}</>
    : <span dir={textDir(lang)} lang={lang}>{text}</span>;
}

const S: Record<string, Record<string, string>> = {
  title: { en: "Isharati Academy", ar: "أكاديمية إشارتي", tr: "İşaretim Akademi", ur: "اشارتی اکیڈمی" },
  lead: { en: "Learn Islamic signs step by step: watch the avatar, then test yourself.", ar: "تعلّم الإشارات الإسلامية خطوة بخطوة: شاهد الشخصية ثم اختبر نفسك.", tr: "İslami işaretleri adım adım öğrenin: avatarı izleyin, sonra kendinizi sınayın.", ur: "اسلامی اشارے قدم بہ قدم سیکھیں: اوتار دیکھیں، پھر خود کو آزمائیں۔" },
  path: { en: "Learning path", ar: "مسار التعلّم", tr: "Öğrenme yolu", ur: "سیکھنے کا راستہ" },
  practice: { en: "Practice", ar: "التدريب", tr: "Alıştırma", ur: "مشق" },
  dictionary: { en: "Dictionary", ar: "القاموس", tr: "Sözlük", ur: "لغت" },
  streak: { en: "day streak", ar: "أيام متتالية", tr: "gün seri", ur: "دن مسلسل" },
  xp: { en: "XP", ar: "نقطة", tr: "XP", ur: "XP" },
  learned: { en: "signs learned", ar: "إشارة مُتعلَّمة", tr: "öğrenilen işaret", ur: "سیکھے گئے اشارے" },
  signLang: { en: "Sign language", ar: "لغة الإشارة", tr: "İşaret dili", ur: "اشاروں کی زبان" },
  lesson: { en: "Lesson", ar: "الدرس", tr: "Ders", ur: "سبق" },
  start: { en: "Start", ar: "ابدأ", tr: "Başla", ur: "شروع کریں" },
  review: { en: "Review", ar: "مراجعة", tr: "Tekrar", ur: "دہرائیں" },
  locked: { en: "Finish the previous lesson to unlock", ar: "أكمل الدرس السابق لفتح هذا الدرس", tr: "Kilidi açmak için önceki dersi bitirin", ur: "کھولنے کے لیے پچھلا سبق مکمل کریں" },
  next: { en: "Next", ar: "التالي", tr: "İleri", ur: "اگلا" },
  quiz: { en: "Quiz: what does this sign mean?", ar: "اختبار: ما معنى هذه الإشارة؟", tr: "Sınav: bu işaret ne anlama geliyor?", ur: "کوئز: اس اشارے کا مطلب کیا ہے؟" },
  correct: { en: "Correct!", ar: "إجابة صحيحة!", tr: "Doğru!", ur: "درست!" },
  wrong: { en: "Not quite: it means", ar: "ليست صحيحة: معناها", tr: "Tam değil: anlamı", ur: "درست نہیں: اس کا مطلب" },
  done: { en: "Lesson complete", ar: "اكتمل الدرس", tr: "Ders tamamlandı", ur: "سبق مکمل" },
  back: { en: "Back to the path", ar: "العودة إلى المسار", tr: "Yola dön", ur: "راستے پر واپس" },
  otherLangs: { en: "See it in another sign language:", ar: "شاهدها بلغة إشارة أخرى:", tr: "Başka bir işaret dilinde görün:", ur: "کسی اور اشاروں کی زبان میں دیکھیں:" },
  flash: { en: "Flashcards", ar: "البطاقات", tr: "Kartlar", ur: "فلیش کارڈ" },
  match: { en: "Sign matching", ar: "مطابقة الإشارات", tr: "İşaret eşleştirme", ur: "اشاروں کا ملاپ" },
  flip: { en: "Show the meaning", ar: "اعرض المعنى", tr: "Anlamı göster", ur: "مطلب دکھائیں" },
  knew: { en: "I knew it", ar: "عرفتها", tr: "Bildim", ur: "مجھے معلوم تھا" },
  again: { en: "Again later", ar: "أعدها لاحقًا", tr: "Sonra tekrar", ur: "بعد میں دوبارہ" },
  matchHelp: { en: "Pick a sign, then the meaning it shows.", ar: "اختر إشارة، ثم المعنى الذي تدل عليه.", tr: "Bir işaret seçin, sonra gösterdiği anlamı.", ur: "ایک اشارہ چنیں، پھر اس کا مطلب۔" },
  newRound: { en: "New round", ar: "جولة جديدة", tr: "Yeni tur", ur: "نیا راؤنڈ" },
  search: { en: "Search signs…", ar: "ابحث عن إشارة…", tr: "İşaret ara…", ur: "اشارہ تلاش کریں…" },
  none: { en: "No signs recorded in this sign language for this lesson yet.", ar: "لا توجد إشارات مسجّلة بهذه اللغة لهذا الدرس بعد.", tr: "Bu ders için bu işaret dilinde henüz kayıtlı işaret yok.", ur: "اس سبق کے لیے اس زبان میں ابھی کوئی ریکارڈ شدہ اشارہ نہیں۔" },
  avatarMode: { en: "Avatar", ar: "الشخصية", tr: "Avatar", ur: "اوتار" },
  keypointsMode: { en: "Keypoints", ar: "النقاط المفصلية", tr: "Eklem noktaları", ur: "کی پوائنٹس" },
  loading: { en: "Loading…", ar: "جارٍ التحميل…", tr: "Yükleniyor…", ur: "لوڈ ہو رہا ہے…" },
  hadithPath: { en: "Hadith", ar: "الحديث", tr: "Hadis", ur: "حدیث" },
  hadithLead: { en: "Whole hadith signed by interpreters for the Deaf, An-Nawawi's Forty first.", ar: "أحاديث كاملة بإشارة مترجمين للصمّ، تبدأ بالأربعين النووية.", tr: "Sağırlar için tercümanların işaretlediği hadisler, önce Nevevî'nin Kırk Hadisi.", ur: "بہروں کے ترجمانوں کی اشاروں میں مکمل احادیث، پہلے اربعین نووی۔" },
  quranPath: { en: "Qur'an", ar: "القرآن الكريم", tr: "Kur'an", ur: "قرآن" },
  quranLead: { en: "The Sign Mushaf: signed recitation and tafsir, ayah by ayah.", ar: "المصحف الإشاري: التلاوة والتفسير بالإشارة، آيةً آية.", tr: "İşaretli Mushaf: ayet ayet işaretli tilavet ve tefsir.", ur: "اشاراتی مصحف: آیت بہ آیت اشاروں میں تلاوت اور تفسیر۔" },
  quranOpen: { en: "Open the Sign Mushaf", ar: "افتح المصحف الإشاري", tr: "İşaretli Mushaf'ı aç", ur: "اشاراتی مصحف کھولیں" },
  hadithNo: { en: "Hadith", ar: "الحديث", tr: "Hadis", ur: "حدیث" },
  hadithOpen: { en: "Watch", ar: "شاهد", tr: "İzle", ur: "دیکھیں" },
  hadithAll: { en: "Browse all signed hadith", ar: "تصفّح كل الأحاديث بالإشارة", tr: "Tüm işaretli hadislere göz at", ur: "تمام اشاراتی احادیث دیکھیں" },
};

type Progress = { done: Record<string, boolean>; xp: number; streak: number; last: string; learned: Record<string, boolean>; misses: Record<string, number> };
const EMPTY: Progress = { done: {}, xp: 0, streak: 0, last: "", learned: {}, misses: {} };
const today = () => new Date().toISOString().slice(0, 10);

function loadProgress(): Progress {
  try { return { ...EMPTY, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return { ...EMPTY }; }
}

function useProgress() {
  const [p, setP] = useState<Progress>(loadProgress);
  const save = (f: (q: Progress) => Progress) => setP((q) => {
    const n = f(q);
    try { localStorage.setItem(KEY, JSON.stringify(n)); } catch { /* private window: progress lives for this visit */ }
    return n;
  });
  const touchStreak = (q: Progress): Progress => {   // a day with at least one lesson or practice keeps the streak
    const d = today();
    if (q.last === d) return q;
    const y = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
    return { ...q, last: d, streak: q.last === y ? q.streak + 1 : 1 };
  };
  return { p, save, touchStreak };
}

function shuffle<T>(a: T[]): T[] {
  const b = [...a];
  for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; }
  return b;
}

/** One item's sign, looping, on the chosen avatar (the skeleton when small or when the avatar cannot load). */
function SignPlayer({ item, lang, small = false }: { item: AcademyItem; lang: SL; small?: boolean }) {
  const { avatar, outfit } = useAppState();
  const mode = useContext(ModeCtx);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    setFrames(null);
    getAcademySign(lang, item.id).then((f) => live && setFrames(f), () => live && setFrames(null));
    return () => { live = false; };
  }, [item.id, lang]);
  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { loop: true, autoplay: true });
  return (
    <div className={"viewer ac-viewer" + (small ? " small" : "")}>
      {!frames ? <div className="ac-loading">…</div>
        : small || failed || mode === "keypoints" ? <KeypointCanvas frames={frames} time={player.t} />
        : <AvatarView model={avatar} outfit={outfit} frames={frames} time={player.t} onError={() => setFailed(true)} />}
    </div>
  );
}

export function Academy() {
  const { ui, dir } = useI18n();
  const s = (k: string) => S[k]?.[ui] ?? S[k]?.en ?? k;
  const [cur, setCur] = useState<Curriculum | null>(null);
  const [tab, setTab] = useState<"path" | "practice" | "dict">("path");
  const [lang, setLang] = useState<SL>(((["ar", "en", "tr", "ur"] as SL[]).includes(ui as SL) ? ui : "en") as SL);
  const [lessonId, setLessonId] = useState<string | null>(null);
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const prog = useProgress();
  useEffect(() => { getCurriculum(ui).then(setCur, () => setCur(null)); }, [ui]);

  const lessons = useMemo(() => cur ? cur.paths.flatMap((p) => p.lessons.map((l) => ({ ...l, path: p }))) : [], [cur]);
  const allItems = useMemo(() => {
    const seen = new Map<string, AcademyItem>();
    lessons.forEach((l) => l.items.forEach((i) => seen.set(i.id, i)));
    return [...seen.values()];
  }, [lessons]);
  const learnedCount = Object.keys(prog.p.learned).length;
  // each path opens on its first lesson; inside a path, a lesson opens when the one before it is done
  const unlocked = (k: number) => k === 0 || lessons[k - 1].path.id !== lessons[k].path.id || !!prog.p.done[lessons[k - 1].id];

  const lesson = lessons.find((l) => l.id === lessonId);
  return (
    <section id="academy" className="academy">
      <div className="wrap">
        <h2>{s("title")}</h2>
        <p className="lead">{s("lead")}</p>
        <div className="ac-stats">
          <span>🔥 <b>{prog.p.streak}</b> {s("streak")}</span>
          <span>⭐ <b>{prog.p.xp}</b> {s("xp")}</span>
          <span>✅ <b>{learnedCount}</b> {s("learned")}</span>
        </div>
        <div className="ac-bar">
          <div className="seg" role="tablist">
            {(["path", "practice", "dict"] as const).map((k) => (
              <button key={k} type="button" role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""}
                      onClick={() => { setTab(k); setLessonId(null); }}>{s(k === "dict" ? "dictionary" : k)}</button>
            ))}
          </div>
          <div className="seg">
            <button type="button" className={mode === "avatar" ? "on" : ""} onClick={() => setMode("avatar")}>{s("avatarMode")}</button>
            <button type="button" className={mode === "keypoints" ? "on" : ""} onClick={() => setMode("keypoints")}>{s("keypointsMode")}</button>
          </div>
          <label className="ac-lang">{s("signLang")}:
            <select value={lang} onChange={(e) => setLang(e.target.value as SL)}>
              {(Object.keys(SL_LABEL) as SL[]).map((l) => <option key={l} value={l}>{SL_LABEL[l]}</option>)}
            </select>
          </label>
        </div>

        {mode === "avatar" && <div className="ac-style"><StyleSwitch /></div>}
        <ModeCtx.Provider value={mode}>
        {!cur ? <p className="status">{s("loading")}</p>
          : lesson ? <Lesson lesson={lesson} lang={lang} s={s} dir={dir} allItems={allItems}
                             onExit={() => setLessonId(null)}
                             onDone={(score, missed) => prog.save((q) => {
                               const n = prog.touchStreak({ ...q, done: { ...q.done, [lesson.id]: true }, xp: q.xp + score * 10 + 20,
                                 learned: { ...q.learned, ...Object.fromEntries(lesson.items.map((i) => [i.id, true])) },
                                 misses: { ...q.misses } });
                               missed.forEach((id) => { n.misses[id] = (n.misses[id] || 0) + 1; });
                               return n;
                             })} />
          : tab === "path" ? (
            <div className="ac-paths">
              {cur.paths.map((p) => (
                <div key={p.id} className="ac-path panel">
                  <h3><span aria-hidden>{p.icon}</span> {p.title}</h3>
                  <div className="ac-nodes">
                    {p.lessons.map((l) => {
                      const k = lessons.findIndex((x) => x.id === l.id), open = unlocked(k), done = !!prog.p.done[l.id];
                      return (
                        <button key={l.id} type="button" className={"ac-node" + (done ? " done" : "") + (open ? "" : " locked")}
                                disabled={!open} title={open ? "" : s("locked")} onClick={() => setLessonId(l.id)}>
                          <span className="ac-node-dot">{done ? "✓" : open ? k + 1 : "🔒"}</span>
                          <span className="ac-node-label">{s("lesson")} {l.id.split("-").pop()}</span>
                          <span className="ac-node-words" dir={textDir(lang)}>{l.items.map((i) => tx(i, lang)).slice(0, 3).join(" · ")}</span>
                          {open && <span className="ac-node-cta">{done ? s("review") : s("start")}</span>}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
              <SacredTexts lang={lang} s={s} />
            </div>
          )
          : tab === "practice" ? <Practice items={allItems} lang={lang} s={s} dir={dir} prog={prog} />
          : <Dictionary paths={cur.paths} lang={lang} s={s} dir={dir} />}
        </ModeCtx.Provider>
      </div>
    </section>
  );
}

/**
 * The sacred texts in the chosen sign language, beside the lesson paths: the Sign Mushaf (ArSL: KFC, al-Mukhtasar,
 * al-Kharj, Tebyan; TİD: Diyanet) and the signed hadith (ArSL, TİD), each a nested page of the Academy. A sign language
 * with no such data shows neither.
 */
function SacredTexts({ lang, s }: { lang: SL; s: (k: string) => string }) {
  const sl = HADITH_SL[lang];
  const [list, setList] = useState<HadithListItem[]>([]);
  useEffect(() => {
    setList([]);
    if (sl) getHadithList("", sl).then(setList, () => setList([]));
  }, [sl]);
  return (
    <>
      {QURAN_LANGS.includes(lang) && (
        <div className="ac-path ac-hadith panel">
          <h3><span aria-hidden>📖</span> {s("quranPath")}</h3>
          <p className="note">{s("quranLead")}</p>
          <a className="btn ghost" href="/#/quran" style={{ display: "inline-block" }}>{s("quranOpen")} →</a>
        </div>
      )}
      {list.length > 0 && (
    <div className="ac-path ac-hadith panel">
      <h3><span aria-hidden>📜</span> {s("hadithPath")}</h3>
      <p className="note">{s("hadithLead")}</p>
      <div className="ac-nodes">
        {list.slice(0, 8).map((h) => (
          <a key={h.ref} className="ac-node" href={hadithHref(h.ref)}>
            <span className="ac-node-dot">{h.number}</span>
            <span className="ac-node-label">{s("hadithNo")} {h.number} · {h.sign_languages.join(" · ")}</span>
            <span className="ac-node-words" dir="rtl" lang="ar">{h.text}</span>
            <span className="ac-node-cta">{s("hadithOpen")}</span>
          </a>
        ))}
      </div>
      <a className="btn ghost" href="/#/hadith" style={{ marginTop: 12, display: "inline-block" }}>{s("hadithAll")} →</a>
    </div>
      )}
    </>
  );
}

function Lesson({ lesson, lang, s, dir, allItems, onExit, onDone }: {
  lesson: { id: string; items: AcademyItem[] }; lang: SL; s: (k: string) => string; dir: string; allItems: AcademyItem[];
  onExit: () => void; onDone: (score: number, missed: string[]) => void;
}) {
  const items = lesson.items.filter((i) => i.langs.includes(lang));
  const [step, setStep] = useState(0);              // 0..n-1 teaching, n..2n-1 quiz, 2n finished
  const [view, setView] = useState<SL>(lang);
  const [pick, setPick] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [missed, setMissed] = useState<string[]>([]);
  const n = items.length;
  const options = useMemo(() => {
    if (step < n || step >= 2 * n) return [];
    const target = items[step - n];
    const pool = shuffle(allItems.filter((i) => i.id !== target.id)).slice(0, 3);
    return shuffle([target, ...pool]);
  }, [step]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setView(lang); }, [step, lang]);
  if (!n) return <div className="panel ac-lesson"><p>{s("none")}</p><button type="button" className="btn ghost" onClick={onExit}>{s("back")}</button></div>;
  if (step >= 2 * n) {
    return (
      <div className="panel ac-lesson ac-finish">
        <div className="ac-big">🎉</div>
        <h3>{s("done")}</h3>
        <p>⭐ +{score * 10 + 20} {s("xp")} · {score}/{n}</p>
        <button type="button" className="btn" onClick={onExit}>{s("back")}</button>
      </div>
    );
  }
  const teaching = step < n;
  const item = teaching ? items[step] : items[step - n];
  return (
    <div className="panel ac-lesson">
      <div className="ac-progress"><div style={{ width: `${(step / (2 * n)) * 100}%` }} /></div>
      {teaching ? (
        <>
          <SignPlayer item={item} lang={view} />
          <Txt item={item} lang={view} big />
          <p className="note"><bdi>{item.words[view]}</bdi> · {item.src?.[view] || SL_LABEL[view]}</p>
          {item.langs.length > 1 && (
            <div className="ac-dialects"><span className="note">{s("otherLangs")}</span>
              {item.langs.map((l) => <button key={l} type="button" className={l === view ? "on" : ""} onClick={() => setView(l as SL)}>{SL_LABEL[l as SL]}</button>)}
            </div>
          )}
          <button type="button" className="btn demo-go" onClick={() => setStep(step + 1)}>{s("next")} →</button>
        </>
      ) : (
        <>
          <p className="ac-q">{s("quiz")}</p>
          <SignPlayer item={item} lang={lang} />
          <div className="ac-options" dir={dir}>
            {options.map((o) => {
              const state = pick ? (o.id === item.id ? " right" : o.id === pick ? " bad" : "") : "";
              return <button key={o.id} type="button" disabled={!!pick} className={"ac-option" + state} onClick={() => {
                setPick(o.id);
                if (o.id === item.id) setScore(score + 1); else setMissed([...missed, item.id]);
              }}><Txt item={o} lang={lang} /></button>;
            })}
          </div>
          {pick && <p className={"status" + (pick === item.id ? "" : " err")}>{pick === item.id ? s("correct") : `${s("wrong")} «${tx(item, lang)}»`}</p>}
          {pick && <button type="button" className="btn demo-go" onClick={() => {
            setPick(null);
            if (step + 1 >= 2 * n) onDone(score, missed);
            setStep(step + 1);
          }}>{s("next")} →</button>}
        </>
      )}
    </div>
  );
}

function Practice({ items, lang, s, dir, prog }: { items: AcademyItem[]; lang: SL; s: (k: string) => string; dir: string; prog: ReturnType<typeof useProgress> }) {
  const [mode, setMode] = useState<"flash" | "match">("flash");
  const usable = items.filter((i) => i.langs.includes(lang));
  // spaced repetition: signs missed more often come first; then signs already learned; then the rest
  const ranked = useMemo(() => [...usable].sort((a, b) =>
    (prog.p.misses[b.id] || 0) - (prog.p.misses[a.id] || 0) || Number(!!prog.p.learned[b.id]) - Number(!!prog.p.learned[a.id])), [lang, items.length]);  // eslint-disable-line react-hooks/exhaustive-deps
  const [k, setK] = useState(0);
  const [shown, setShown] = useState(false);
  const [round, setRound] = useState(0);
  const four = useMemo(() => shuffle(ranked.slice(0, 8)).slice(0, 4), [round, ranked]);
  const meanings = useMemo(() => shuffle(four), [four]);
  const [sel, setSel] = useState<string | null>(null);
  const [matched, setMatched] = useState<Record<string, boolean>>({});
  const [wrongPair, setWrongPair] = useState<string | null>(null);
  if (!usable.length) return <p className="status">{s("none")}</p>;
  const card = ranked[k % ranked.length];
  return (
    <div className="panel ac-lesson">
      <div className="seg" style={{ marginBottom: 14 }}>
        <button type="button" className={mode === "flash" ? "on" : ""} onClick={() => setMode("flash")}>{s("flash")}</button>
        <button type="button" className={mode === "match" ? "on" : ""} onClick={() => setMode("match")}>{s("match")}</button>
      </div>
      {mode === "flash" ? (
        <>
          <SignPlayer item={card} lang={lang} />
          {shown ? <Txt item={card} lang={lang} big />
            : <button type="button" className="btn ghost" onClick={() => setShown(true)}>{s("flip")}</button>}
          {shown && (
            <div className="ac-options two">
              <button type="button" className="ac-option right" onClick={() => {
                prog.save((q) => prog.touchStreak({ ...q, xp: q.xp + 5, misses: { ...q.misses, [card.id]: Math.max(0, (q.misses[card.id] || 0) - 1) } }));
                setShown(false); setK(k + 1);
              }}>{s("knew")}</button>
              <button type="button" className="ac-option bad" onClick={() => {
                prog.save((q) => prog.touchStreak({ ...q, misses: { ...q.misses, [card.id]: (q.misses[card.id] || 0) + 1 } }));
                setShown(false); setK(k + 1);
              }}>{s("again")}</button>
            </div>
          )}
        </>
      ) : (
        <>
          <p className="note">{s("matchHelp")}</p>
          <div className="ac-match">
            {four.map((i, idx) => (
              <button key={i.id} type="button" className={"ac-match-sign" + (sel === i.id ? " on" : "") + (matched[i.id] ? " done" : "")}
                      disabled={!!matched[i.id]} onClick={() => setSel(i.id)}>
                <SignPlayer item={i} lang={lang} small /><span>{String.fromCharCode(65 + idx)}</span>
              </button>
            ))}
          </div>
          <div className="ac-options" dir={dir}>
            {meanings.map((m) => (
              <button key={m.id} type="button" disabled={!!matched[m.id] || !sel}
                      className={"ac-option" + (matched[m.id] ? " right" : wrongPair === m.id ? " bad" : "")}
                      onClick={() => {
                        if (sel === m.id) { setMatched({ ...matched, [m.id]: true }); setSel(null); setWrongPair(null);
                          prog.save((q) => prog.touchStreak({ ...q, xp: q.xp + 5 })); }
                        else { setWrongPair(m.id); if (sel) prog.save((q) => ({ ...q, misses: { ...q.misses, [sel]: (q.misses[sel] || 0) + 1 } })); }
                      }}><Txt item={m} lang={lang} /></button>
            ))}
          </div>
          {Object.keys(matched).length === four.length &&
            <button type="button" className="btn demo-go" onClick={() => { setMatched({}); setSel(null); setRound(round + 1); }}>{s("newRound")}</button>}
        </>
      )}
    </div>
  );
}

function Dictionary({ paths, lang, s, dir }: { paths: Curriculum["paths"]; lang: SL; s: (k: string) => string; dir: string }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<AcademyItem | null>(null);
  const [view, setView] = useState<SL>(lang);
  useEffect(() => setView(lang), [lang, open]);
  const match = (i: AcademyItem) => !q || (i.meaning + " " + Object.values(i.words).join(" ")).toLowerCase().includes(q.toLowerCase());
  return (
    <div>
      <input className="demo-text ac-search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={s("search")} dir={dir} />
      {open && (
        <div className="panel ac-lesson">
          <SignPlayer item={open} lang={open.langs.includes(view) ? view : (open.langs[0] as SL)} />
          <Txt item={open} lang={open.langs.includes(view) ? view : open.langs[0]} big />
          <div className="ac-dialects">{open.langs.map((l) => <button key={l} type="button" className={l === view ? "on" : ""} onClick={() => setView(l as SL)}>{SL_LABEL[l as SL]}</button>)}</div>
        </div>
      )}
      {paths.map((p) => {
        const items = [...new Map(p.lessons.flatMap((l) => l.items).filter(match).map((i) => [i.id, i])).values()];
        if (!items.length) return null;
        return (
          <div key={p.id} className="ac-dict-group">
            <h3><span aria-hidden>{p.icon}</span> {p.title}</h3>
            <div className="ac-dict">
              {items.map((i) => (
                <button key={i.id} type="button" className={"ac-dict-card" + (open?.id === i.id ? " on" : "")} onClick={() => setOpen(i)}>
                  <Txt item={i} lang={lang} />
                  {tx(i, lang) !== i.meaning && <span className="ac-dict-mean" dir={dir}>{i.meaning.split(" (")[0]}</span>}
                  <span className="ac-dict-langs">{i.langs.map((l) => i.src?.[l] || SL_LABEL[l as SL]).join(" · ")}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
