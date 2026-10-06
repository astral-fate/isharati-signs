"""How much of the Turkish corpus's uncovered content would each external TİD dataset fill?

Follows scripts/eval/corpus_eda.py's "tr" block: corpus data/tr/corpus.jsonl, words by tr_norm, content = not in
TR_STOP, a word is covered when one of its tr_forms (word, first five letters) is a form of a lexicon gloss. The lexicon
is tid_lexicon() (word signs, not letters). Dataset glosses are often ASCII-folded (acikmak, Aciklamak); they are
restored to Turkish spelling from a Turkish word list (F:/tid_datasets/_shared/tr_words.txt, mertemin/turkish-word-list)
and the corpus vocabulary, preferring the spelling most frequent in the corpus.

  .venv/Scripts/python scripts/lexicon/tid_datasets/gap_overlap.py
Output: F:/tid_datasets/_shared/gap_report.json, gloss_restore.json, and a printed table.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.retrieval import turkish_urdu as M  # noqa: E402
from isharati.lexicon.turkish import tid_lexicon  # noqa: E402

SHARED = Path("F:/tid_datasets/_shared")
FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u", "̇": ""})
ISLAMIC = ("namaz abdest oruç zekat zekât hac hacı allah peygamber kuran dua cami mescit melek şeytan cennet "
           "cehennem iman günah sevap helal haram bayram inşallah maşallah şükür şükretmek ramazan kurban ahiret "
           "secde rab rabb tövbe sadaka kıble ezan imam hadis sünnet farz mübarek cuma hayır ibadet din dinî "
           "kıyamet ölüm ölmek miras nikah nikâh evlenmek boşanmak yetim fakir zengin sabır sabretmek").split()


def fold(s):
    return M.tr_norm(s).translate(FOLD).replace(" ", "")


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def corpus_counts():
    c = Counter()
    for r in load(DATA / "tr" / "corpus.jsonl"):
        c.update(w for w in M.tr_norm(r["text"]).split() if w.isalpha())
    return c


def lexicon_forms():
    lex = tid_lexicon()
    forms = set()
    for k in lex._signs:
        if k and " " not in k:
            forms |= M.tr_forms(k)
    return forms, len(lex._signs)


# Word-level fixes where the corpus-frequency choice is wrong for the dataset's meaning (checked against the datasets'
# English glosses: AUTSL kira=rent, hali=carpet; Bosphorus Dis=teeth, Bas=head, Alim=purchase, Kar=profit, ...)
OVERRIDE = {"kira": "kira", "hali": "halı", "ataturk": "atatürk", "hoscakal": "hoşçakal", "yarabandi": "yarabandı",
            "ad": "ad", "alim": "alım", "arsa": "arsa", "fon": "fon", "dis": "diş", "bas": "baş", "obur": "obur",
            "surat": "surat", "surup": "şurup", "kesik": "kesik", "test": "test", "su": "su", "asi": "aşı",
            "ilik": "ilik", "kar": "kâr", "halsizlik": "halsizlik", "tahlili": "tahlili", "istirahati": "istirahati",
            "tayini": "tayini", "vasi": "vasi", "gurubu": "grubu", "sismesi": "şişmesi", "alginligi": "algınlığı",
            "degnegi": "değneği", "kaybetmis": "kaybetmiş", "toplamis": "toplamış", "hastasi": "hastası",
            "karti": "kartı", "masrafi": "masrafı", "gorevlisi": "görevlisi", "cuzdani": "cüzdanı",
            "numarasi": "numarası", "sigortasi": "sigortası", "siramatik": "sıramatik", "hosbulduk": "hoşbulduk",
            "hosgeldiniz": "hoşgeldiniz", "ozurdilerim": "özürdilerim", "tesekkurler": "teşekkürler",
            "sinifta": "sınıfta", "sinirli": "sinirli", "sakin": "sakin", "arttirma": "artırma"}


class Restorer:
    """ASCII-folded Turkish word -> Turkish spelling."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.by_fold = {}
        words = set(corpus)
        p = SHARED / "tr_words.txt"
        if p.exists():
            words |= {M.tr_norm(w) for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}
        for w in words:
            self.by_fold.setdefault(fold(w), set()).add(w)

    def word(self, a):
        a = M.tr_norm(a)
        if fold(a) in OVERRIDE:
            return OVERRIDE[fold(a)], True
        cands = self.by_fold.get(fold(a))
        if not cands:
            return a, False
        return max(cands, key=lambda w: (self.corpus.get(w, 0), w != a, -len(w))), True

    def phrase(self, g):
        """Dataset gloss -> (Turkish gloss, restored?). Drops variant suffixes (_2) and parentheticals."""
        g = re.sub(r"\(.*?\)", " ", g)
        g = re.sub(r"_\d+$", "", g.strip())
        g = g.replace("_", " ").replace("-", " ")
        out, ok = [], True
        for w in g.split():
            # a full phrase lookup first (aba güreşi), else word by word
            r, f = self.word(w)
            out.append(r)
            ok &= f
        return " ".join(out), ok


# ---- dataset gloss lists -------------------------------------------------------------------------------------------
def autsl():
    rows = list(csv.DictReader(open("F:/tid_datasets/autsl/SignList_ClassId_TR_EN.csv", encoding="utf-8")))
    return [r["TR"] for r in rows]


def bosphorus():
    rows = list(csv.DictReader(open("F:/tid_datasets/bosphorussign22k/BosphorusSign22k.csv", encoding="utf-8")))
    out = []
    for r in rows:
        # "Baba_Erkek", "Kadin_Kiz": one class glossed by two words -> both
        name = re.sub(r"_\d+$", "", r["ClassName_tr"])
        for part in name.split("_"):
            if part not in out:
                out.append(part)
    return out


def turksign446_partial():
    """TurkSign446's 446-word list is not public (Zenodo record is access-restricted). These are the words named in
    the authors' Create Dataset.ipynb comments: a partial list, for a lower bound only."""
    return ("gulegule gunaydin hosbulduk hosgeldiniz merhaba ozurdilerim neden tesekkurler yapmak siyah kopek neseli "
            "beyaz degil hangi adam heyecanli tembel saygi var efendi dede tanimiyor yer kiz kedi bir sinifta bes "
            "ogrenci akilli sinirli yaramaz hediye kutuda burda yapiyor dondurmaci engelli sinif sakin kim gordu ne "
            "yardim paylasmak verdi getirdi nasil merakli").split()


DATASETS = {"AUTSL": autsl, "BosphorusSign22k": bosphorus, "TurkSign446 (partial, 46/446)": turksign446_partial}


def main():
    corpus = corpus_counts()
    content = {w: n for w, n in corpus.items() if w not in M.TR_STOP}
    total = sum(content.values())
    forms, n_lex = lexicon_forms()
    covered = lambda w, fs: any(f in fs for f in M.tr_forms(w))  # noqa: E731
    base = sum(n for w, n in content.items() if covered(w, forms))
    gaps = Counter({w: n for w, n in content.items() if not covered(w, forms)})
    R = Restorer(corpus)
    report = {"lexicon_glosses": n_lex, "content_tokens": total, "coverage": round(base / total, 4),
              "top_gaps": gaps.most_common(300), "datasets": {}}
    restore_log = {}
    print(f"lexicon {n_lex} glosses; coverage {base / total:.4f} of {total} content tokens")
    for name, fn in DATASETS.items():
        try:
            glosses = fn()
        except FileNotFoundError as e:
            print(name, "missing", e)
            continue
        rest, single = {}, set()
        for g in glosses:
            t, ok = R.phrase(g)
            rest[g] = (t, ok)
            if " " not in t:
                single.add(t)
        restore_log[name] = rest
        new = {g for g in single if not (M.tr_forms(g) & forms)}
        dforms = set().union(*(M.tr_forms(g) for g in new)) if new else set()
        filled = Counter({w: n for w, n in gaps.items() if covered(w, dforms)})
        gain = sum(filled.values())
        # gloss -> corpus tokens it fills
        per = Counter()
        for w, n in filled.items():
            for g in new:
                if M.tr_forms(w) & M.tr_forms(g):
                    per[g] += n
                    break
        isl = sorted({g for g in new if any(g == k or (len(k) >= 4 and g.startswith(k)) for k in ISLAMIC)})
        report["datasets"][name] = {
            "glosses": len(set(glosses)), "single_word": len(single), "multiword": len({t for t, _ in rest.values()}) - len(single),
            "unrestored": sorted(g for g, (t, ok) in rest.items() if not ok),
            "new_to_lexicon": len(new), "fill_gap_glosses": len(per), "gap_tokens_filled": gain,
            "coverage_after": round((base + gain) / total, 4), "gain_pp": round(100 * gain / total, 2),
            "top_fill": per.most_common(60), "islamic_new": isl, "new_glosses": sorted(new)}
        print(f"{name:32s} glosses {len(set(glosses)):4d} new {len(new):4d} filling {len(per):4d} "
              f"+{gain:6d} tok  {base / total:.4f} -> {(base + gain) / total:.4f}  islamic {isl}")
    SHARED.mkdir(parents=True, exist_ok=True)
    (SHARED / "gap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (SHARED / "gloss_restore.json").write_text(json.dumps(restore_log, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
