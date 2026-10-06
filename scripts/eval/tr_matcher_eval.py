"""Turkish word -> TİD sign matching, old rule against new: coverage of the Turkish Qur'an/hadith corpus, the match
route of every covered token, precision samples to judge by hand, and the most frequent uncovered lemmas.

OLD  the rule corpus_eda.py used before (and TIDLexicon until the morphology change): a word is covered when it, or its
     first five letters, equals a single-word gloss or a gloss's first five letters (tr_forms), over the same sources.
NEW  TIDLexicon.match_tokens on tid_lexicon(): exact -> compound -> lemma (zeyrek) -> synonym (synonyms_tr.json) ->
     prefix fallback (words the analyser does not know), multi-word glosses first. Also reported without the
     prefix fallback, to show what it adds.

  PYTHONPATH=src py scripts/eval/tr_matcher_eval.py [--sample]
Output: results/tr_matcher_eval.json; with --sample also results/tr_precision_sample.tsv (150 matched tokens,
stratified by route) and results/tr_precision_old_prefix.tsv (50 tokens the old five-letter rule matched by prefix),
whose "ok" column is filled in by hand; results/tr_uncovered_lemmas.tsv (top 300 uncovered lemmas).
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon.turkish import tid_lexicon  # noqa: E402
from isharati.retrieval.turkish_urdu import TR_STOP, tr_forms  # noqa: E402
from isharati.text import turkish_morph as morph  # noqa: E402
from isharati.text.turkish import tr_norm  # noqa: E402

RES = ROOT / "results"
# case and copula endings that tr_norm splits off after an apostrophe (Allah'ın -> allah ın): parts of a word, not words
SUFFIX_FRAGMENTS = set("a e ı i u ü ya ye yı yi yu yü ın in un ün nın nin nun nün da de ta te dan den tan ten la le "
                       "yla yle ya dır dir dur dür tır tir ki nda nde ndan nden na ne nı ni nu nü dı di ydı ydi".split())
SAMPLE_PLAN = {"exact": 30, "compound": 10, "lemma": 50, "synonym": 35, "prefix": 25}


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def words(text):
    return [w for w in tr_norm(text).split() if w.isalpha()]


def old_forms():
    """The old corpus_eda.py "tr" forms: single-word glosses of the four sources, each as itself and its first five letters."""
    srcs = [DATA / "lexicon_tr" / "dictionaries" / "entries_tid_dictionary.jsonl", DATA / "lexicon_tr" / "names" / "entries.jsonl",
            DATA / "lexicon_tr" / "tid_youtube" / "entries.jsonl", DATA / "lexicon_tr" / "autsl" / "entries.jsonl"]
    forms = {}
    for p in srcs:
        if not p.exists():
            continue
        for e in load(p):
            if e.get("is_letter") or e.get("variant_of"):
                continue
            g = tr_norm(e["gloss"])
            if g and " " not in g:
                for f in tr_forms(g):
                    forms.setdefault(f, g)
    return forms


def old_match(forms, w):
    """(gloss, how) under the old rule: "exact" when the word is a gloss, "prefix5" when only five letters are shared."""
    if w in forms and forms[w] == w:
        return forms[w], "exact"
    for f in tr_forms(w):
        if f in forms:
            return forms[f], ("exact" if forms[f] == w else "prefix5")
    return None, None


def table(cov_tokens, totals, cov_types, types):
    out = {}
    for k in ("quran", "hadith", "all"):
        out[k] = {"content_tokens": totals[k], "content_token_coverage": round(cov_tokens[k] / max(1, totals[k]), 4),
                  "content_types": types[k], "content_type_coverage": round(cov_types[k] / max(1, types[k]), 4)}
    return out


def main(sample=False):
    rows = load(DATA / "tr" / "corpus.jsonl")
    lex = tid_lexicon()
    seqs = [(r["kind"], words(r["text"]), r["text"]) for r in rows]
    counts = {"quran": Counter(), "hadith": Counter()}
    for kind, ws, _ in seqs:
        counts[kind].update(ws)
    counts["all"] = counts["quran"] + counts["hadith"]
    morph.analyze(list(counts["all"]))  # fills the lemma cache once (zeyrek); later runs read the cache
    content = {k: {w: n for w, n in c.items() if w not in TR_STOP} for k, c in counts.items()}
    totals = {k: sum(c.values()) for k, c in content.items()}
    ntypes = {k: len(c) for k, c in content.items()}
    res = {"lexicon_signs": len(lex.sign_entries()), "synonyms_signed": sum(1 for g in lex._aliases.values() if g in lex._signs)}

    # OLD
    forms = old_forms()
    res["old_glosses"] = len(set(forms.values()))
    om = {w: old_match(forms, w) for w in counts["all"]}
    cov_tok = {k: sum(n for w, n in c.items() if om[w][0]) for k, c in content.items()}
    cov_typ = {k: sum(1 for w in c if om[w][0]) for k, c in content.items()}
    res["old"] = table(cov_tok, totals, cov_typ, ntypes)
    res["old"]["by_route_tokens"] = {h: sum(n for w, n in content["all"].items() if om[w][1] == h)
                                     for h in ("exact", "prefix5")}

    # NEW (with and without the prefix fallback)
    for name, fallback in (("new", True), ("new_no_prefix", False)):
        lex.prefix_fallback = fallback
        memo = {}
        cov_tok, routes, examples = Counter(), Counter(), defaultdict(list)
        for kind, ws, text in seqs:
            for s, e, entry, route in lex.match_tokens(ws):
                if e - s > 1:
                    route = "phrase"
                for j in range(s, e):
                    if ws[j] in TR_STOP:
                        continue
                    cov_tok[kind] += 1
                    cov_tok["all"] += 1
                    routes[route] += 1
                    if name == "new":
                        examples[route].append((ws[j], entry.gloss, " ".join(ws[max(0, s - 4):e + 4]), " ".join(ws[s:e])))
        for w in counts["all"]:
            memo[w] = lex.match(w)
        cov_typ = {k: sum(1 for w in c if memo[w][0] is not None) for k, c in content.items()}
        res[name] = table(cov_tok, totals, cov_typ, ntypes)
        res[name]["by_route_tokens"] = dict(routes.most_common())
        res[name]["by_route_types"] = dict(Counter(memo[w][1] for w in content["all"] if memo[w][0] is not None).most_common())
        if name == "new":
            new_memo, new_examples = memo, examples
    lex.prefix_fallback = True

    # the same, apostrophe suffix fragments (Allah'ın -> «ın») not counted as words
    frag_tot = sum(n for w, n in content["all"].items() if w in SUFFIX_FRAGMENTS)
    res["suffix_fragment_tokens"] = frag_tot
    for name in ("old", "new"):
        cov = res[name]["all"]["content_token_coverage"] * totals["all"]
        res[name]["all"]["content_token_coverage_excl_fragments"] = round(cov / (totals["all"] - frag_tot), 4)

    # how the old five-letter rule's prefix matches fare now
    moved = Counter()
    for w, n in content["all"].items():
        if om[w][1] == "prefix5":
            e, route = new_memo[w]
            moved["same sign" if e is not None and tr_norm(e.gloss) == om[w][0] else
                  ("other sign" if e is not None else "now unmatched")] += n
    res["old_prefix5_tokens_now"] = dict(moved)

    # uncovered lemmas: the next signs to collect
    unc = Counter()
    for w, n in content["all"].items():
        if new_memo[w][0] is None and w not in SUFFIX_FRAGMENTS:
            ls = morph.lemmas(w)
            unc[ls[0] if ls else w] += n
    res["top_uncovered_lemmas"] = unc.most_common(30)
    RES.mkdir(exist_ok=True)
    with open(RES / "tr_uncovered_lemmas.tsv", "w", encoding="utf-8") as f:
        f.write("lemma\ttokens\n" + "".join(f"{l}\t{n}\n" for l, n in unc.most_common(300)))

    if sample:
        rnd = random.Random(7)
        with open(RES / "tr_precision_sample.tsv", "w", encoding="utf-8") as f:
            f.write("route\tword\tgloss\tmatched\tcontext\tok\n")
            plan = dict(SAMPLE_PLAN)
            avail = {r: len(new_examples.get(r, [])) for r in list(plan) + ["phrase"]}
            if avail.get("phrase"):
                plan["phrase"] = min(10, avail["phrase"])
            short = sum(max(0, n - avail.get(r, 0)) for r, n in plan.items())
            plan["lemma"] += short  # a route with fewer tokens than planned gives its share to the lemma route
            for r, n in plan.items():
                ex = new_examples.get(r, [])
                for w, g, ctx, m in rnd.sample(ex, min(n, len(ex))):
                    f.write(f"{r}\t{w}\t{g}\t{m}\t{ctx}\t\n")
        old_ex = []
        for kind, ws, _ in seqs:
            for j, w in enumerate(ws):
                if w not in TR_STOP and om[w][1] == "prefix5":
                    old_ex.append((w, om[w][0], " ".join(ws[max(0, j - 4):j + 5])))
        with open(RES / "tr_precision_old_prefix.tsv", "w", encoding="utf-8") as f:
            f.write("word\tgloss\tcontext\tnew_route\tnew_gloss\tok\n")
            for w, g, ctx in rnd.sample(old_ex, min(50, len(old_ex))):
                e, route = new_memo[w]
                f.write(f"{w}\t{g}\t{ctx}\t{route or '-'}\t{e.gloss if e else '-'}\t\n")

    (RES / "tr_matcher_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("old", "new", "new_no_prefix"):
        print(name, {k: (res[name][k]["content_token_coverage"], res[name][k]["content_type_coverage"])
                     for k in ("all", "quran", "hadith")}, res[name].get("by_route_tokens"))
    print("old prefix5 tokens now:", res["old_prefix5_tokens_now"])
    print("top uncovered:", res["top_uncovered_lemmas"])


if __name__ == "__main__":
    main("--sample" in sys.argv)
