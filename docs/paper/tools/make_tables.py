"""LaTeX tables for the corpus analysis, from results/eda.json -> sections/gen_eda.tex (run from the project root)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
r = json.loads((ROOT / "results" / "eda.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "paper" / "sections" / "gen_eda.tex"
ESC = str.maketrans({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"})


def n(x):
    return f"{x:,}"


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}\\%"


def word(w, lang):
    w = w.translate(ESC)
    return f"\\ar{{{w}}}" if lang in ("ar", "ur") else w


lines = [r"\begin{table}[ht]", r"\centering\small",
         r"\caption{The four knowledge corpora: size, and the share of content-word occurrences that the sign lexicon of",
         r"each mode can sign. Urdu answers are signed with Indian Sign Language, whose glosses are English words, so its",
         r"coverage is measured on the English corpus of the same verses and hadith.}",
         r"\label{tab:eda_corpora}",
         r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"Mode & Passages & Tokens & Distinct words & Content tokens & Qur'an covered & Hadith covered \\", r"\midrule"]
for key, name, cov in (("en", "English / ASL", "en"), ("ar", "Arabic / ArSL", "ar"),
                       ("tr", "Turkish / T\\.{I}D", "tr"), ("ur", "Urdu / ISL", "isl_on_en")):
    a = r[key]["all"]
    c = r[cov]
    lines.append(f"{name} & {n(r[key]['passages'])} & {n(a['tokens'])} & {n(a['types'])} & {n(a['content_tokens'])} & "
                 f"{pct(c['quran'].get('content_token_coverage'))} & {pct(c['hadith'].get('content_token_coverage'))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

lines += [r"\begin{table}[ht]", r"\centering\small",
          r"\caption{The ten most frequent content words of the Qur'an and of the hadith in each corpus (tokens of two",
          r"letters or fewer, mostly clitic fragments, are not shown). $^\dagger$: no sign in that mode's lexicon.}",
          r"\label{tab:eda_top}", r"\begin{tabularx}{\textwidth}{l l Y}", r"\toprule", r"Corpus & Part & Most frequent content words \\",
          r"\midrule"]
for key, name, cov in (("en", "English", "en"), ("ar", "Arabic", "ar"), ("tr", "Turkish", "tr"), ("ur", "Urdu", None)):
    for part, pname in (("quran", "Qur'an"), ("hadith", "Hadith")):
        unc = {w for w, _ in r[cov]["all"].get("top_uncovered", [])} if cov else set()
        # top_uncovered holds only the 25 most frequent uncovered words; enough for the ten shown
        ws = [(w, k) for w, k in r[key][part]["top_content"] if len(w) > 2][:10]
        cells = ", ".join(word(w, key) + ("$^\\dagger$" if w in unc else "") + f" ({n(k)})" for w, k in ws)
        lines.append(f"{name if part == 'quran' else ''} & {pname} & {cells} \\\\")
    lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

lines += [r"\begin{table}[ht]", r"\centering\small",
          r"\caption{The signs used most often when the Qur'an and hadith are signed: the share of all content-word",
          r"occurrences that each sign covers.}", r"\label{tab:eda_signs}",
          r"\begin{tabularx}{\textwidth}{l Y}", r"\toprule", r"Lexicon & Most used signs (share of content-word occurrences) \\",
          r"\midrule"]
for key, name in (("en", "ASL"), ("ar", "ArSL")):
    at = r[key]["attribution"]
    tot = at["content_tokens"]
    cells = ", ".join(word(g, key) + f" ({100 * k / tot:.1f}\\%)" for g, k in at["top_signs"][:15])
    lines.append(f"{name} & {cells} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
