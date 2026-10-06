"""Second revision: remove file/notebook references, update Turkish and YouTube-ASL status, new section order."""
from pathlib import Path

P = Path(__file__).resolve().parents[1]
EDITS = {
"main.tex": [
 ("\\input{sections/lexicons}\n", "\\input{sections/lexicons}\n\\input{sections/eda}\n"),
 ("\\input{sections/evaluation}\n", "\\input{sections/evaluation}\n\\input{sections/results}\n"),
],
"sections/results.tex": [
 ("was answered from the hadith\nof Gabriel with all 8 glosses signed.", "was answered from a single authentic hadith\nwith all 8 glosses signed."),
],
"sections/coverage_ar.tex": [
 ("(Table~\\ref{tab:coverage_ar},\n\\code{results/coverage\\_ar.json}) gives", "(Table~\\ref{tab:coverage_ar})\ngives"),
 ("Arabic Qur'an and hadith corpora (\\code{results/coverage\\_ar.json}). Distinct", "Arabic Qur'an and hadith corpora. Distinct"),
],
"sections/introduction.tex": [
 ("Every number in the tables comes from\nthe project's own files (dataset records, lexicon files, result files in \\code{results/}); items that are planned but\nnot yet measured are marked as such.",
  "Every number reported was measured on\nthe system and data described here; items that are planned but not yet measured are marked as such."),
],
"sections/knowledge.tex": [
 ("\\caption{Two runs of the live system (2 October 2026), from \\code{docs/paper/data/ask\\_\\{en,ar\\}.json}.}",
  "\\caption{Two runs of the live system (2 October 2026).}"),
],
"sections/architecture.tex": [
 ("\\caption{The local web application (\\code{http://127.0.0.1:8765}) answering", "\\caption{The web application answering"),
],
"sections/lexicons.tex": [
 ("\\caption{ASL lexicon (\\code{data/asl/lexicon/lexicon\\_all.jsonl}): 3,055 glosses", "\\caption{ASL lexicon: 3,055 glosses"),
 ("\\caption{ArSL lexicon (\\code{data/lexicon\\_v2/lexicon\\_qa.jsonl}): 3,979 glosses", "\\caption{ArSL lexicon: 3,979 glosses"),
 ("we wrote a Colab notebook that streams it to cloud storage one archive at a time, converts each clip to the Isharati\npose format (about 80$\\times$ smaller on a tested clip: 1.3\\,MB to 17\\,KB), builds a manifest and a word index for the\nmissing words, and uploads the result to a Hugging Face dataset with the CC BY 4.0 attribution. The notebook has been\ntested on sample files only; the full download has not been run, so no YouTube-ASL pose is used in this work.",
  "it is converted in the cloud one archive at a time to the Isharati pose format (about 40$\\times$ smaller), with a\nmanifest per archive, and published as a Hugging Face dataset with the CC BY 4.0 attribution. The first of the ten\narchives (39,052 clips from 7,818 videos; 99.7\\% matched to their English sentence) is converted; it is used for the\nsign-mining experiments of Section~\\ref{sec:results}, and no YouTube-ASL sign is in the lexicon yet."),
 ("\\paragraph{Sentence-level ASL data (planned, not yet used).}", "\\paragraph{Sentence-level ASL data.}"),
 ("are intended for two uses: training the back-translation\nrecogniser, and mining single signs",
  "serve two purposes: training the back-translation\nrecogniser, and mining single signs"),
 ("& T\\.{I}D & 774 extracted (669 glosses) of 5,017 &", "& T\\.{I}D & 1,308 extracted (1,374 glosses) of 5,017 &"),
 ("T\\.{I}D (so far) & 774 & 69.1 & 68 & 44 & 95.3 & 23 & 128 \\\\", "T\\.{I}D (so far) & 1,308 & 68.2 & 67.5 & 44 & 94 & 23 & 128 \\\\"),
],
"sections/statements.tex": [
 ("are in the project repository (\\code{D:\\textbackslash\nislam\\textbackslash isharati}; private during the challenge). A notebook that converts the YouTube-ASL keypoints into a\nHugging Face dataset (\\code{youtube-asl-isharati}) is included; it has not been run on the full data yet.",
  "are in the project repository (private during the challenge). The converted YouTube-ASL keypoints are being published\nas a Hugging Face dataset with the original CC BY 4.0 attribution."),
 ("keypoints are CC BY 4.0, so their converted form may be shared with attribution once produced.",
  "keypoints are CC BY 4.0, so their converted form is shared with attribution."),
 ("YouTube-ASL Clip Keypoints \\cite{youtubeaslkp} & CC BY 4.0 & planned: conversion and sharing with attribution \\\\",
  "YouTube-ASL Clip Keypoints \\cite{youtubeaslkp} & CC BY 4.0 & converted, shared with attribution (1 of 10 parts so far) \\\\"),
],
"sections/abstract.tex": [
 ("For Arabic the ArSL lexicon covers 59.6\\% of all word occurrences.",
  "For Arabic the ArSL lexicon covers 59.6\\% of all word occurrences; the Turkish and Urdu lexicons, still incomplete,\ncover 16\\% and 51\\% of content-word occurrences, and their gaps are the religious terms."),
],
}
miss = 0
for name, pairs in EDITS.items():
    f = P / name
    s = f.read_text(encoding="utf-8")
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new, 1)
        else:
            print("NOT FOUND", name, repr(old[:70])); miss += 1
    f.write_text(s, encoding="utf-8")
print("missing", miss)
