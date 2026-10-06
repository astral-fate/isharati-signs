"""One-off: article -> report layout. Sections become chapters (title on its own page), subsections sections."""
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
CHAPTER_LABELS = {"sec:intro", "sec:related", "sec:method", "sec:architecture", "sec:knowledge", "sec:lexicons",
                  "sec:eda", "sec:production", "sec:evaluation", "sec:results", "sec:discussion", "sec:statements"}
for f in sorted((P / "sections").glob("*.tex")):
    s = f.read_text(encoding="utf-8")
    t = s.replace("\\subsection{", "\\SUBSEC{").replace("\\section{", "\\chapter{").replace("\\SUBSEC{", "\\section{")
    t = re.sub(r"Section~\\ref\{(sec:[a-z]+)\}", lambda m: ("Chapter" if m.group(1) in CHAPTER_LABELS else "Section")
               + "~\\ref{" + m.group(1) + "}", t)
    t = re.sub(r"Sections~\\ref\{(sec:[a-z]+)\}", lambda m: ("Chapters" if m.group(1) in CHAPTER_LABELS else "Sections")
               + "~\\ref{" + m.group(1) + "}", t)
    if t != s:
        f.write_text(t, encoding="utf-8")
        print("converted", f.name)
