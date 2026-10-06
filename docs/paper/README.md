# Isharati technical documentation (LaTeX)

Build (MiKTeX/TeX Live with XeLaTeX; fonts TeX Gyre Termes, Amiri):

    xelatex main && bibtex main && xelatex main && xelatex main   # -> main.pdf

- `main.tex`, `sections/*.tex`: the document; `refs.bib`: verified IEEE references.
- `figures/`: generated figures. Rebuild from the project root:
  - `.venv/Scripts/python docs/paper/tools/make_figures.py docs/paper/data` (plots, skeleton strips)
  - avatar renders: `.venv/Scripts/python docs/paper/tools/serve_fig.py`, then
    `python docs/paper/tools/shoot.py <out.png> "preset=hijab&id=<run id>&t=<s>" ...` (Playwright + Chrome),
    then `.venv/Scripts/python docs/paper/tools/compose.py`
- `data/`: inputs behind the numbers — `inventory.md` (file-backed fact sheet), the two live demo runs
  (`ask_*.json`), sign-length CSVs.
