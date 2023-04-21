# If there are ONLY English characters, xelatex can be replaced with pdflatex.
xelatex main.tex && bibtex main.aux && xelatex main.tex && xelatex main.tex