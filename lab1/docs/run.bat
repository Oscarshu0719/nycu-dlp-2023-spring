% If there are ONLY English characters, xelatex.exe can be replaced with pdflatex.exe. %
xelatex.exe main.tex && bibtex.exe main.aux && xelatex.exe main.tex && xelatex.exe main.tex