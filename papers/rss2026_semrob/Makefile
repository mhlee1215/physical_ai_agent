MAIN = main

.PHONY: all clean fonts

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib IEEEtran.cls
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

fonts: $(MAIN).pdf
	pdffonts $(MAIN).pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
