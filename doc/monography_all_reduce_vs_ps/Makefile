# Monography: Parameter Server vs All-Reduce in O.N.O.
# Self-contained build. Run from this directory.

PY      := ../../.venv/bin/python
MAIN    := main
SCRIPTS := scripts

.PHONY: all pdf plots tables datasets clean distclean

all: pdf

## Compile the PDF (assumes figures/ and tables/ are populated).
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

## Regenerate every figure from results/processed/results.csv.
plots:
	$(PY) $(SCRIPTS)/make_figures.py

## Regenerate every LaTeX table from results/processed/.
tables:
	$(PY) $(SCRIPTS)/make_tables.py

## Stage / download datasets into datasets/ (MNIST always; EMNIST with EMNIST=1).
datasets:
	$(PY) $(SCRIPTS)/download_datasets.py $(if $(EMNIST),--emnist,)

## Remove LaTeX build intermediates (keeps main.pdf).
clean:
	latexmk -c $(MAIN).tex || true
	rm -f $(MAIN).bbl $(MAIN).run.xml

## Remove everything generated, including the PDF.
distclean: clean
	rm -f $(MAIN).pdf
