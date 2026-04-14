REPORT := report

.PHONY: report clean

report: $(REPORT).pdf ## Compile the report to PDF

$(REPORT).pdf: $(REPORT).tex references.bib
	pdflatex $(REPORT)
	bibtex $(REPORT)
	pdflatex $(REPORT)
	pdflatex $(REPORT)

clean: ## Remove build artifacts
	rm -f $(REPORT).pdf $(REPORT).aux $(REPORT).bbl $(REPORT).blg \
	      $(REPORT).log $(REPORT).out $(REPORT).toc
