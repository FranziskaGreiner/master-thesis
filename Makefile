DISS  = thesis

$(DISS): $(DISS).ps


$(DISS).ps: $(DISS).tex chapters.tex abstract.tex $(DISS).bib append.tex title.tex toc.tex
	-pdflatex $(DISS)
	-bibtex -min-crossrefs=10 $(DISS)
	-makeindex $(DISS)
	-pdflatex $(DISS)
	-pdflatex $(DISS)

clean:
	-rm $(DISS).ps
	-rm $(DISS).pdf
	-rm *~ $(DISS).{log,aux,bbl,blg,dvi,idx,ilg,ind,lof,lot,not,toc}
	-rm *.{log,aux,bbl,blg,dvi,idx,ilg,ind,lof,lot,not,toc}


