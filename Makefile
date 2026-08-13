.PHONY: figures paper test experiment clean

figures:
	python3 figures/generate_figures.py

paper: figures
	latexmk -pdf paper.tex

test:
	python3 -m unittest discover -s tests -v

experiment:
	python3 reproduce_experiment.py --jobs 4 --output results/reproduced

clean:
	latexmk -c paper.tex
