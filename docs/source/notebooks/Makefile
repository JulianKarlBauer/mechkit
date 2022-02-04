# Convert scripts to notebooks

# Find all Python scripts
PYFILES := $(wildcard *.py)
# Define targets: For each Python scripts ask for corresponding notebook
NOTEBOOKFILES := $(PYFILES:.py=.ipynb)

# Do this if bare 'make'-command is executed
.PHONY: all

# 'all' requires all notebook files to exist and be up to date
all: $(NOTEBOOKFILES)

# Define a rule which is applied if a notebook file is required
%.ipynb: %.py
	# p2j -o $<
	jupytext --to notebook $<
	jupyter nbconvert --execute --to notebook --inplace $@


