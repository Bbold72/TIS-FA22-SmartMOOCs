.PHONY: clean run corpus

#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON_INTERPRETER = "python3"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
run: data breakpoints

data: transcripts corpus

transcripts: 
	@echo ">>> Processing transcripts"
	@$(PYTHON_INTERPRETER) src/data/process_transcripts.py
	

corpus: 
	@echo ">>> Making document corpus and calculating document similarities"
	@$(PYTHON_INTERPRETER) src/data/make_corpus.py


breakpoints: 
	@echo ">>> Calculating breakpoints"
	@$(PYTHON_INTERPRETER) src/models/breakpoints.py


## Delete all compiled Python files and processed datasets
clean:
	@echo ">>> Cleaning files."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find data/intermediate ! -name '.gitkeep' -type f -exec rm -f {} +
