.PHONY: clean run similarities

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
run: clean data

data: transcripts similarities

transcripts: 
	@echo ">>> Processing transcripts"
	@$(PYTHON_INTERPRETER) src/data/process_transcripts.py
	

similarities: 
	@echo ">>> Calculating segment similarities"
	@$(PYTHON_INTERPRETER) src/data/calc_segment_similarities.py


## Delete all compiled Python files and processed datasets
clean:
	@echo ">>> Cleaning files."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find data/intermediate ! -name '.gitkeep' -type f -exec rm -f {} +

# clean:
# 	rm -rf __pycache__
# 	rm -rf venv