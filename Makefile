export USE_TF=0
export HYDRA_FULL_ERROR=1

EXP_NAME=new_exp
PYTHON=/Users/dmitrii/miniconda3/bin/python

train-base:
	$(PYTHON) train_diffusion.py global_setings.exp_name=$(EXP_NAME) training_pipeline=base

train-fast-dllm:
	$(PYTHON) train_diffusion.py global_setings.exp_name=$(EXP_NAME) training_pipeline=fast_dllm

train-core:
	$(PYTHON) train_diffusion.py global_setings.exp_name=$(EXP_NAME) training_pipeline=core