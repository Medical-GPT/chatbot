#!make
#----------------------------------------
# Settings
#----------------------------------------
.DEFAULT_GOAL := help

#--------------------------------------------------
# Targets
#--------------------------------------------------
install: clean ## Creates venv and installs the package
	@echo "==> Creating virtual environment..."
	@python3 -m venv venv/
	@venv/bin/pip install black
	@echo "    [✓]"
	@echo

	@echo "==> Installing utility and dependencies..."
	@venv/bin/pip install --upgrade pip
	@venv/bin/pip install -r requirements.txt
	@echo "    [✓]"
	@echo

uninstall: clean ## Uninstalls utility, deletes data, and destroys venv
	@echo "==> Removing data..."
	@venv/bin/python src/delete.py
	@echo "    [✓]"
	@echo
	@echo "==> Uninstalling utility and dependencies..."
	@rm -rf venv/
	@echo "    [✓]"
	@echo

clean: ## Cleans up temporary files
	@echo "==> Cleaning up..."
	@find . -name "*.pyc" -exec rm -f {} \;
	@echo "    [✓]"
	@echo

pretrain: ## Pre-trains the model
	@echo "==> Pre-training model..."
	@venv/bin/python -m src.pretraining.pretrain
	@echo "    [✓]"
	@echo

test_pretrained: ## Interact with model (pass model={model.pt} to specify model)
	@echo "==> Starting model..."
	@venv/bin/python -m src.pretraining.test $(model)
	@echo "    [✓]"
	@echo

finetune: ## Finetunes the model (pass model={model} dataset={medical | empathic | path_to_dataset} output={model_name} to specify finetune mode and destination. Omit model to start with a clean gpt2 model)
	@echo "==> Finetuning model..."
	@venv/bin/python -m src.finetuning.finetune ${model} $(dataset) $(output)
	@echo "    [✓]"
	@echo

test_finetuned: ## Interact with model (pass model={model.pt} to specify model)
	@echo "==> Starting model..."
	@venv/bin/python -m src.finetuning.test $(model)
	@echo "    [✓]"
	@echo

.PHONY: install uninstall clean help
help: ## Shows available targets
	@fgrep -h "## " $(MAKEFILE_LIST) | fgrep -v fgrep | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-13s\033[0m %s\n", $$1, $$2}'
