# chatbot
GPT model used for generating the responses of user medical queries

This repository contains two development branches:
- main - containing the main implementation of the model
- notebook - containing a notebook with the whole implementation of the model (used for initial research and explanation of the model)

The model is implemented based on the tutorial by Andrej Karpaty: *Let's build GPT: from scratch, in code, spelled out.*


## Usage

### Pretraining step
- Once all steps from the *data* repository have been followed run `make pretrain` to pretrain a network.
- While training the checkpoints will be saved under `models/pretrained/checkpoints/ch_{epoch}_{validation_loss}.pt`. Once the pretraining is done, the model will be saved under `models/pretrained/pretrained.pt`.
- To interact with the model run `make test_pretrained path={PATH_TO_MODEL}`.

### Finetuning step
Due to the infeasible training resources required for the pretraining the fine-tuning is done on the free to use the `GPT2LMHeadModel` from [HuggingFace GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) is used for fine-tuning.

*NOTE*: If the model is pretrained well enough it can be substituted in the finetuning step.

- Run `make finetune` to finetune `GPT2LMHeadModel`
- The resulting model will be saved under models/finetuned/finetuned
- Run `make test_finetuned path={PATH_TO_MODEL}` to interact with the finetuned model
