# chatbot
GPT model used for generating the responses of user medical queries

This repository contains two development branches:
- main - containing the main implementation of the model
- notebook - containing a notebook with the whole implementation of the model (used for initial research and explanation of the model)

The model is implemented based on the tutorial by Andrej Karpaty: *Let's build GPT: from scratch, in code, spelled out.*


## Usage

Once all steps from the *data* repository have been followed run `make pretrain` to pretrain a network.
While training the checkpoints will be saved under `models/checkpoints/ch_{epoch}_{validation_loss}.pt`.
Once the pretraining is done, the model will be saved under `models/pretrained.pt`.
To interact with the model run `make test_pretrained {PATH_TO_MODEL}`.