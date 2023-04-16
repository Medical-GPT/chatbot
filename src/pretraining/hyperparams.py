BATCH_SIZE = 16  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 32  # what is the maximum context length for predictions?
MAX_ITERS = 10000  # how many training iterations to run?
EVAL_INTERVAL = 500  # how often to evaluate the model on train and val sets?
EVAL_ITERS = 200  # the mean of how many losses to use for evaluation?
LEARNING_RATE = 3e-4

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
