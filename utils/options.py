import argparse


def train_argparser():
    parser = argparse.ArgumentParser(description="Training language models.")

    # Data
    parser.add_argument('--data', type=str,
                        help="Directory containing the data.")
    parser.add_argument('--train', type=str, default='train.txt',
                        help="The training data filename under data directory.")
    parser.add_argument('--valid', type=str, default='valid.txt',
                        help="The validation data filename under data directory.")
    parser.add_argument('--vocab', type=str, default='vocab.txt',
                        help="The vocab filename under data directory.")
    parser.add_argument('--vocab-min-count', type=int, default=0,
                        help="The minimum count for a word to be include in vocab.")
    parser.add_argument('--vocab-max-size', type=int,
                        help="The max vocab size (excluding special tokens).")
    parser.add_argument('--delimiter', type=str,
                        help="Delimiter for parsing data files.")
    parser.add_argument('--checkpoint', type=str,
                        help="The path of checkpoint to continue training. Resume from start of "
                             "next epoch.")

    # Output
    parser.add_argument('--path', type=str,
                        help="Path to store the model, log, cache, etc. The cache will store "
                             "processed data. Subsequent run with same --path will load from "
                             "cache. A directory named with current time will be the workspace.")
    parser.add_argument('--run-name', type=str,
                        help="The name for current training run. Default to datetime.")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Use tensorboard logging. The log dir is named tensorboard under "
                             "workspace (see --path).")
    parser.add_argument('--log-norm', action='store_true',
                        help="Log the norm of each parameter at every step. Used for debugging.")
    parser.add_argument('--step-per-save', type=int, default=0,
                        help="Step between each checkpoint.")
    parser.add_argument('--epoch-per-save', type=int, default=1,
                        help="Epoch between each validation and checkpoint.")
    parser.add_argument('--keep-step', type=int,
                        help="The number of last step to keep checkpoints."
                             "Value of 1 means keep checkpoints from the last step.")
    parser.add_argument('--keep-epoch', type=int,
                        help="The number of last epoch to keep checkpoints."
                             "Value of 1 means keep checkpoints from the last epoch.")
    parser.add_argument('--step-per-valid', type=int, default=0,
                        help="Number of step between each validation.")
    parser.add_argument('--epoch-per-valid', type=int, default=1,
                        help="Number of epoch between each validation.")
    parser.add_argument('--max-step', type=int,
                        help="Number of training step to stop at.")
    parser.add_argument('--max-epoch', type=int,
                        help="Number of epoch to stop at.")

    # Hardware
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for random number generation. "
                             "Must be the same among distributed training process.")
    parser.add_argument('--cuda', action='store_true',
                        help="Use GPU for training.")
    parser.add_argument('--dist', action='store_true',
                        help="Use distributed multi-GPU training.")
    parser.add_argument('--gpu-per-node', type=int, default=0,
                        help="Number of GPU on each node for distributed training. "
                             "Use all GPU by default, if --dist is enabled.")
    parser.add_argument('--nodes', type=int, default=1,
                        help="Number of nodes for distributed training.")
    parser.add_argument('--node-rank', type=int, default=0,
                        help="Rank among nodes in the distributed training. "
                             "Environment variable RANK takes precedence.")
    parser.add_argument('--port', type=int, default=16688,
                        help="The master port used for distributed training. "
                             "Environment variable MASTER_PORT takes precedence.")
    parser.add_argument('--address', type=str, default='localhost',
                        help="The master address for distributed training. "
                             "Environment variable MASTER_ADDR takes precedence.")
    parser.add_argument('--fp16', action='store_true',
                        help="Use floating point 16 precision for training.")

    # Optimizer
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'nag', 'sgd'],
                        help="The optimizer to use.")
    parser.add_argument('--step-freq', type=int, default=1,
                        help="Batch to accumulate per step.")
    parser.add_argument('--trim-step', action='store_true',
                        help="Trim off batches that doesn't fit in --step-freq.")
    parser.add_argument('--clip-norm', type=float, default=0.,
                        help="Norm for gradient clipping, 0 for no clipping.")
    parser.add_argument('--lr', type=float,
                        help="Learning rate of the optimizer.")
    parser.add_argument('--adam-betas', nargs='+', type=float, default=[0.9, 0.999],
                        help="Betas for Adam optimizer.")
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--momentum', type=float, default=0,
                        help="Momentum of the optimizer (NAG, SGD).")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help="Weight decay of the optimizer (Adam, NAG, SGD).")
    parser.add_argument('--scheduler', type=str, default='inv_sqrt',
                        choices=['cosine', 'inv_sqrt', 'plateau', 'constant'],
                        help="The learning rate scheduler to use.\n"
                             "cosine: Cosine annealing with decayed restart.\n"
                             "inv_sqrt: Inverse square root, used in original transformer.\n"
                             "plateau: Reduce learning rate on plateau.\n"
                             "constant: Constant learning rate.")
    parser.add_argument('--min-lr', type=float, default=1e-9,
                        help="Minimum learning rate.")
    parser.add_argument('--step-per-period', type=int, default=200000,
                        help="Initial number of step per period for cosine schedule")
    parser.add_argument('--period-factor', type=int, default=1,
                        help="The factor to scale period length for cosine schedule.")
    parser.add_argument('--period-decay', type=float, default=1,
                        help="The factor to multiple learning rate for each successive period in "
                             "cosine schedule.")
    parser.add_argument('--warmup-step', type=int, default=0,
                        help='Number of linear warmup step for schedulers.')
    parser.add_argument('--decay-factor', type=float, default=0.5,
                        help='Factor to multiple learning rate for plateau schedule.')
    parser.add_argument('--patience', type=int, default=10000,
                        help="Number of step to wait for plateau schedule.")

    # Data loading
    dataloading_args(parser)

    # Model
    model_args(parser)

    return parser


def test_argparser():
    parser = argparse.ArgumentParser(description="Evaluating language models.")

    # Data
    parser.add_argument('--eval-valid', action='store_true',
                        help="Evaluate the validation data as well. Loaded from cache.")
    parser.add_argument('--data', type=str,
                        help="Directory containing the data where the vocab file is.")
    parser.add_argument('--valid', type=str, default='valid.txt',
                        help="The validation data filename under data directory.")
    parser.add_argument('--test', type=str, default='test.txt',
                        help="The test data filename under data directory.")
    parser.add_argument('--vocab', type=str, default='vocab.txt',
                        help="The vocab filename under data directory.")
    parser.add_argument('--vocab-min-count', type=int, default=0,
                        help="The minimum count for a word to be include in vocab.")
    parser.add_argument('--vocab-max-size', type=int,
                        help="The max vocab size (excluding special tokens).")
    parser.add_argument('--checkpoint', type=str,
                        help="The path of  model checkpoint for evaluate.")

    # Hardware
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for random number generation. "
                             "Must be the same among distributed training process.")
    parser.add_argument('--cuda', action='store_true',
                        help="Use GPU for training.")
    parser.add_argument('--fp16', action='store_true',
                        help="Use floating point 16 precision for training.")

    # Data loading
    dataloading_args(parser)

    # Model
    model_args(parser)

    return parser


def dataloading_args(parser):
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'masked_lm'],
                        help="The task for train for.\n"
                             "lm: Standard left-to-right language model.\n"
                             "masked_lm: BERT-styled masked language model. Evaluation can be "
                             "slow. Recommend to set a low --eval-token.")
    parser.add_argument('--worker', type=int, default=1,
                        help="Number of workers spawn by DataLoader.")

    # Training
    parser.add_argument('--max-token', type=int, default=4096,
                        help="Max number of token per batch. Includes context-size.")
    parser.add_argument('--train-token', type=int, default=4096,
                        help="The max number of training token for each sample. For masked_lm, "
                             "the actual number of training token depends on --proc-prob.")
    parser.add_argument('--context-size', type=int, default=0,
                        help="The context window size for left-to-right language model.")
    parser.add_argument('--context-type', type=str, default='line',
                        choices=['sent', 'merge_sent', 'line', 'merge_line', 'file'],
                        help="The type of context range.\n"
                             "sent: Split line into sentence by the '.' token."
                             "line: Each line is a context range."
                             "file: All lines are merged into a single context range."
                             "merge: Allow context range to be merge while respecting boundary.")
    parser.add_argument('--min-length', type=int, default=0,
                        help="Minimum token length for data to be included in training.")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle the dataset during training.")
    parser.add_argument('--trim-data', action='store_true',
                        help="Trim off data the doesn't produce a complete batch.")

    # Evaluation
    parser.add_argument('--eval-max-token', type=int,
                        help="Max number of token per batch for evaluation.")
    parser.add_argument('--eval-token', type=int,
                        help="The number of training token for each sample for evaluation.")
    parser.add_argument('--eval-context-size', type=int,
                        help='The context window size for language model for evaluation.')
    parser.add_argument('--eval-context-type', type=str,
                        choices=['sent', 'merge_sent', 'line', 'merge_line', 'file'],
                        help="The type of context range for evaluation.")
    parser.add_argument('--eval-min-length', type=int,
                        help="Minimum token length for data to be included for evaluation.")

    # Masked Language Model
    parser.add_argument('--proc-prob', type=float, default=0.15,
                        help="Probability to use masking procedure on a token during training masked_lm")
    parser.add_argument('--mask-prob', type=float, default=0.8,
                        help="Probability for replacing with <mask> token during training masked_lm")
    parser.add_argument('--rand-prob', type=float, default=0.1,
                        help="Probability for replacing with random token during training masked_lm")


def model_args(parser):
    parser.add_argument('--arch', type=str, default='transformer',
                        choices=['transformer',
                                 'single_layer_transformer',
                                 'layer_permute_transformer',
                                 'layer_pool_transformer',
                                 # Old
                                 'tree_transformer',
                                 'original_tree_transformer',
                                 'recurrent_tree_transformer'],
                        help="The architecture to use.")
    parser.add_argument('--adaptive-input', action='store_true',
                        help="Use adaptive input embedding.")
    parser.add_argument('--adaptive-softmax', action='store_true',
                        help="Use adaptive softmax.")
    parser.add_argument('--adaptive-softmax-dropout', type=float, default=0.,
                        help="Dropout rate for adaptive softmax.")
    parser.add_argument('--cutoff', nargs='+', type=int,
                        help="Cutoff for both adaptive input embedding and adaptive softmax."
                             "Overrides --input-cutoff and --softmax-cutoff.")
    parser.add_argument('--input-cutoff', nargs='+', type=int,
                        help="Cutoff for adaptive input embedding.")
    parser.add_argument('--softmax-cutoff', nargs='+', type=int,
                        help="Cutoff for adaptive softmax.")
    parser.add_argument('--factor', type=int,
                        help="Factor of dimension reduction for both adaptive input embedding and adaptive softmax."
                             "Overrides --input-factor and --softmax-factor.")
    parser.add_argument('--input-factor', type=int, default=4,
                        help="Factor of dimension reduction for adaptive input embedding.")
    parser.add_argument('--softmax-factor', type=int, default=4,
                        help="Factor of dimension reduction for adaptive softmax.")
    parser.add_argument('--tied-adaptive', action='store_true',
                        help="Tie weights across adaptive input and adaptive softmax."
                             "Implies --tied-adaptive-proj and --tied-adaptive-embed are both true.")
    parser.add_argument('--tied-adaptive-proj', action='store_true',
                        help="Tie the projection weights across adaptive input and adaptive softmax.")
    parser.add_argument('--tied-adaptive-embed', action='store_true',
                        help="Tie the embedding weights across adaptive input and adaptive softmax.")
    parser.add_argument('--tied-embed', action='store_true',
                        help="Tie the weights of embedding layer and pre-softmax linear layer.")
    parser.add_argument('--embed-dim', type=int, default=512,
                        help="Dimension of embedding.")
    parser.add_argument('--num-head', type=int, default=8,
                        help="Number of head of multi-head attention.")
    parser.add_argument('--head-dim', type=int, default=None,
                        help="Dimension of the head of multi-head attention.")
    parser.add_argument('--inner-dim', type=int, default=2048,
                        help="Inner dimension of position-wise feed forward.")
    parser.add_argument('--num-layer', type=int, default=6,
                        help="Number of network layer.")
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'],
                        help="Activation function for position wise feed forward.")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout for model.")
    parser.add_argument('--attn-dropout', type=float, default=0.,
                        help="Dropout for attention.")
    parser.add_argument('--layer-dropout', type=float, default=0.,
                        help="Dropout for each layer.")

    # Experiments
    # Single layer Transformer (fully weight tied layers)
    parser.add_argument('--eval-num-layer', type=int,
                        help="Number of layers during evaluation.")

    # Layer Permute Transformer, Layer Pool Transformer
    parser.add_argument('--pool-size', nargs='+', type=int,
                        help="A list of number of layers in each pool.")
    # Layer Pool Transformer
    parser.add_argument('--pool-depth', nargs='+', type=int,
                        help="The depth to form from each pool of layers.")

    # Old Experiments
    parser.add_argument('--attn-type', type=str,
                        choices=['recurrent_dot_product'],
                        help="Type of attention for recurrent tree transformer.")
