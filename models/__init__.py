from .transformer_lm import TransformerLanguageModel
from .dynamic_depth_transformer import SingleLayerTransformer
from .tree_transformer import TreeTransformer, OriginalTreeTransformer, RecurrentTreeTransformer


def get_model(vocab, args):
    if args.arch == 'transformer':
        model = TransformerLanguageModel(vocab=vocab, args=args)
    elif args.arch == 'single_layer_transformer':
        model = SingleLayerTransformer(vocab=vocab, args=args)
    elif args.arch == 'tree_transformer':
        model = TreeTransformer(vocab=vocab, args=args)
    elif args.arch == 'original_tree_transformer':
        model = OriginalTreeTransformer(vocab=vocab, args=args)
    elif args.arch == 'recurrent_tree_transformer':
        model = RecurrentTreeTransformer(vocab=vocab, args=args)
    else:
        raise NotImplementedError
    return model


def reset_parameters(module):
    """Initialize module parameters.

    Use with nn.Module.apply(), e.g., model.apply(reset_parameters)
    """
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def count_param(module):
    """Return parameter count of a module."""
    return sum([param.nelement() for param in module.parameters()])
