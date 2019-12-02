from .transformer_lm import TransformerLanguageModel
from .tree_transformer import TreeTransformer


def get_model(vocab, args):
    if args.arch == 'transformer':
        model = TransformerLanguageModel(vocab=vocab, args=args)
    elif args.arch == 'tree_transformer':
        model = TreeTransformer(vocab=vocab, args=args)
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
