import functools


def weighted_loss(loss_func):
    """Decorator to add weight parameter to loss functions.

    Args:
        loss_func: Loss function to be decorated.

    Returns:
        Decorated loss function with weight parameter.
    """
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, *args, **kwargs):
        loss = loss_func(pred, target, *args, **kwargs)
        if weight is not None:
            loss = loss * weight
        return loss
    return wrapper
