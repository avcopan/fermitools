import numpy


def einsum(*args, **kwargs):
    """Call optimized einsum
    """
    kwargs['optimize'] = 'optimal'
    return numpy.einsum(*args, **kwargs)
