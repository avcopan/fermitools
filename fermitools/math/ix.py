import numpy
from typing import Iterable


def cast(a, ax, ndim=None):
    ax = tuple(ax) if isinstance(ax, Iterable) else (ax,)
    assert numpy.ndim(a) == len(ax)
    ndim = max(ax) + 1 if ndim is None else ndim
    ix = (slice(None) if i in ax else None for i in range(ndim))
    at = numpy.transpose(a, numpy.argsort(ax))
    return at[tuple(ix)]


def diagonal_indices(n, ax, ndim=None):
    ndim = max(ax) + 1 if ndim is None else ndim
    ix = (numpy.arange(n) if i in ax else slice(None) for i in range(ndim))
    return tuple(ix)
