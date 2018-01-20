import numpy
from functools import reduce

from ..trans import transform as _transform


def transform_onebody(a, transformers):
    ndim = numpy.ndim(a)
    c1, c2 = transformers
    brakets = ((ndim-2, ndim-1),)
    return transform(a, transformers=(c1, c2), brakets=brakets)


def transform_twobody(a, transformers, antisymmetrize=True):
    ndim = numpy.ndim(a)
    x1, x2, x3, x4 = ndim-4, ndim-3, ndim-2, ndim-1
    c1, c2, c3, c4 = transformers
    brakets = ((x1, x3), (x2, x4))
    t = transform(a, transformers=(c1, c2, c3, c4), brakets=brakets)

    same_bra = all(numpy.array_equal(x, y) for x, y in zip(c1, c2))
    same_ket = all(numpy.array_equal(x, y) for x, y in zip(c3, c4))

    if not antisymmetrize:
        u = 0.
    elif same_bra:
        u = numpy.swapaxes(t, x1, x2)
    elif same_ket:
        u = numpy.swapaxes(t, x3, x4)
    else:
        u = transform(a, transformers=(c1, c2, c4, c3), brakets=brakets)
        u = numpy.swapaxes(u, x3, x4)

    return t - u


def transform(a, transformers, brakets):
    t1 = transform_spatial(a, transformers)

    ndim = numpy.ndim(a)
    ntrans = len(transformers)
    x0 = ndim - ntrans
    na = {dx+x0: a.shape[1] for dx, (a, _) in enumerate(transformers)}
    t2 = reduce(spin_integrator(na), brakets, t1)
    return t2


def transform_spatial(a, transformers):
    ts = tuple(map(numpy.ascontiguousarray, map(numpy.hstack, transformers)))
    return _transform(a, ts)


def spin_integrator(na):

    def _integrate(a, braket):
        bra, ket = braket
        ndim = numpy.ndim(a)
        ix1 = [slice(None)] * ndim
        ix2 = [slice(None)] * ndim
        ix1[bra], ix1[ket] = slice(None, na[bra]), slice(na[ket], None)
        ix2[bra], ix2[ket] = slice(na[bra], None), slice(None, na[ket])
        a[ix1] = a[ix2] = 0.
        return a

    return _integrate
