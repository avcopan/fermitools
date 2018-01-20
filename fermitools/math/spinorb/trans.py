import numpy
from ..trans import transform as general_transform


def transform(a, transformers, brakets):
    cs = tuple(map(numpy.hstack, transformers))
    ta = general_transform(a, transformers=cs)

    ndim = numpy.ndim(ta)
    adims = tuple(ac.shape[1] for ac, bc in transformers)
    aslcs = tuple(slice(None, adim) for adim in adims)
    bslcs = tuple(slice(adim, None) for adim in adims)

    for (bra, ket) in brakets:
        ix1 = [slice(None)] * ndim
        ix2 = [slice(None)] * ndim
        ix1[bra], ix1[ket] = aslcs[bra], bslcs[ket]
        ix2[bra], ix2[ket] = bslcs[bra], aslcs[ket]
        ta[ix1] = ta[ix2] = 0.
    return ta
