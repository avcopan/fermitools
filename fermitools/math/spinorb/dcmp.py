import numpy
from .srt import ov2ab
from .srt import sort


def decompose_onebody(a, na, nb):
    nso, _ = numpy.shape(a)
    assert nso % 2 == 0
    nbf = nso // 2
    t = sort(a, order=ov2ab(nbf, na, nb), axes=(0, 1))
    return t[:nbf, :nbf], t[nbf:, nbf:]
