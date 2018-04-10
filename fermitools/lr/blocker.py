import numpy
from toolz import functoolz
from .linmap import zero, block_diag, bmat
from ..math import raveler, unraveler
from ..math.asym import megaraveler, megaunraveler


def count_excitations(no, nv):
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4
    return n1, n2


def build_ravelers(no, nv):
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})
    r2 = megaraveler({0: ((0, 1), (2, 3))})
    u2 = megaunraveler({0: {(0, 1): no, (2, 3): nv}})
    return r1, r2, u1, u2


def build_block_linmap(no, nv, l11, l12, l21, l22=None):
    r1, r2, u1, u2 = build_ravelers(no, nv)
    n1, n2 = count_excitations(no, nv)
    l11 = functoolz.compose(r1, l11, u1)
    l12 = functoolz.compose(r1, l12, u2)
    l21 = functoolz.compose(r2, l21, u1)
    l22 = functoolz.compose(r2, l22, u2) if l22 is not None else zero
    return bmat([[l11, l12], [l21, l22]], (n1,))


def build_block_diag_linmap(no, nv, l11, l22):
    r1, r2, u1, u2 = build_ravelers(no, nv)
    n1, n2 = count_excitations(no, nv)
    l11 = functoolz.compose(r1, l11, u1)
    l22 = functoolz.compose(r2, l22, u2)
    return block_diag((l11, l22), (n1,))


def build_block_vec(no, nv, v1, v2):
    r1, r2, u1, u2 = build_ravelers(no, nv)
    v1 = r1(v1)
    v2 = r2(v2)
    v = numpy.concatenate((v1, v2), axis=0)
    return v
