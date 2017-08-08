import sympy.combinatorics.permutations as spp


def signature(p, elems):
    pi = tuple(elems.index(e) for e in p)
    return (-1) ** spp._af_parity(pi)
