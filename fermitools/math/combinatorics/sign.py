"""Permutation signature."""
from sympy.combinatorics.permutations import _af_parity


def signature(p, i):
    """signature of a permutation

    :param p: permutation, as a sequence of elements
    :type p: tuple
    :param i: identity permutation
    :type i: tuple

    :rtype: int
    """
    px = tuple(tuple(i).index(e) for e in p)
    return (-1) ** _af_parity(px)
