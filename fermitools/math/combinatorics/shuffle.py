"""Generalized riffle-shuffle permutations."""
import more_itertools as mit


# Public
def riffle_shuffles(i, ksizes):
    """cut an iterable into packets and interleave them in all possible ways

    :param i: identity permutation
    :type i: tuple
    :param ksizes: packet sizes, an integer composition of the length of `i`
    :type ksizes: tuple

    :rtype: typing.Iterator
    """
    assert sum(ksizes) == len(i)  # sanity check on packet sizes

    pk_to_px = _indexer(ksizes)  # maps packets -> indices
    px_to_p = _dereferencer(i)   # maps indices -> elements in `i`

    ik = _packets(ksizes)
    pks = mit.distinct_permutations(ik)
    pxs = map(pk_to_px, pks)
    ps = map(px_to_p, pxs)
    return ps


# Private
def _packets(ksizes):
    """packets, encoded as integers, `(0, 0, ..., 1, 1, 1, ..., 2, ..., ...)`

    :param ksizes: packet sizes
    :type ksizes: tuple

    :rtype: tuple
    """
    return sum((sz * (k,) for k, sz in enumerate(ksizes)), ())


def _indexer(ksizes):
    """maps packet permutations to index permutations

    :param ksizes: packet sizes
    :type ksizes: tuple

    :rtype: typing.Callable
    """

    def px(pk):
        return tuple(pk[:n].count(k) + sum(ksizes[:k]) for n, k in
                     enumerate(pk))

    return px


def _dereferencer(i):
    """maps index permutations to permutations

    :param i: identity permutation
    :type i: tuple

    :rtype: typing.Callable
    """

    def p(px):
        return tuple(i[x] for x in px)

    return p
