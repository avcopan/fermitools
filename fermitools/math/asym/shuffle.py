import more_itertools as mit


# Public
def riffle_shuffles(elems, dsizes):
    pds = _deck_permutations(dsizes)
    pis = map(_indexer(dsizes), pds)
    return map(_dereferencer(elems), pis)


# Private
def _deck_permutations(dsizes):
    ds = sum((n * (d,) for d, n in enumerate(dsizes)), ())
    return mit.distinct_permutations(ds)


def _indexer(dsizes):

    def permutation_indices(pd):
        return tuple(pd[:i].count(d) + sum(dsizes[:d]) for i, d
                     in enumerate(pd))

    return permutation_indices


def _dereferencer(elems):

    def permutation(pi):
        return tuple(elems[i] for i in pi)

    return permutation


# Testing
if __name__ == '__main__':
    print(tuple(riffle_shuffles(elems='abcd', dsizes=(2, 2))))
