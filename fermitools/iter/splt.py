import itertools


def split(i, sizes):
    i_ = iter(i)
    for n in sizes:
        yield tuple(itertools.islice(i_, n))
