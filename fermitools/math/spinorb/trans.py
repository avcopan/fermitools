import numpy
from ..trans import transform as general_transform


def transform(a, transformers, brakets):
    cs = tuple(map(numpy.hstack, transformers))
    for c in cs:
        print(c.shape)
    print(a.shape)
    b = general_transform(a, *cs)
    print(b.shape)
    return b
