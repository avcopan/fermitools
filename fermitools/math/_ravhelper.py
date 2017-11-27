import numpy
import operator
import functools
from toolz import functoolz

dict_values = functoolz.compose(tuple, operator.methodcaller('values'))
dict_keys = functoolz.compose(tuple, operator.methodcaller('keys'))


def presorter(src):
    dst = range(len(src))
    return functools.partial(numpy.moveaxis, source=src, destination=dst)


def resorter(dst):
    src = range(-len(dst), 0)
    return functools.partial(numpy.moveaxis, source=src, destination=dst)


def reverse_map(function, iterable):
    return reversed(tuple(map(function, iterable)))
