import numpy
import operator
import itertools
import functools
from toolz import functoolz

dict_values = functoolz.compose(tuple, operator.methodcaller('values'))
dict_keys = functoolz.compose(tuple, operator.methodcaller('keys'))
dict_items = functoolz.compose(tuple, operator.methodcaller('items'))


def presorter(src):
    dst = range(len(src))
    return functools.partial(numpy.moveaxis, source=src, destination=dst)


def resorter(dst):
    src = range(-len(dst), 0)
    return functools.partial(numpy.moveaxis, source=src, destination=dst)


def reverse_map(function, iterable):
    return reversed(tuple(map(function, iterable)))


def reverse_starmap(function, iterable):
    return reversed(tuple(itertools.starmap(function, iterable)))
