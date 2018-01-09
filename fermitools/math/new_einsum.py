import numpy
from more_itertools import distinct_permutations
from toolz.itertoolz import accumulate
from toolz.itertoolz import drop


def einsum(subscripts, *operands):
    target = None
    if '->' in subscripts:
        subscripts, target = subscripts.split('->')
    subscripts = subscripts.split(',')
    shapes = list(map(numpy.shape, operands))
    dims = {}
    for subscript, shape in zip(subscripts, shapes):
        dims.update(zip(subscript, shape))
    subscripts = min(orderings(subscripts), key=expenser(dims))
    print(target)
    print(subscripts)
    print(shapes)
    print(dims)
    print(list(contractions(subscripts)))
    print(expenser(dims)(subscripts))


def orderings(subscripts):
    return distinct_permutations(subscripts)


def expenser(dims):
    cost = flop_counter(dims)

    def _cost(subscripts):
        return sum(map(cost, contractions(subscripts)))

    return _cost


def flop_counter(dims):

    def _cost(subscripts):
        s = set(''.join(subscripts))
        return numpy.product(tuple(map(dims.__getitem__, s)))

    return _cost


def contractions(subscripts):
    return drop(1, accumulate(contract, subscripts))


def contract(s0, s1):

    if len(s0) == 2:
        s0 = ''.join(s0)
        s0 = ''.join(c for c in s0 if s0.count(c) is 1)

    return (s0, s1)


if __name__ == '__main__':
    a = numpy.random.random((3, 4))
    b = numpy.random.random((4, 5))
    c = numpy.random.random((5))
    einsum('ik,kl,l->i', a, b, c)
    einsum('ik,kl,l', a, b, c)
