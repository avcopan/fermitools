import numpy
from operator import mul
from functools import reduce
from itertools import starmap
from itertools import permutations
from toolz.itertoolz import drop
from toolz.itertoolz import accumulate

from .util import parse_einsum_subscripts


def einsum_argsort(subscripts, *operands):
    shps = map(numpy.shape, operands)
    subs, _ = parse_einsum_subscripts(subscripts)
    dds = tuple(starmap(dimdict, zip(shps, subs)))
    cost = cost_function(dds)
    orderings = permutations(range(len(subs)))
    return min(orderings, key=cost)


def cost_function(dds):

    def _cost(ordering):
        ordered_dds = (dds[i] for i in ordering)
        ctrs = contractions(ordered_dds)
        return sum(starmap(flops, ctrs))

    return _cost


def flops(dd1, dd2):
    ret = reduce(mul, dd1.values())
    ret *= reduce(mul, (dd2[x] for x in dd2 if x == '#' or x not in dd1))
    return ret


def contractions(dds):

    def _contract(arg1, arg2):
        dd1, dd2 = (arg1, {'#': 1}) if isinstance(arg1, dict) else arg1
        dd = {**{x: d for x, d in dd1.items() if x not in dd2},
              **{x: d for x, d in dd2.items() if x not in dd1},
              '#': dd1['#'] * dd2['#']}
        return (dd, arg2)

    return drop(1, accumulate(_contract, dds))


def dimdict(shp, sub):
    ssub = str.strip(sub, '#')
    nact = len(ssub)
    act = shp[-nact:] if str.startswith(sub, '#') else shp[:nact]
    psv = shp[:-nact] if str.startswith(sub, '#') else shp[nact:]
    dd = dict(zip(ssub, act))
    dd['#'] = reduce(mul, psv, 1)
    return dd


if __name__ == '__main__':
    a = numpy.random.random((2, 3))
    b = numpy.random.random((3, 11, 5))
    c = numpy.random.random((5, 6))
    D = numpy.einsum('ik,k...l,lj->...ji', a, b, c)
    print(einsum_argsort('ik,k#l,lj->#ji', a, b, c))
