import numpy
from operator import add
from operator import mul
from functools import partial
from functools import reduce
from itertools import chain
from itertools import starmap
from itertools import combinations
from itertools import permutations
from toolz.itertoolz import drop
from toolz.itertoolz import accumulate

from .util import parse_einsum_subscripts


def einsum_argsort(subscripts, *operands):
    shps = tuple(map(numpy.shape, operands))
    subs, _ = parse_einsum_subscripts(subscripts)
    dds = tuple(starmap(dimdict, zip(shps, subs)))
    cost = cost_function(dds)
    orderings = contraction_orderings(len(subs))
    return min(orderings, key=cost) if len(dds) > 1 else (0,)


def contraction_orderings(n):

    def _ijfilter(i, j):

        def __f(k):
            return k != i and k != j

        return __f

    def _link(i, j):
        notij = _ijfilter(i, j)
        return map(partial(add, (i, j)), permutations(filter(notij, range(n))))

    return chain(*starmap(_link, combinations(range(n), r=2)))


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
    ndim = len(shp)
    assert len(sub) == ndim if '#' not in sub else len(sub) <= ndim + 1
    assert str.count(sub, '#') in (0, 1)
    stt, end = str.split(sub, '#') if '#' in sub else (sub, '')
    nstt = len(stt)
    nend = len(end)
    psv = reduce(mul, shp[nstt:ndim-nend], 1)
    dd = {**dict(zip(stt, shp[:nstt])), '#': psv,
          **dict(zip(end, shp[-nend:]))}
    return dd


if __name__ == '__main__':
    print(list(contraction_orderings(4)))
