import numpy
from operator import mul
from functools import reduce
from functools import partial
from itertools import starmap
from itertools import permutations
from toolz.itertoolz import accumulate
from toolz.itertoolz import drop

from .util import parse_einsum_subscripts
from .util import free_indices
from .util import contraction_indices


# TODO: optimize by sorting only over unique contraction orderings
def einsum_argsort(subscripts, *operands):
    subs, _ = parse_einsum_subscripts(subscripts)
    shps = map(numpy.shape, operands)
    dims = {x: dim for sub, shp in zip(subs, shps) for x, dim in zip(sub, shp)}
    cost = cost_function(subs, dims)
    orderings = permutations(range(len(subs)))
    return min(orderings, key=cost)


def cost_function(subs, dims):

    def _cost(ordering):
        ordered_subs = (subs[i] for i in ordering)
        ctrs = contractions(ordered_subs)
        cost = partial(flops, dims=dims)
        return sum(starmap(cost, ctrs))

    return _cost


def flops(sub1, sub2, dims):
    sub = free_indices(sub1, sub2) + contraction_indices(sub1, sub2)
    return reduce(mul, (dims[x] for x in sub))


def contractions(subs):

    def _pair(sub1, sub2):
        sub1 = free_indices(*sub1) if len(sub1) is 2 else sub1
        return (sub1, sub2)

    return drop(1, accumulate(_pair, subs))
