import numpy
from functools import reduce

from .srt import einsum_argsort
from .util import parse_einsum_subscripts
from .util import free_indices
from .util import contraction_indices


def einsum(subscripts, *operands):
    subs, rsub = parse_einsum_subscripts(subscripts)
    order = einsum_argsort(subscripts, *operands)
    ordered_subs = (subs[i] for i in order)
    ordered_ops = (operands[i] for i in order)
    array, sub = reduce(contract, zip(ordered_ops, ordered_subs))
    rsub = sub if rsub is None else rsub
    axes = tuple(map(sub.index, rsub))
    return numpy.transpose(array, axes)


def contract(arg1, arg2):
    array1, sub1 = arg1
    array2, sub2 = arg2
    csub = contraction_indices(sub1, sub2)
    ax1 = tuple(map(sub1.index, csub))
    ax2 = tuple(map(sub2.index, csub))
    sub = free_indices(sub1, sub2)
    array = numpy.tensordot(array1, array2, axes=(ax1, ax2))
    return array, sub
