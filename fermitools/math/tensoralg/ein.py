import numpy
from functools import partial
from functools import reduce

from .srt import einsum_argsort
from .util import parse_einsum_subscripts


def einsum(subscripts, *operands):
    subs, rsub = parse_einsum_subscripts(subscripts)
    order = einsum_argsort(subscripts, *operands)
    ordered_subs = (subs[i] for i in order)
    ordered_ops = (operands[i] for i in order)
    array, sub = reduce(contract, zip(ordered_ops, ordered_subs))
    rsub = ''.join(sorted(sub)) if rsub is None else rsub
    ax = partial(axes, sub=sub, ndim=array.ndim)
    transp = sum(map(ax, rsub), ())
    return numpy.transpose(array, transp)


def contract(arg1, arg2):
    array1, sub1 = arg1
    array2, sub2 = arg2
    sxs = ''.join(x for x in sub1 if x in sub2)
    ax1 = axes(sxs, sub1, array1.ndim)
    ax2 = axes(sxs, sub2, array2.ndim)
    sub = (''.join(x for x in sub1 if x not in sub2) +
           ''.join(x for x in sub2 if x not in sub1))
    array = numpy.tensordot(array1, array2, axes=(ax1, ax2))
    return array, sub


def axes(xs, sub, ndim):
    stt, end = str.split(sub, '#') if '#' in sub else (sub, '')
    nstt = len(stt)
    nend = len(end)
    sxs = tuple(stt.index(x) for x in xs if x in stt)
    ixs = tuple(range(nstt, ndim - nend)) if '#' in xs else ()
    exs = tuple(end.index(x) + ndim - nend for x in xs if x in end)
    return sxs + ixs + exs


if __name__ == '__main__':
    from numpy.testing import assert_almost_equal
    a = numpy.random.random((2, 3))
    b = numpy.random.random((3, 11, 12, 5))
    c = numpy.random.random((5, 6))
    D = numpy.einsum('ik,k...l,lj', a, b, c)
    d = einsum('ik,k#l,lj', a, b, c)
    print(D.shape)
    print(d.shape)
    assert_almost_equal(d, D)
