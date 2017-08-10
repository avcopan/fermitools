# Public
def permuter(p, i):
    """applies a permutation to an iterable

    Translates an iterable according to a permutation, acting as the identity
    on non-permutation elements.

    :param p: permutation
    :type p: tuple
    :param i: identity permutation
    :type i: tuple

    :rtype: typing.Callable
    """

    def permute(xs):
        return tuple(map(_item_permuter(p, i), xs))

    return permute


# Private
def _item_permuter(p, i):
    """applies a permutation to an individual object

    :param p: permutation
    :type p: tuple
    :param i: identity permutation
    :type i: tuple

    :rtype: typing.Callable
    """

    def permute(x):
        return x if x not in i else p[i.index(x)]

    return permute
