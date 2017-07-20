def sort_order(dim, na, nb):
    ai = tuple(range(0, dim))
    bi = tuple(range(dim, 2 * dim))
    return ai[:na] + bi[:nb] + ai[na:] + bi[nb:]
