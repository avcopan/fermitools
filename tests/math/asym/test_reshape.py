from fermitools.math.asym import reshape

import numpy


def test__compound_indices():
    a = numpy.random.rand(3, 5, 6, 6, 3, 5, 6)
    b0 = reshape.compound_indices(a, {0: (0, 4), 1: (1, 5), 2: (2, 3, 6)})
    b1 = reshape.compound_indices(a, {0: (0, 4), 2: (1, 5), 1: (2, 3, 6)})
    b2 = reshape.compound_indices(a, {1: (0, 4), 0: (1, 5), 2: (2, 3, 6)})
    b3 = reshape.compound_indices(a, {1: (0, 4), 2: (1, 5), 0: (2, 3, 6)})
    b4 = reshape.compound_indices(a, {2: (0, 4), 0: (1, 5), 1: (2, 3, 6)})
    b5 = reshape.compound_indices(a, {2: (0, 4), 1: (1, 5), 0: (2, 3, 6)})
    assert b0.shape == (3, 10, 20)
    assert b1.shape == (3, 20, 10)
    assert b2.shape == (10, 3, 20)
    assert b3.shape == (20, 3, 10)
    assert b4.shape == (10, 20, 3)
    assert b5.shape == (20, 10, 3)
