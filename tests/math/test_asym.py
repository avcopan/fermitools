import fermitools.math.asym as asym

import numpy
import pytest
from numpy.testing import assert_almost_equal


def assert_not_equal(*args, **kwargs):
    with pytest.raises(AssertionError):
        assert_almost_equal(*args, **kwargs)


def test__antisymmetrizer_product():
    a = numpy.random.rand(10, 10, 10, 10, 10, 10)

    b = asym.antisymmetrizer_product('0/1|2/3|4/5')(a)

    c = asym.antisymmetrizer((0, 1))(
            asym.antisymmetrizer((2, 3))(
                asym.antisymmetrizer((4, 5))(a)))

    assert_almost_equal(b, c, decimal=10)


def test__antisymmetrizer():
    a = numpy.random.rand(10, 10, 10, 10, 10)

    b = asym.antisymmetrizer((1, 3))(asym.antisymmetrizer((0, 2, 4))(a))

    assert_almost_equal(b, -numpy.transpose(b, (0, 3, 2, 1, 4)), decimal=10)
    assert_almost_equal(b, -numpy.transpose(b, (2, 1, 0, 3, 4)), decimal=10)
    assert_not_equal(b, -numpy.transpose(b, (1, 0, 2, 3, 4)), decimal=10)
    assert_not_equal(b, -numpy.transpose(b, (0, 2, 1, 3, 4)), decimal=10)
    assert_not_equal(b, -numpy.transpose(b, (0, 1, 3, 2, 4)), decimal=10)
    assert_not_equal(b, -numpy.transpose(b, (0, 1, 2, 4, 3)), decimal=10)

    c = asym.antisymmetrizer(((0, 2, 4), (1, 3)))(b)

    d = asym.antisymmetrizer((0, 1, 2, 3, 4))(a)

    assert_almost_equal(d, c, decimal=10)
    assert_not_equal(d, 0., decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (0, 3, 2, 1, 4)), decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (2, 1, 0, 3, 4)), decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (1, 0, 2, 3, 4)), decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (0, 2, 1, 3, 4)), decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (0, 1, 3, 2, 4)), decimal=10)
    assert_almost_equal(d, -numpy.transpose(d, (0, 1, 2, 4, 3)), decimal=10)


def test___process_bartlett_string():
    assert (asym._process_bartlett_string('0,1/2|3,4')
            == (((0, 1), (2,)), ((3, 4),),))