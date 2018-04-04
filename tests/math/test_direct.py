import fermitools.math.direct as direct

import numpy
import scipy.linalg
import scipy.sparse

from numpy.testing import assert_almost_equal


def test__solve():
    dim = 2000
    noise = numpy.random.uniform(low=-4e-3, high=+4e-3, size=(dim, dim))

    s = numpy.eye(dim) + noise
    vals = numpy.ones(dim) + numpy.arange(dim)
    a = numpy.linalg.multi_dot([scipy.linalg.inv(s), numpy.diag(vals), s])
    b = numpy.random.random((dim, 2))

    X = scipy.linalg.solve(a, b)

    ad = numpy.diag(a)
    a_ = scipy.sparse.linalg.aslinearoperator(a)

    x, info = direct.solve(a=a_, b=b, ad=ad, print_conv=True, tol=1e-7)
    assert_almost_equal(x, X)


def test__eig():
    dim = 2000
    k = 7
    noise = numpy.random.uniform(low=-4e-3, high=+4e-3, size=(dim, dim))

    s = numpy.eye(dim) + noise
    vals = numpy.ones(dim) + numpy.arange(dim)
    a = numpy.linalg.multi_dot([scipy.linalg.inv(s), numpy.diag(vals), s])

    ad = numpy.diag(a)

    W = vals[:k]

    a_ = scipy.sparse.linalg.aslinearoperator(a)

    w, V, INFO = direct.eig_simple(
            a=a_, k=k, ad=ad, nguess=2*k, maxdim=8*k, tol=1e-8,
            print_conv=True)

    assert_almost_equal(w, W)

    w, v, info = direct.eig(
            a=a_, k=k, ad=ad, nguess=2*k, maxdim=8*k, tol=1e-8,
            print_conv=True)

    assert_almost_equal(w, W)
    assert_almost_equal(numpy.abs(v), numpy.abs(V))
    assert info['niter'] <= INFO['niter'] + 1

    w, v, info = direct.eig_disk(
            a=a_, k=k, ad=ad, blsize=3, nguess=2*k, maxdim=8*k, tol=1e-8,
            print_conv=True)

    assert_almost_equal(w, W)
    assert_almost_equal(numpy.abs(v), numpy.abs(V))
    assert info['niter'] <= INFO['niter'] + 1


if __name__ == '__main__':
    test__solve()
