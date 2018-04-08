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
    dim = 1000
    k = 7
    noise = numpy.random.uniform(low=-4e-3, high=+4e-3, size=(dim, dim))

    s = numpy.eye(dim) + noise
    vals = numpy.ones(dim) + numpy.arange(dim)
    a = numpy.linalg.multi_dot([scipy.linalg.inv(s), numpy.diag(vals), s])

    W = vals[:k]
    vals, vecs = scipy.linalg.eig(a=a)
    select = numpy.argsort(vals)[:k]
    V = vecs[:, select]

    ad = numpy.diag(a)

    a_ = scipy.sparse.linalg.aslinearoperator(a)

    w, v, INFO = direct.eig(
            a=a_, k=k, ad=ad, nguess=2*k, maxdim=8*k, tol=1e-8,
            print_conv=True)

    assert_almost_equal(w, W)
    assert_almost_equal(numpy.abs(v), numpy.abs(V))


def test__eigh():
    dim = 2000
    k = -7
    noise = numpy.random.uniform(low=-4e-3, high=+4e-3, size=(dim, dim))

    a = numpy.eye(dim)
    ad = numpy.ones(dim)

    s = numpy.eye(dim) + noise
    vals = numpy.ones(dim) + numpy.arange(dim)
    b = numpy.linalg.multi_dot([numpy.transpose(s), numpy.diag(vals), s])
    bd = numpy.diag(b)

    vals, vecs = scipy.linalg.eigh(a=a, b=b)
    W = vals[k:]
    V = vecs[:, k:]

    a_ = scipy.sparse.linalg.aslinearoperator(a)
    b_ = scipy.sparse.linalg.aslinearoperator(b)

    w, v, info = direct.eigh(
            a=a_, k=k, ad=ad, b=b_, bd=bd, nguess=2*abs(k), maxdim=8*abs(k),
            maxiter=100, tol=1e-8, print_conv=True)
    print(w)

    assert_almost_equal(w, W)
    assert_almost_equal(numpy.abs(v), numpy.abs(V))


if __name__ == '__main__':
    test__eigh()
