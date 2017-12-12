import fermitools.math.linalg.eig as eig

import numpy
from numpy.testing import assert_almost_equal
import time

numpy.random.seed(0)


def a_sigma(dim):
    ra = numpy.arange(1, dim+1)
    rn = numpy.random.rand(dim)
    cp3 = 1. * numpy.pad(rn, (3, 0), mode='constant')[:-3, None]
    cp2 = 2. * numpy.pad(rn, (2, 0), mode='constant')[:-2, None]
    cp1 = 3. * numpy.pad(rn, (1, 0), mode='constant')[:-1, None]
    cp0 = ra[:, None]
    cm1 = 3. * numpy.pad(rn[:-1], (0, 1), mode='constant')[:, None]
    cm2 = 2. * numpy.pad(rn[:-2], (0, 2), mode='constant')[:, None]
    cm3 = 1. * numpy.pad(rn[:-3], (0, 3), mode='constant')[:, None]

    def _a(r):
        shape = r.shape
        r = r if r.ndim is 2 else numpy.reshape(r, (-1, 1))
        dm3 = numpy.pad(cm3 * r, ((3, 0), (0, 0)), mode='constant')[:-3]
        dm2 = numpy.pad(cm2 * r, ((2, 0), (0, 0)), mode='constant')[:-2]
        dm1 = numpy.pad(cm1 * r, ((1, 0), (0, 0)), mode='constant')[:-1]
        dp0 = r * cp0
        dp1 = numpy.pad(cp1 * r, ((0, 1), (0, 0)), mode='constant')[1:]
        dp2 = numpy.pad(cp2 * r, ((0, 2), (0, 0)), mode='constant')[2:]
        dp3 = numpy.pad(cp3 * r, ((0, 3), (0, 0)), mode='constant')[3:]
        ar = dm3 + dm2 + dm1 + dp0 + dp1 + dp2 + dp3
        return numpy.reshape(ar, shape)

    return _a


def pc_(w):

    d = len(w)
    a_diag = numpy.arange(1, d+1)

    def _pc(r):
        return r / (w - a_diag)

    return _pc


def test__eigh_direct():
    dim = 2000
    neig = 4
    nguess = 4
    r_thresh = 1e-11
    a_ = a_sigma(dim)

    a = a_(numpy.eye(dim))
    vals, vecs = numpy.linalg.eigh(a)
    W = vals[:neig]
    U = vecs[:, :neig]

    w, u, info = eig.eigh_direct(
            a=a_, neig=neig, guess=U, pc=pc_, niter=100, nvecs=100,
            r_thresh=r_thresh)
    assert_almost_equal(w, W, decimal=10)
    assert_almost_equal(numpy.abs(u), numpy.abs(U), decimal=10)
    assert info['niter'] == 1
    assert info['rdim'] == 4

    t0 = time.time()
    guess = numpy.eye(dim, nguess)
    w, u, info = eig.eigh_direct(
            a=a_, neig=neig, guess=guess, pc=pc_, niter=1000, nvecs=20,
            r_thresh=r_thresh)
    dt = time.time() - t0
    assert_almost_equal(w, W, decimal=10)
    assert_almost_equal(numpy.abs(u), numpy.abs(U), decimal=10)
    assert info['niter'] < 14
    assert info['rdim'] < 18
    assert dt < 0.3
