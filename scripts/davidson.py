import numpy
import fermitools


def eig_direct(a, neig, guess, maxdim, tol):
    dim, nguess = guess.shape
    r_thresh = tol * dim
    v_old = av_old = numpy.zeros((dim, 0))
    v_new = guess
    niter = maxdim // nguess

    for iteration in range(niter):
        v_new = fermitools.math.orthogonalize(v_new, against=v_old)
        av_new = numpy.dot(a, v_new)

        v = numpy.hstack((v_old, v_new))
        av = numpy.hstack((av_old, av_new))
        a_red = numpy.dot(v.T, av)
        w_unsrt, s_unsrt = numpy.linalg.eig(a_red)
        idx = numpy.argsort(w_unsrt)
        w = w_unsrt[idx]
        s = s_unsrt[:, idx]
        r = (numpy.dot(av, s[:, :neig])
             - numpy.dot(v, s[:, :neig]) * w[:neig])
        v_new = r / (w[:neig] - numpy.diag(a[:neig, :neig]))
        v_old = v
        av_old = av
        r_norm = numpy.linalg.norm(r)
        converged = r_norm < r_thresh
        print(("{:-3d} {:7.1e}" + neig * " {:13.9f}")
              .format(iteration, r_norm, *w))
        if converged:
            break

    vals = w[:neig]
    vecs = numpy.dot(v, s[:, :neig])
    return vals, vecs, len(w)


def main():
    import time
    from numpy.testing import assert_almost_equal
    numpy.random.seed(2)

    dim = 1200

    sparsity = 0.0001
    a = (numpy.diag(numpy.arange(1, dim+1))
         + sparsity * numpy.random.rand(dim, dim))
    a = (a + a.T) / 2

    tol = 1e-8
    maxdim = dim//2
    nguess = 8
    neig = 4
    guess = numpy.eye(dim, nguess)

    print('numpy')
    t0 = time.time()
    w_numpy, u_numpy = numpy.linalg.eigh(a)
    dt_numpy = time.time() - t0
    w_numpy = w_numpy[:neig]
    u_numpy = u_numpy[:, :neig]
    print(w_numpy)
    print(dt_numpy)

    print('davidson, perfect guess')
    _, _, rdim = eig_direct(
            a=a, neig=neig, guess=u_numpy, maxdim=maxdim, tol=tol)
    print(rdim)
    assert rdim == neig

    print('davidson, bad guess')
    t0 = time.time()
    w, u, rdim = eig_direct(
            a=a, neig=neig, guess=guess, maxdim=maxdim, tol=tol)
    dt_davidson = time.time() - t0
    print(rdim)
    print(w)
    print(dt_davidson)

    assert_almost_equal(w, w_numpy)
    assert_almost_equal(numpy.abs(u), numpy.abs(u_numpy))
    assert(rdim <= 265)
    assert(dt_davidson < 5.)


if __name__ == '__main__':
    main()
