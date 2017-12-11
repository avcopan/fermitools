import numpy
import fermitools


def eig_direct(a, pc, neig, guess, niter=100, r_thresh=1e-6):
    dim, nguess = guess.shape
    v_old = av_old = numpy.zeros((dim, 0))
    v_new = guess

    for iteration in range(niter):
        v_new = fermitools.math.orthogonalize(v_new, against=v_old, tol=1e-3)
        av_new = a(v_new)

        v = numpy.hstack((v_old, v_new))
        av = numpy.hstack((av_old, av_new))
        a_red = numpy.dot(v.T, av)
        w_unsrt, s_unsrt = numpy.linalg.eig(a_red)
        idx = numpy.argsort(w_unsrt)
        w = w_unsrt[idx]
        s = s_unsrt[:, idx]
        r = (numpy.dot(av, s[:, :neig])
             - numpy.dot(v, s[:, :neig]) * w[:neig])
        v_new = pc(w[:neig])(r)
        v_old = v
        av_old = av
        r_rms = numpy.linalg.norm(r) / numpy.sqrt(numpy.size(r))
        converged = r_rms < r_thresh
        print(("{:-3d} {:7.1e}" + neig * " {:13.9f}")
              .format(iteration, r_rms, *w))
        if converged:
            break

    vals = w[:neig]
    vecs = numpy.dot(v, s[:, :neig])
    return vals, vecs, len(w)


def main():
    import scipy.sparse.linalg
    import time
    from numpy.testing import assert_almost_equal
    numpy.random.seed(0)

    dim = 1200

    sparsity = 0.0001
    a = (numpy.diag(numpy.arange(1, dim+1))
         + sparsity * numpy.random.rand(dim, dim))
    a = (a + a.T) / 2
    a_ = scipy.sparse.linalg.aslinearoperator(a)

    def pc_(w):

        d = len(w)

        def _pc(r):
            return r / (w - numpy.diag(a[:d, :d]))

        return _pc

    r_thresh = 1e-7
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
            a=a_, pc=pc_, neig=neig, guess=u_numpy, r_thresh=r_thresh)
    print(rdim)
    assert rdim == neig

    print('davidson, bad guess')
    t0 = time.time()
    w, u, rdim = eig_direct(
            a=a_, pc=pc_, neig=neig, guess=guess, r_thresh=r_thresh)
    dt_davidson = time.time() - t0
    print(rdim)
    print(w)
    print(dt_davidson)

    assert_almost_equal(w, w_numpy)
    assert_almost_equal(numpy.abs(u), numpy.abs(u_numpy))
    assert(rdim <= 200)
    assert(dt_davidson < 3.)


if __name__ == '__main__':
    main()
