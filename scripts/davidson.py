import numpy


def eig_direct(a, neig, guess, maxdim, tol):
    dim, nguess = guess.shape
    v = numpy.zeros((dim, dim))
    v[:, :nguess] = guess
    w_old = 1

    for m in range(nguess, maxdim, nguess):
        v, r = numpy.linalg.qr(v)
        v_red = v[:, :(m+1)]
        av_red = numpy.dot(a, v_red)
        a_red = numpy.dot(v_red.T, av_red)
        w_unsrt, s_unsrt = numpy.linalg.eig(a_red)
        idx = numpy.argsort(w_unsrt)
        w = w_unsrt[idx]
        s = s_unsrt[:, idx]
        r = numpy.dot(av_red, s[:, :nguess]) - av_red[:, :nguess] * w[:nguess]
        q = r / (w[:nguess] - numpy.diag(a[:nguess, :nguess]))
        v[:, m+1:m+nguess+1] = q
        norm = numpy.linalg.norm(w[:neig] - w_old)
        w_old = w[:neig]
        if norm < tol:
            break

    vals = w[:neig]
    vecs = numpy.dot(v_red, s[:, :neig])
    return vals, vecs, m


def main():
    import time
    from numpy.testing import assert_almost_equal
    numpy.random.seed(0)

    dim = 1200

    sparsity = 0.0001
    a = (numpy.diag(numpy.arange(1, dim+1))
         + sparsity * numpy.random.rand(dim, dim))
    a = (a + a.T) / 2

    tol = 1e-12
    maxdim = dim//2
    nguess = 8
    neig = 4
    guess = numpy.eye(dim, nguess)

    t0 = time.time()
    w, u, rdim = eig_direct(
            a=a, neig=neig, guess=guess, maxdim=maxdim, tol=tol)
    dt_davidson = time.time() - t0
    print(w)
    print(dt_davidson)

    t0 = time.time()
    w_numpy, u_numpy = numpy.linalg.eigh(a)
    dt_numpy = time.time() - t0
    w_numpy = w_numpy[:neig]
    u_numpy = u_numpy[:, :neig]
    print(w)
    print(dt_numpy)

    assert(rdim <= 265)
    assert(dt_davidson < 10.)
    assert_almost_equal(w, w_numpy)
    assert_almost_equal(numpy.abs(u), numpy.abs(u_numpy))


if __name__ == '__main__':
    main()
