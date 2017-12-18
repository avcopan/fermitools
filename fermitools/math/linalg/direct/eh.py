import numpy
import scipy.linalg
import warnings

from ..ot import orth


def eigh(a, neig, ad, guess, niter=100, nvec=100, r_thresh=1e-6):
    """solve for the lowest eigenvalues of a hermitian matrix

    :param a: the matrix, as a callable linear operator
    :type a: typing.Callable
    :param dim: the dimension of `a`
    :type dim: int
    :param neig: the number of eigenvalues to solve
    :type neig: int
    :param guess: initial guess vectors
    :type guess: numpy.ndarray
    :param ad: the diagonal elements of a, or an approximation to them
    :type ad: numpy.ndarray
    :param niter: the maximum number of iterations
    :type niter: int
    :param r_thresh: the maximum number of vectors to hold in memory
    :type r_thresh: float

    :returns: eigenvalues, eigenvectors, convergence info
    :rtype: (numpy.ndarray, numpy.ndarray, dict)
    """
    dim, nguess = guess.shape

    rdim0 = 0
    v = numpy.zeros((dim, nvec))
    av = numpy.zeros((dim, nvec))

    rdim1 = nguess
    rdim = rdim0 + rdim1
    v[:, rdim0:rdim] = guess

    for iteration in range(niter):
        av[:, rdim0:rdim] = a(v[:, rdim0:rdim])

        a_red = numpy.dot(v[:, :rdim].T, av[:, :rdim])

        vals, vecs = scipy.linalg.eigh(a_red)

        w = vals[:neig]
        u = vecs[:, :neig]

        x = numpy.dot(v[:, :rdim], u)
        ax = numpy.dot(av[:, :rdim], u)

        r = ax - x * w
        r_rms = scipy.linalg.norm(r) / numpy.sqrt(numpy.size(r))

        converged = r_rms < r_thresh

        if converged:
            break

        denom = numpy.reshape(w[None, :] - ad[:, None], r.shape)
        vstep = r / denom
        v1 = orth(vstep, against=v[:, :rdim], tol=r_thresh)

        _, rdim1 = v1.shape
        rdim0 = rdim
        rdim = rdim0 + rdim1

        if rdim > nvec:
            rdim0 = 0
            rdim = neig
            v[:, rdim0:rdim] = x
        else:
            v[:, rdim0:rdim] = v1

    info = {'niter': iteration + 1, 'rdim': rdim, 'r_rms': r_rms}

    if not converged:
        warnings.warn("Did not converge! (r_rms: {:7.1e})".format(r_rms))

    return w, x, info


def eighg(a, b, neig, ad, bd, guess, niter=100, nvec=100, r_thresh=1e-6):
    """solve for the lowest generalized eigenvalues of a hermitian matrix

    :param a: the matrix, as a callable linear operator
    :type a: typing.Callable
    :param b: the metric, as a callable linear operator
    :type b: typing.Callable
    :param dim: the dimension of `a`
    :type dim: int
    :param neig: the number of eigenvalues to solve
    :type neig: int
    :param guess: initial guess vectors
    :type guess: numpy.ndarray
    :param ad: the diagonal elements of a, or an approximation to them
    :type ad: numpy.ndarray
    :param bd: the diagonal elements of b, or an approximation to them
    :type bd: numpy.ndarray
    :param niter: the maximum number of iterations
    :type niter: int
    :param r_thresh: the maximum number of vectors to hold in memory
    :type r_thresh: float

    :returns: eigenvalues, eigenvectors, convergence info
    :rtype: (numpy.ndarray, numpy.ndarray, dict)
    """
    dim, nguess = guess.shape

    rdim0 = 0
    v = numpy.zeros((dim, nvec))
    av = numpy.zeros((dim, nvec))
    bv = numpy.zeros((dim, nvec))

    rdim1 = nguess
    rdim = rdim0 + rdim1
    v[:, rdim0:rdim] = guess

    for iteration in range(niter):
        av[:, rdim0:rdim] = a(v[:, rdim0:rdim])
        bv[:, rdim0:rdim] = b(v[:, rdim0:rdim])

        a_red = numpy.dot(v[:, :rdim].T, av[:, :rdim])
        b_red = numpy.dot(v[:, :rdim].T, bv[:, :rdim])

        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[:neig]
        u = vecs[:, :neig]

        x = numpy.dot(v[:, :rdim], u)
        ax = numpy.dot(av[:, :rdim], u)
        bx = numpy.dot(bv[:, :rdim], u)

        r = ax - bx * w
        r_rms = scipy.linalg.norm(r) / numpy.sqrt(numpy.size(r))

        converged = r_rms < r_thresh

        info = {'niter': iteration + 1, 'rdim': rdim, 'r_rms': r_rms}
        print(info)

        if converged:
            break

        denom = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / denom
        v1 = orth(vstep, against=v[:, :rdim], tol=r_thresh)

        _, rdim1 = v1.shape
        rdim0 = rdim
        rdim = rdim0 + rdim1

        if rdim > nvec:
            rdim0 = 0
            rdim = neig
            v[:, rdim0:rdim] = x
        else:
            v[:, rdim0:rdim] = v1

    if not converged:
        warnings.warn("Did not converge! (r_rms: {:7.1e})".format(r_rms))

    return w, x, info
