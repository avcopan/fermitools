import numpy
import scipy
import warnings

from .ot import orth


def eigh_direct_guess(pc, dim, n):
    """generate a set of guess eigenvectors from the preconditioner

    :param pc: the preconditioner, usually some approximation of -(a - w)^-1
    :type pc: typing.Callable
    :param dim: the dimension of the eigenproblem
    :type dim: int
    :param n: the number of guess vectors
    :type n: int

    :rtype: numpy.ndarray
    """
    pc_diag = pc(0.)(numpy.ones(dim))
    srt = numpy.argsort(pc_diag)
    values = numpy.ones(n)
    indices = (srt[:n], range(n))
    return scipy.sparse.coo_matrix((values, indices), shape=(dim, n)).toarray()


def eigh_direct(a, neig, pc, guess, niter=100, nvecs=100, r_thresh=1e-6):
    """direct solver for the lowest eigenvalues of a hermitian matrix

    :param a: a callable linear operator
    :type a: typing.Callable
    :param neig: the number of eigenvalues to solve
    :type neig: int
    :param guess: the initial guess vectors
    :type guess: numpy.ndarray
    :param pc: the preconditioner, usually some approximation of -(a - w)^-1
    :type pc: typing.Callable
    :param niter: the maximum number of iterations
    :type niter: int
    :param r_thresh: the maximum number of vectors to hold in memory
    :type r_thresh: float

    :returns: eigenvalues, eigenvectors, convergence info
    :rtype: (numpy.ndarray, numpy.ndarray, dict)
    """
    rdim0 = 0
    dim, nguess = guess.shape
    v = numpy.zeros((dim, nvecs))
    av = numpy.zeros((dim, nvecs))

    rdim1 = nguess
    rdim = rdim0 + rdim1
    v[:, rdim0:rdim] = guess

    for iteration in range(niter):
        av[:, rdim0:rdim] = a(v[:, rdim0:rdim])

        a_red = numpy.dot(v[:, :rdim].T, av[:, :rdim])

        vals, vecs = numpy.linalg.eigh(a_red)

        w = vals[:neig]
        u = vecs[:, :neig]

        x = numpy.dot(v[:, :rdim], u)
        ax = numpy.dot(av[:, :rdim], u)

        r = ax - x * w
        r_rms = numpy.linalg.norm(r) / numpy.sqrt(numpy.size(r))

        converged = r_rms < r_thresh

        if converged:
            break

        vstep = pc(w)(r)
        v1 = orth(vstep, against=v[:, :rdim], tol=r_thresh)

        _, rdim1 = v1.shape
        rdim0 = rdim
        rdim = rdim0 + rdim1

        if rdim > nvecs:
            rdim0 = 0
            rdim = neig
            v[:, rdim0:rdim] = x
        else:
            v[:, rdim0:rdim] = v1

    info = {'niter': iteration + 1, 'rdim': rdim, 'r_rms': r_rms}

    if not converged:
        warnings.warn("Did not converge! (r_rms: {:7.1e})".format(r_rms))

    return w, x, info
