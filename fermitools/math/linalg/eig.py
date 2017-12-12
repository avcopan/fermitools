import numpy
import warnings

from .ot import orth


def eigh_direct(a, neig, guess, pc, niter=100, nvecs=100, r_thresh=1e-6):
    """direct solver for the lowest eigenvalues of a hermitian matrix

    :param a:
    :type a: typing.Callable
    :param neig:
    :type neig: int
    :param guess:
    :type guess: numpy.ndarray
    :param pc:
    :type pc: typing.Callable
    :param niter:
    :type niter: int
    :param r_thresh:
    :type r_thresh: float
    :param w_thresh:
    :type w_thresh: float
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
