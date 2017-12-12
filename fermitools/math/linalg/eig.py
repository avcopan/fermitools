import numpy
import warnings
from .ot import orth


def eigh_direct(a, neig, guess, pc, niter=100, r_thresh=1e-6,
                print_conv=False):
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
    :param print_conv:
    :type print_conv: bool
    """
    dim, _ = guess.shape
    v0 = av0 = numpy.zeros((dim, 0))
    v1 = guess

    for iteration in range(niter):
        v1 = orth(v1, against=v0, tol=r_thresh)
        av1 = a(v1)

        v = numpy.concatenate((v0, v1), axis=1)
        av = numpy.concatenate((av0, av1), axis=1)

        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)

        vals, vecs = numpy.linalg.eigh(a_red)
        w = vals[:neig]
        u = vecs[:, :neig]

        ax = numpy.dot(av, u)
        x = numpy.dot(v, u)

        r = ax - x * w
        v1 = pc(w)(r)

        v0, av0 = v, av

        r_rms = numpy.linalg.norm(r) / numpy.sqrt(numpy.size(r))
        converged = r_rms < r_thresh

        if converged:
            break

    info = {'niter': iteration + 1, 'rdim': rdim, 'r_rms': r_rms}

    if not converged:
        warnings.warn("Did not converge! (r_rms: {:7.1e})".format(r_rms))

    if print_conv:
        print("w = ", w)
        print("({:-3d} iterations, {:-3d} vectors, r_rms: {:7.1e})"
              .format(info['niter'], info['rdim'], info['r_rms']))

    return w, x, info
