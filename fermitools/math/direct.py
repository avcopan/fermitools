import sys
import numpy
import scipy.linalg
import itertools
import warnings
from .ix import cast


def standard_basis_vectors(dim, axes):
    xs = list(axes)

    n = int(dim)
    m = len(xs)
    u = numpy.zeros((n, m))

    for i, j in zip(xs, range(m)):
        u[i, j] = 1.

    return u


def project_out(x, ys):
    for y in ys:
        x -= numpy.linalg.multi_dot([y, numpy.transpose(y), x])
    return x


def orth(x, compress=True):
    tol = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
    x, s, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
    return x if not compress else x[:, s > tol]


def split_rdot(xs, y):
    splits = tuple(itertools.accumulate(numpy.shape(x)[-1] for x in xs))[:-1]
    ys = numpy.split(y, splits, axis=-2)
    return sum(numpy.dot(x, y) for x, y in zip(xs, ys))


def eig0(a, k, ad, nconv=None, nguess=None, maxdim=None, maxiter=100,
         tol=1e-5, print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)

    dim = len(ad)
    guess = standard_basis_vectors(dim=dim, axes=range(nguess))

    ax = x = numpy.zeros((dim, 0))

    converged = False

    xi = guess

    for iteration in range(maxiter):
        xi = orth(project_out(xi, (x,)), compress=True)
        axi = a(xi)

        x = numpy.concatenate((x, xi), axis=1)
        ax = numpy.concatenate((ax, axi), axis=1)

        ap = numpy.dot(numpy.transpose(x), ax)

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = numpy.real(vals[select])
        vp = numpy.real(vecs[:, select])
        vp_im_norm = scipy.linalg.norm(numpy.imag(vecs[:, select]))

        v = numpy.dot(x, vp)
        av = numpy.dot(ax, vp)

        r = av - v * cast(w, 1, 2)
        xi = -r / (cast(ad, 0, 2) - cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if vp_im_norm > numpy.finfo(float).eps:
            warnings.warn("Discarding imaginary component.")

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, printf(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxdim:
            ax = x = numpy.zeros((dim, 0))
            xi = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info


def eig1(a, k, ad, nconv=None, nguess=None, maxdim=None, maxiter=100, tol=1e-5,
         print_conv=False, printf=None):
    nconv = nconv if nconv is not None else k
    nguess = nguess if nguess is not None else 2*k
    maxdim = maxdim if maxdim is not None else nguess + 40*k
    printf = printf if printf is not None else (lambda x: x)

    dim = len(ad)
    guess = standard_basis_vectors(dim=dim, axes=range(nguess))

    xs = ()
    axs = ()

    ap_bls = {}

    converged = False

    xi = guess

    for iteration in range(maxiter):
        xi = orth(project_out(xi, xs), compress=True)
        axi = a(xi)

        i = len(xs)

        ap_bls[i, i] = numpy.dot(numpy.transpose(xi), axi)
        for j, (xj, axj) in enumerate(zip(xs, axs)):
            ap_bls[i, j] = numpy.dot(numpy.transpose(xi), axj)
            ap_bls[j, i] = numpy.dot(numpy.transpose(xj), axi)

        ap = numpy.bmat(
                [[ap_bls[j, k] for k in range(i+1)] for j in range(i+1)])

        xs += (xi,)
        axs += (axi,)

        vals, vecs = scipy.linalg.eig(a=ap)

        select = numpy.argsort(vals)[:k]
        w = numpy.real(vals[select])
        vp = numpy.real(vecs[:, select])
        vp_im_norm = scipy.linalg.norm(numpy.imag(vecs[:, select]))

        v = split_rdot(xs, vp)
        av = split_rdot(axs, vp)

        r = av - v * cast(w, 1, 2)
        xi = -r / (cast(ad, 0, 2) - cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if vp_im_norm > numpy.finfo(float).eps:
            warnings.warn("Discarding imaginary component.")

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, printf(wj), rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxdim:
            xs = ()
            axs = ()
            ap_bls = {}
            xi = v

    if not converged:
        warnings.warn("Did not converge! rmax={:3.1e}".format(rmax))

    return w, v, info
