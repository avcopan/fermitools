import sys
import numpy
import scipy
import itertools
from .ix import cast


def nslice(n, highest=False):
    return slice(None, n) if not highest else slice(None, -n-1, -1)


def eigenvector_guess(ad, bd, nx, highest=False):
    n = len(ad)
    rows = numpy.argsort(ad/bd)[::-1] if highest else numpy.argsort(ad/bd)

    x = numpy.zeros((n, nx))

    for row, col in zip(rows, range(nx)):
        x[row, col] = 1.

    return x


def orth(x):
    tol = (max(x.shape) * numpy.amax(numpy.abs(x)) * numpy.finfo(float).eps)
    x, s, _ = scipy.linalg.svd(x, full_matrices=False, overwrite_a=True)
    return x[:, s > tol]


def project_out(x, ys):
    for y in ys:
        x -= numpy.linalg.multi_dot([y, numpy.transpose(y), x])
    return x


def tril_bmat(trilbls):
    mat, trilbls = numpy.bmat(trilbls[0]), trilbls[1:]
    for blrow in trilbls:
        row = numpy.bmat(blrow)
        col = numpy.transpose(numpy.zeros_like(numpy.bmat(blrow[:-1])))
        mat = numpy.hstack((mat, col))
        mat = numpy.vstack((mat, row))
    return mat


def symmetrize_trilmat(mat):
    n, _ = numpy.shape(mat)
    smat = numpy.zeros((n, n))
    smat[numpy.tril_indices(n)] = mat[numpy.tril_indices(n)]
    smat.T[numpy.tril_indices(n, -1)] = mat[numpy.tril_indices(n, -1)]
    return smat


def eigh(a, k, ad, b=None, bd=None, nconv=None, x0=None, nx0=None,
         highest=False, maxvecs=None, maxiter=100, tol=1e-5, print_conv=False):
    b = b if b is not None else (lambda x: x)
    bd = bd if bd is not None else numpy.ones_like(ad)
    nconv = nconv if nconv is not None else k
    assert 0 < nconv <= k
    x0 = x0 if x0 is not None else eigenvector_guess(ad=ad, bd=bd, nx=nx0,
                                                     highest=highest)
    _, nx0 = numpy.shape(x0)
    maxvecs = maxvecs if maxvecs is not None else 50 * k

    kslc = nslice(k, highest=highest)

    xs = ()
    axs = ()
    bxs = ()

    ar_trilbls = ()
    br_trilbls = ()

    xi = x0

    for iteration in range(maxiter):
        xi = orth(project_out(xi, xs))
        axi = a(xi)
        bxi = b(xi)
        xs += (xi,)
        axs += (axi,)
        bxs += (bxi,)
        ar_trilbls += (tuple(numpy.dot(numpy.transpose(axi), xj)
                             for xj in xs),)
        br_trilbls += (tuple(numpy.dot(numpy.transpose(bxi), xj)
                             for xj in xs),)
        ar = symmetrize_trilmat(tril_bmat(ar_trilbls))
        br = symmetrize_trilmat(tril_bmat(br_trilbls))
        vals, vecs = scipy.linalg.eigh(a=ar, b=br)
        w = vals[kslc]
        vr = vecs[:, kslc]

        splits = tuple(itertools.accumulate(numpy.shape(xj)[1] for xj in xs))
        vrs = numpy.split(vr, splits, axis=0)
        v = sum(numpy.dot(xj, vrj) for xj, vrj in zip(xs, vrs))
        av = sum(numpy.dot(axj, vrj) for axj, vrj in zip(axs, vrs))
        bv = sum(numpy.dot(bxj, vrj) for bxj, vrj in zip(bxs, vrs))

        r = av - bv * w
        xi = -r / (cast(ad, 0, 2) - cast(bd, 0, 2) * cast(w, 1, 2))

        rmaxv = numpy.amax(numpy.abs(r), axis=0)
        rmax = max(rmaxv[:nconv])
        converged = rmax < tol
        rdim = len(vals)
        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        if print_conv:
            print(info)
            for j, (wj, rmaxj) in enumerate(zip(w, rmaxv)):
                print('  eigenvalue {:d}: {:13.10f}, rmax={:3.1e}'
                      .format(j, wj, rmaxj))
            sys.stdout.flush()

        if converged:
            break

        if rdim + k > maxvecs:
            xs = ()
            axs = ()
            bxs = ()

            ar_trilbls = ()
            br_trilbls = ()
            xi = v
