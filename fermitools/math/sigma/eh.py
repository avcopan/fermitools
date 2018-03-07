import numpy
import itertools
import scipy.linalg
from toolz import itertoolz

import sys
import warnings

import h5py
import tempfile


def eighg(a, b, neig, ad, bd, guess, niter=100, nsvec=100, nvec=100,
          rthresh=1e-5, print_conv=True, highest=False, disk=False):
    slc = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    ns = ()
    vs = ()
    avs = ()
    bvs = ()

    dim, ni = guess.shape
    vi = guess
    i = 0

    for iteration in range(niter):
        _, finame = tempfile.mkstemp(suffix='.hdf5')
        fi = h5py.File(finame, mode='w')

        bounds = numpy.concatenate((numpy.arange(0, ni, nsvec), (ni,)))

        avi = numpy.empty((dim, ni))

        for i, (start, end) in enumerate(itertoolz.sliding_window(2, bounds)):
            vij = numpy.array(vi[:, start:end])
            avi[:, start:end] = a(vij)
            print('subiteration a {:d}, {:d} vectors'.format(i, end-start))

        if disk:
            avi = fi.create_dataset('av', data=avi)

        bvi = numpy.empty((dim, ni))

        for i, (start, end) in enumerate(itertoolz.sliding_window(2, bounds)):
            vij = numpy.array(vi[:, start:end])
            bvi[:, start:end] = b(vij)
            print('subiteration b {:d}, {:d} vectors'.format(i, end-start))

        if disk:
            bvi = fi.create_dataset('bv', data=bvi)

        if disk:
            vi = fi.create_dataset('v', data=vi)

        ns += (ni,)
        vs += (vi,)
        avs += (avi,)
        bvs += (bvi,)

        a_blocks = [[numpy.dot(numpy.transpose(avj), vk)
                     for vk in vs] for avj in avs]
        b_blocks = [[numpy.dot(numpy.transpose(bvj), vk)
                     for vk in vs] for bvj in bvs]

        a_red = numpy.bmat(a_blocks)
        b_red = numpy.bmat(b_blocks)

        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[slc]
        y = vecs[:, slc]
        sections = tuple(itertools.accumulate(ns))
        ys = numpy.split(y, sections, axis=0)

        x = sum(numpy.dot(vj, yj) for vj, yj in zip(vs, ys))
        ax = sum(numpy.dot(avj, yj) for avj, yj in zip(avs, ys))
        bx = sum(numpy.dot(bvj, yj) for bvj, yj in zip(bvs, ys))

        r = ax - bx * w
        rmax = numpy.amax(numpy.abs(r))

        rdim = sum(ns)

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            print(1/w)
            sys.stdout.flush()

        if converged:
            break

        precnd = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / precnd
        vcomp = 0.
        for vj in vs:
            vj = numpy.array(vj)
            vcomp += numpy.linalg.multi_dot([vj, vj.T, vstep])
        vproj = vstep - vcomp

        vi = scipy.linalg.orth(vproj)
        _, ni = vi.shape
        i += 1

        if rdim + ni > nvec:
            ns = ()
            vs = ()
            avs = ()
            bvs = ()

            vi = x
            _, ni = vi.shape
            i = 0

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return w, x, info
