import sys
import time
import numpy
import fermitools
from toolz import functoolz
from itertools import starmap

import h5py
import tempfile


def energy(labels, coords, charge, spin, basis, angstrom=False, niter=100,
           rthresh=1e-10, diis_start=3, diis_nvec=20, interface=None):
    coords = numpy.divide(coords, 0.52917720859) if angstrom else coords

    # Spaces
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    nbf = interface.integrals.nbf(basis, labels)
    no = na + nb
    nv = 2*nbf - no

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    p_ao = interface.integrals.dipole(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    # Mean-field guess orbitals
    c_guess = interface.hf.unrestricted_orbitals(
            basis, labels, coords, charge, spin)
    t2_guess = numpy.zeros((no, no, nv, nv))

    n = ((na,), (nb,))
    co_guess, cv_guess = zip(*starmap(numpy.hsplit, zip(c_guess, n)))

    print("Running ODC-12 ground-state and linear response computation...\n")

    # Solve ground state
    t = time.time()
    en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
            h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
            t2_guess=t2_guess, niter=niter, rthresh=rthresh,
            diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en_tot = en_elec + en_nuc
    print("\nODC-12 ground state energy: {:20.15f}".format(en_tot))
    print('ODC-12 ground state time: {:8.1f}s'.format(time.time() - t))
    sys.stdout.flush()

    # Evaluate dipole moment as expectation value
    poo = fermitools.math.spinorb.transform_onebody(p_ao, (co, co))
    pvv = fermitools.math.spinorb.transform_onebody(p_ao, (cv, cv))
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    mu = numpy.array([numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
                      for pxoo, pxvv in zip(poo, pvv)])
    print("Electric dipole:")
    print(mu.round(12))

    info['h_ao'] = h_ao
    info['p_ao'] = p_ao
    info['r_ao'] = r_ao
    info['co'] = co
    info['cv'] = cv
    info['t2'] = t2
    info['mu'] = mu
    return en_elec, info


def spectrum(labels, coords, charge, spin, basis, angstrom=False, nroot=1,
             nguess=10, nsvec=10, nvec=100, niter=50, rthresh=1e-7,
             guess_random=False, oo_niter=200, oo_rthresh=1e-10, diis_start=3,
             diis_nvec=20, disk=False, interface=None):
    en_elec, oo_info = energy(
            labels=labels, coords=coords, charge=charge, spin=spin,
            basis=basis, angstrom=angstrom, niter=oo_niter,
            rthresh=oo_rthresh, diis_start=diis_start, diis_nvec=diis_nvec,
            interface=interface)

    info = {k: v for k, v in oo_info.items()
            if k not in ('niter', 'r1max', 'r2max')}

    info['oo_niter'] = oo_info['niter']
    info['oo_r1max'] = oo_info['r1max']
    info['oo_r2max'] = oo_info['r2max']

    # LR inputs
    print("\nTransforming the integrals and computing the density matrices...")
    sys.stdout.flush()

    co = oo_info['co']
    cv = oo_info['cv']
    h_ao = info['h_ao']
    r_ao = info['r_ao']
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hov = fermitools.math.spinorb.transform_onebody(h_ao, (co, cv))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))

    if disk:
        flname = tempfile.mkstemp()[1]
        fl = h5py.File(flname, mode='w')
        gvvvv = fl.create_dataset('gvvvv', data=gvvvv)

    t2 = oo_info['t2']
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    # Compute spectrum by linear response
    no, _, nv, _ = t2.shape
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4

    eye = fermitools.math.sigma.eye
    zero = fermitools.math.sigma.zero

    r1 = fermitools.math.raveler({0: (0, 1)})
    u1 = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2 = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2 = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    ad1u = fermitools.lr.odc12.onebody_hessian_zeroth_order_diagonal(
            foo, fvv)
    ad2u = fermitools.lr.odc12.twobody_hessian_zeroth_order_diagonal(
            foo, fvv, t2)
    a11u, b11u = fermitools.lr.odc12.onebody_hessian(
            foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12u, b12u, a21u, b21u = fermitools.lr.odc12.mixed_hessian(
            fov, gooov, govvv, t2)
    a22u, b22u = fermitools.lr.odc12.twobody_hessian(
            foo, fvv, goooo, govov, gvvvv, t2, disk=disk)
    s11u = fermitools.lr.odc12.onebody_metric(t2)

    ad1 = r1(ad1u)
    ad2 = r2(ad2u)
    a11 = functoolz.compose(r1, a11u, u1)
    b11 = functoolz.compose(r1, b11u, u1)
    a12 = functoolz.compose(r1, a12u, u2)
    b12 = functoolz.compose(r1, b12u, u2)
    a21 = functoolz.compose(r2, a21u, u1)
    b21 = functoolz.compose(r2, b21u, u1)
    a22 = functoolz.compose(r2, a22u, u2)
    b22 = functoolz.compose(r2, b22u, u2)
    s11 = functoolz.compose(r1, s11u, u1)

    sd = numpy.ones(n1+n2)
    ad = numpy.concatenate((ad1, ad2), axis=0)

    s = fermitools.math.sigma.block_diag((s11, eye), (n1,))
    d = zero
    a = fermitools.math.sigma.bmat([[a11, a12], [a21, a22]], (n1,))
    b = fermitools.math.sigma.bmat([[b11, b12], [b21, b22]], (n1,))
    t = time.time()
    solve_spectrum(
            a=a, b=b, s=s, d=d, ad=ad, sd=sd, nroot=nroot, nguess=nguess,
            nsvec=nsvec, nvec=nvec, niter=niter, rthresh=rthresh,
            guess_random=guess_random, disk=disk)
    print('time: {:8.1f}s'.format(time.time() - t))


def solve_spectrum(a, b, s, d, ad, sd, nroot=1, nguess=10, nsvec=10, nvec=100,
                   niter=50, rthresh=1e-7, guess_random=False, disk=False):
    from fermitools.math.sigma import bmat, negative, evec_guess
    from fermitools.math.sigma.eh import eighg as eighg

    e = bmat([[a, b], [b, a]], 2)
    m = bmat([[s, d], [negative(d), negative(s)]], 2)
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))
    dim = len(ed)

    if disk:
        _, finame = tempfile.mkstemp()
        fi = h5py.File(finame, mode='w')
        guess = fi.create_dataset('guess', (dim, nguess*nroot))
    else:
        guess = numpy.empty((dim, nguess*nroot))

    guess[:] = evec_guess(md, nguess*nroot, bd=ed, highest=True)
    eighg(a=m, b=e, neig=nroot, ad=md, bd=ed, guess=guess, rthresh=rthresh,
          nsvec=nsvec, nvec=nvec*nroot, niter=niter, highest=True, disk=disk)


if __name__ == '__main__':
    import interfaces.psi4 as interface
    # import interfaces.pyscf as interface

    LABELS = ('N', 'N')
    COORDS = ((0., 0., 0.), (0., 0., 1.5))

    spectrum(
            labels=LABELS,
            coords=COORDS,
            charge=0,
            spin=0,
            basis='3-21g',
            angstrom=True,
            nroot=20,
            nguess=12,              # number of guess vectors per root
            nsvec=10,               # max number of sigma vectors per sub-iter
            nvec=20,                # max number of subspace vectors per root
            niter=50,
            rthresh=1e-5,           # convergence threshold
            guess_random=False,     # use a random guess?
            oo_niter=200,           # number of iterations for ground state
            oo_rthresh=1e-8,        # convergence threshold for ground state
            diis_start=3,           # when to start DIIS extrapolations
            diis_nvec=20,           # maximum number of DIIS vectors
            disk=True,              #
            interface=interface)    # interface for computing integrals
