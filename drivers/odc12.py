import sys
import time
import numpy
import fermitools
from toolz import functoolz
from itertools import starmap


def energy(labels, coords, charge, spin, basis, angstrom=False, niter=100,
           rthresh=1e-10, diis_start=3, diis_nvec=20, interface=None):
    '''
    :param labels: nuclear labels
    :type labels: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray
    :param charge: total charge of the system
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param basis: basis set name
    :type basis: str
    :param angstrom: coordinates are in angstroms?
    :type angstrom: bool
    :param niter: number of iterations
    :type niter: int
    :param rthresh: maximum residual
    :type rthresh: float
    :param diis_start: when to start DIIS extrapolations
    :type diis_start: int
    :param diis_nvec: maximum number of DIIS vectors
    :type diis_start: int
    :param interface: interface for computing integrals and SCF orbitals
    :type interface: module
    '''
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
             nconv=None, nguess=None, maxdim=None, maxiter=100, rthresh=1e-5,
             oo_niter=200, oo_rthresh=1e-10, diis_start=3,
             diis_nvec=20, disk=False, interface=None):
    '''
    :param labels: nuclear labels
    :type labels: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray
    :param charge: total charge of the system
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param basis: basis set name
    :type basis: str
    :param angstrom: coordinates are in angstroms?
    :type angstrom: bool
    :param nroot: number of roots to compute
    :type nroot: int
    :param nconv: number of roots to converge
    :type nconv: int
    :param nguess: number of guess vectors
    :type nguess: int
    :param maxdim: maximum number of expansion vectors
    :type maxdim: int
    :param maxiter: maximum iteration
    :type maxiter: int
    :param rthresh: maximum residual for linear response
    :type rthresh: float
    :param oo_niter: number of iterations for orbital optimization
    :type oo_niter: int
    :param oo_rthresh: maximum residual for orbital optimization
    :type oo_rthresh: float
    :param diis_start: when to start DIIS extrapolations
    :type diis_start: int
    :param diis_nvec: maximum number of DIIS vectors
    :type diis_start: int
    :param disk: keep V^4 arrays on disk?
    :type disk: bool
    :param interface: interface for computing integrals and SCF orbitals
    :type interface: module
    '''
    en_elec, oo_info = energy(
            labels=labels, coords=coords, charge=charge, spin=spin,
            basis=basis, angstrom=angstrom, niter=oo_niter,
            rthresh=oo_rthresh, diis_start=diis_start, diis_nvec=diis_nvec,
            interface=interface)

    # LR inputs
    print("\nTransforming the integrals and computing the density matrices...")
    sys.stdout.flush()
    t = time.time()

    co = oo_info['co']
    cv = oo_info['cv']
    h_ao = oo_info['h_ao']
    # p_ao = oo_info['p_ao']
    r_ao = oo_info['r_ao']

    import fermitools.lr.odc12

    a, b, ad = fermitools.lr.odc12.build_hessian_blocks(h_ao, r_ao, co, cv, t2)
    si, sid = fermitools.lr.odc12.build_metric_inverse_blocks(t2)

    h = functoolz.compose(sinv, fermitools.math.sigma.add(a, b),
                          sinv, fermitools.math.sigma.subtract(a, b))
    hd = ad * ad

    print('Integrals and density matrices time: {:8.1f}s\n'
          .format(time.time() - t))
    sys.stdout.flush()

    t = time.time()
    w2, v, info = fermitools.math.direct.eig_simple(
            a=h, k=nroot, ad=hd, nguess=nguess, maxdim=maxdim,
            maxiter=maxiter, tol=rthresh, print_conv=True, printf=numpy.sqrt)
    print('\nODC-12 linear response total time: {:8.1f}s'
          .format(time.time() - t))
    sys.stdout.flush()

    # t = time.time()
    # hminus = functoolz.compose(sinv, fermitools.math.sigma.subtract(a, b),
    #                            sinv, fermitools.math.sigma.add(a, b))
    # w2, v, info = fermitools.math.direct.eig_simple(
    #         a=hminus, k=nroot, ad=hd, nguess=nguess, maxdim=maxdim,
    #         maxiter=maxiter, tol=rthresh, print_conv=True, printf=numpy.sqrt)
    # print('\nODC-12 linear response total time: {:8.1f}s'
    #       .format(time.time() - t))
    # sys.stdout.flush()

    w = numpy.real(numpy.sqrt(w2))

    return w, info


def polarizability(labels, coords, charge, spin, basis, angstrom=False,
                   nvec=100, niter=50, rthresh=1e-7, oo_niter=200,
                   oo_rthresh=1e-10, diis_start=3, diis_nvec=20, disk=False,
                   interface=None):
    '''
    :param labels: nuclear labels
    :type labels: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray
    :param charge: total charge of the system
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param basis: basis set name
    :type basis: str
    :param angstrom: coordinates are in angstroms?
    :type angstrom: bool
    :param nroot: number of roots to compute
    :type nroot: int
    :param nguess: number of guess vectors per root
    :type nguess: int
    :param nvec: max number of subspace vectors per root
    :type nvec: int
    :param niter: number of iterations
    :type niter: int
    :param rthresh: maximum residual for linear response
    :type rthresh: float
    :param oo_niter: number of iterations for orbital optimization
    :type oo_niter: int
    :param oo_rthresh: maximum residual for orbital optimization
    :type oo_rthresh: float
    :param diis_start: when to start DIIS extrapolations
    :type diis_start: int
    :param diis_nvec: maximum number of DIIS vectors
    :type diis_start: int
    :param disk: keep V^4 arrays on disk?
    :type disk: bool
    :param interface: interface for computing integrals and SCF orbitals
    :type interface: module
    '''
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
    t = time.time()

    co = oo_info['co']
    cv = oo_info['cv']
    h_ao = info['h_ao']
    p_ao = info['p_ao']
    r_ao = info['r_ao']
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hov = fermitools.math.spinorb.transform_onebody(h_ao, (co, cv))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    poo = fermitools.math.spinorb.transform_onebody(p_ao, (co, co))
    pov = fermitools.math.spinorb.transform_onebody(p_ao, (co, cv))
    pvv = fermitools.math.spinorb.transform_onebody(p_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))
    gvvvv = fermitools.math.disk.dataset(gvvvv) if disk else gvvvv

    t2 = oo_info['t2']
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    fpoo = fermitools.oo.odc12.fancy_property(poo, m1oo)
    fpvv = fermitools.oo.odc12.fancy_property(pvv, m1vv)

    # Evaluate dipole polarizability by linear response
    no, _, nv, _ = t2.shape
    n1 = no * nv

    r1 = fermitools.math.raveler({0: (0, 1)})
    u1 = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2 = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2 = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    pg1u = fermitools.lr.odc12.onebody_property_gradient(pov, m1oo, m1vv)
    pg2u = fermitools.lr.odc12.twobody_property_gradient(fpoo, -fpvv, t2)
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

    pg1 = r1(pg1u)
    pg2 = r2(pg2u)
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

    pg = numpy.concatenate((pg1, pg2), axis=0)
    ad = numpy.concatenate((ad1, ad2), axis=0)
    a = fermitools.math.sigma.bmat([[a11, a12], [a21, a22]], (n1,))
    b = fermitools.math.sigma.bmat([[b11, b12], [b21, b22]], (n1,))
    t = time.time()
    r, lr_info = fermitools.lr.solve.static_response(
            a=a, b=b, pg=pg, ad=ad, nvec=nvec, niter=niter, rthresh=rthresh)
    alpha = numpy.dot(r.T, pg)
    print("Electric dipole polarizability tensor:")
    print(alpha.round(12))
    print('time: {:8.1f}s'.format(time.time() - t))

    # Remove the integrals file
    if disk:
        fermitools.math.disk.remove_dataset(gvvvv)

    info.update(lr_info)
    info['lr_r'] = r

    return alpha, info
