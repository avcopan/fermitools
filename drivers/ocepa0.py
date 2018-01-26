import sys
import time
import numpy
import fermitools
from itertools import starmap


def energy(labels, coords, charge, spin, basis, angstrom=False, niter=100,
           rthresh=1e-10, interface=None):
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

    print("Running OCEPA0 ground-state and linear response computation...\n")

    # Solve ground state
    t = time.time()
    en_elec, co, cv, t2, info = fermitools.oo.ocepa0.solve(
            h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
            t2_guess=t2_guess, niter=niter, rthresh=rthresh, print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en_tot = en_elec + en_nuc
    print("\nOCEPA0 ground state energy: {:20.15f}".format(en_tot))
    print('OCEPA0 ground state time: {:8.1f}s'.format(time.time() - t))
    sys.stdout.flush()

    # Evaluate dipole moment as expectation value
    poo = fermitools.math.spinorb.transform_onebody(p_ao, (co, co))
    pvv = fermitools.math.spinorb.transform_onebody(p_ao, (cv, cv))
    m1oo, m1vv = fermitools.oo.ocepa0.onebody_density(t2)
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
             nguess=10, nvec=100, niter=50, rthresh=1e-7, guess_random=False,
             oo_niter=200, oo_rthresh=1e-10, interface=None):
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
    :param nvec: number of subspace vectors to start with
    :type nvec: int
    :param niter: number of iterations
    :type niter: int
    :param rthresh: maximum residual for linear response
    :type rthresh: float
    :param guess_random: use a random guess?
    :type guess_random: bool
    :param oo_niter: number of iterations for orbital optimization
    :type oo_niter: int
    :param oo_rthresh: maximum residual for orbital optimization
    :type oo_rthresh: float
    :param interface: interface for computing integrals and SCF orbitals
    :type interface: module
    '''
    en_elec, oo_info = energy(
            labels=labels, coords=coords, charge=charge, spin=spin,
            basis=basis, angstrom=angstrom, niter=oo_niter,
            rthresh=oo_rthresh, interface=interface)

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

    t2 = oo_info['t2']
    no, _, nv, _ = t2.shape
    foo = fermitools.oo.ocepa0.fock_xy(hxy=hoo, goxoy=goooo)
    fov = fermitools.oo.ocepa0.fock_xy(hxy=hov, goxoy=gooov)
    fvv = fermitools.oo.ocepa0.fock_xy(hxy=hvv, goxoy=govov)

    # Compute spectrum by linear response
    sd = fermitools.lr.ocepa0.metric_zeroth_order_diagonal(no, nv)
    ad = fermitools.lr.ocepa0.hessian_zeroth_order_diagonal(foo=foo, fvv=fvv)

    s, d = fermitools.lr.ocepa0.metric(t2=t2)
    a, b = fermitools.lr.ocepa0.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)

    print('Integrals and density matrices time: {:8.1f}s\n'
          .format(time.time() - t))
    sys.stdout.flush()

    t = time.time()
    w, x, y, lr_info = fermitools.lr.solve.spectrum(
            a=a, b=b, s=s, d=d, ad=ad, sd=sd, nroot=nroot, nguess=nguess,
            nvec=nvec, niter=niter, rthresh=rthresh,
            guess_random=guess_random)
    print("\nOCEPA0 excitation energies (in a.u.):")
    print(w.reshape(-1, 1))
    print("\nOCEPA0 excitation energies (in eV):")
    print(w.reshape(-1, 1)*27.2114)
    print('\nOCEPA0 linear response total time: {:8.1f}s'
          .format(time.time() - t))
    sys.stdout.flush()

    info.update(lr_info)
    info['lr_x'] = x
    info['lr_y'] = y

    # Copmute the transition dipoles
    pg = fermitools.lr.ocepa0.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    mu_trans = fermitools.lr.transition_dipole(
            s=s, d=d, pg=pg, x=x, y=y)

    print("\nOCEPA0 transition dipoles (a.u.):")
    print(mu_trans.round(12))
    print("\nOCEPA0 norm of transition dipoles (a.u.):")
    print(numpy.sqrt(numpy.diag(numpy.dot(mu_trans, mu_trans.T))
          .reshape(-1, 1)).round(12))
    sys.stdout.flush()

    info['mu_trans'] = mu_trans

    return w, info


def polarizability(labels, coords, charge, spin, basis, angstrom=False,
                   nvec=100, niter=50, rthresh=1e-7, oo_niter=200,
                   oo_rthresh=1e-10, interface=None):
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
    :param nvec: number of subspace vectors to start with
    :type nvec: int
    :param niter: number of iterations
    :type niter: int
    :param rthresh: maximum residual for linear response
    :type rthresh: float
    :param oo_niter: number of iterations for orbital optimization
    :type oo_niter: int
    :param oo_rthresh: maximum residual for orbital optimization
    :type oo_rthresh: float
    :param interface: interface for computing integrals and SCF orbitals
    :type interface: module
    '''
    en_elec, oo_info = energy(
            labels=labels, coords=coords, charge=charge, spin=spin,
            basis=basis, angstrom=angstrom, niter=oo_niter,
            rthresh=oo_rthresh, interface=interface)

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

    t2 = oo_info['t2']
    foo = fermitools.oo.ocepa0.fock_xy(hxy=hoo, goxoy=goooo)
    fov = fermitools.oo.ocepa0.fock_xy(hxy=hov, goxoy=gooov)
    fvv = fermitools.oo.ocepa0.fock_xy(hxy=hvv, goxoy=govov)

    # Evaluate dipole polarizability by linear response
    pg = fermitools.lr.ocepa0.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    a, b = fermitools.lr.ocepa0.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)
    ad = fermitools.lr.ocepa0.hessian_zeroth_order_diagonal(foo=foo, fvv=fvv)
    t = time.time()
    r, lr_info = fermitools.lr.solve.static_response(
            a=a, b=b, pg=pg, ad=ad, nvec=nvec, niter=niter, rthresh=rthresh)
    alpha = numpy.dot(r.T, pg)
    print("Electric dipole polarizability tensor:")
    print(alpha.round(12))
    print('time: {:8.1f}s'.format(time.time() - t))

    info.update(lr_info)
    info['lr_r'] = r

    return alpha, info
