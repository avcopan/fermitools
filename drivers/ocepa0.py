import time
import numpy
import fermitools


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

    # Solve ground state
    t = time.time()
    en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
            na=na, nb=nb, h_ao=h_ao, r_ao=r_ao, c_guess=c_guess,
            t2_guess=t2_guess, niter=oo_niter, r_thresh=oo_rthresh,
            print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en_tot = en_elec + en_nuc
    oo_info = {'en_tot': en_tot, 't2': t2, 'c': c, **info}
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    print('time: {:8.1f}s'.format(time.time() - t))

    # LR inputs
    ac, bc = c
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
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

    t = time.time()
    w, x, y, info = fermitools.lr.solve.spectrum(
            a=a, b=b, s=s, d=d, ad=ad, sd=sd, nroot=nroot, nguess=nguess,
            nvec=nvec, niter=niter, r_thresh=rthresh,
            guess_random=guess_random)
    print("\nExcitation energies:")
    print(w)
    print('time: {:8.1f}s'.format(time.time() - t))

    # Copmute the transition dipoles
    pg = fermitools.lr.ocepa0.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    mu_trans = fermitools.lr.transition_dipole(
            s=s, d=d, pg=pg, x=x, y=y)

    print("\nTransition dipoles")
    print(mu_trans.round(12))

    return w, x, y, mu_trans, info, oo_info


def dipole_polarizability(labels, coords, charge, spin, basis, angstrom=False,
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

    # Solve ground state
    t = time.time()
    en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
            na=na, nb=nb, h_ao=h_ao, r_ao=r_ao, c_guess=c_guess,
            t2_guess=t2_guess, niter=oo_niter, r_thresh=oo_rthresh,
            print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en_tot = en_elec + en_nuc
    oo_info = {'en_tot': en_tot, 't2': t2, 'c': c, **info}
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    print('time: {:8.1f}s'.format(time.time() - t))

    # LR inputs
    ac, bc = c
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
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

    foo = fermitools.oo.ocepa0.fock_xy(hxy=hoo, goxoy=goooo)
    fov = fermitools.oo.ocepa0.fock_xy(hxy=hov, goxoy=gooov)
    fvv = fermitools.oo.ocepa0.fock_xy(hxy=hvv, goxoy=govov)

    # Evaluate dipole moment as expectation value
    m1oo, m1vv = fermitools.oo.ocepa0.onebody_density(t2)
    mu = numpy.array([numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
                      for pxoo, pxvv in zip(poo, pvv)])
    print("Electric dipole:")
    print(mu.round(12))

    # Evaluate dipole polarizability by linear response
    pg = fermitools.lr.ocepa0.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    a, b = fermitools.lr.ocepa0.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)
    ad = fermitools.lr.ocepa0.hessian_zeroth_order_diagonal(foo=foo, fvv=fvv)
    t = time.time()
    r, info = fermitools.lr.solve.static_response(
            a=a, b=b, pg=pg, ad=ad, nvec=nvec, niter=niter, r_thresh=rthresh)
    alpha = numpy.dot(r.T, pg)
    print("Electric dipole polarizability tensor:")
    print(alpha.round(12))
    print('time: {:8.1f}s'.format(time.time() - t))

    return mu, alpha, info, oo_info
