import time
import numpy
import fermitools


def energy(labels, coords, charge, spin, basis, angstrom=False, niter=200,
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
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    # Mean-field guess orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            basis, labels, coords, charge, spin)

    t2_guess = numpy.zeros((no, no, nv, nv))
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))
    foo = fermitools.oo.cepa0.fock_xy(hoo, goooo)
    fvv = fermitools.oo.cepa0.fock_xy(hvv, govov)

    # Solve ground state
    t = time.time()
    en_corr, t2, info = fermitools.oo.cepa0.solve_diis(
            foo=foo, fvv=fvv, goooo=goooo, goovv=goovv, govov=govov,
            gvvvv=gvvvv, t2_guess=t2_guess, niter=niter, rthresh=rthresh,
            print_conv=True)
    print("\nCEPA0 correlation energy:")
    print('{:20.15f}'.format(en_corr))
    print('time: {:8.1f}s'.format(time.time() - t))

    return en_corr, t2, info
