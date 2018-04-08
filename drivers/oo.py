import numpy
import fermitools


def hf_orbitals(labels, coords, charge, spin, basis, angstrom, interface):
    coords = numpy.divide(coords, 0.52917720859) if angstrom else coords

    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    nbf = interface.integrals.nbf(basis, labels)
    no = na + nb
    nv = 2*nbf - no

    ca, cb = interface.hf.unrestricted_orbitals(
            basis, labels, coords, charge, spin)

    cao, cav = numpy.hsplit(ca, (na,))
    cbo, cbv = numpy.hsplit(cb, (nb,))

    co = (cao, cbo)
    cv = (cav, cbv)

    return co, cv, no, nv


def integrals(basis, labels, coords, angstrom, interface):
    coords = numpy.divide(coords, 0.52917720859) if angstrom else coords

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)
    p_ao = interface.integrals.dipole(basis, labels, coords)

    # Doesn't really belong here, but ...
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    print("\nnuclear replusion energy: {:20.15f}".format(en_nuc))

    return h_ao, r_ao, p_ao


def energy(method, labels, coords, charge, spin, basis, angstrom=False,
           maxiter=100, rthresh=1e-10, diis_start=3, diis_nvec=20,
           interface=None):
    h_ao, r_ao, p_ao = integrals(basis, labels, coords, angstrom, interface)
    co_guess, cv_guess, no, nv = hf_orbitals(
            labels, coords, charge, spin, basis, angstrom, interface)
    t2_guess = numpy.zeros((no, no, nv, nv))
    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    print("\nnuclear replusion energy: {:20.15f}".format(en_nuc))

    if method.lower() == 'ocepa0':
        en_elec, co, cv, t2, info = fermitools.oo.ocepa0.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=maxiter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
        fermitools.oo.ocepa0.compute_property(p_ao, co, cv, t2)
    elif method.lower() == 'odc12':
        en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=maxiter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
        fermitools.oo.odc12.compute_property(p_ao, co, cv, t2)
    else:
        raise Exception("Method {:s} does not exist.".format(str(method)))

    print("\ntotal energy: {:20.15f}".format(en_nuc + en_elec))


def spectrum(method, labels, coords, charge, spin, basis, angstrom=False,
             nroot=1, nconv=None, nguess=None, maxdim=None, maxiter=100,
             rthresh=1e-5, oo_maxiter=200, oo_rthresh=1e-10, diis_start=3,
             diis_nvec=20, disk=False, blsize=None, interface=None):
    h_ao, r_ao, p_ao = integrals(basis, labels, coords, angstrom, interface)
    co_guess, cv_guess, no, nv = hf_orbitals(
            labels, coords, charge, spin, basis, angstrom, interface)
    t2_guess = numpy.zeros((no, no, nv, nv))

    if method.lower() == 'ocepa0':
        en_elec, co, cv, t2, info = fermitools.oo.ocepa0.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=maxiter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
        fermitools.lr.ocepa0.solve_spectrum(
                h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
                nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
                rthresh=rthresh, print_conv=True, disk=disk, blsize=blsize,
                p_ao=p_ao)
    elif method.lower() == 'odc12':
        en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=maxiter, rthresh=rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
        fermitools.lr.odc12.solve_spectrum(
                h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
                nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
                rthresh=rthresh, print_conv=True, disk=disk, blsize=blsize,
                p_ao=p_ao)
    else:
        raise Exception("Method {:s} does not exist.".format(str(method)))
