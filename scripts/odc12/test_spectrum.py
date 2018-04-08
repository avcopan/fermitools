import os
import numpy
import fermitools
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
W = numpy.load(os.path.join(data_path, 'neutral/w.npy'))


def test__main():
    import drivers
    import interfaces.psi4 as interface

    nroot = 11
    nconv = 10
    labels = ('O', 'H', 'H')
    coords = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))
    charge = 0
    spin = 0
    basis = 'sto-3g'
    oo_maxiter = 200
    oo_rthresh = 1e-10
    diis_start = 3
    diis_nvec = 20
    maxiter = 50
    rthresh = 1e-7
    nguess = nroot*8
    maxdim = nroot*10
    blsize = 5
    disk = True

    h_ao, r_ao, p_ao = drivers.integrals(
            basis, labels, coords, angstrom=False, interface=interface)
    co_guess, cv_guess, no, nv = drivers.hf_orbitals(
            labels, coords, charge, spin, basis, angstrom=False,
            interface=interface)
    t2_guess = numpy.zeros((no, no, nv, nv))

    en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
                h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
                t2_guess=t2_guess, maxiter=oo_maxiter, rthresh=oo_rthresh,
                diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)
    w, z, info = fermitools.lr.odc12.solve_spectrum(
            h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
            nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
            rthresh=rthresh, print_conv=True, disk=disk, blsize=blsize)

    w = numpy.sort(w)
    print(W[:nconv])
    assert_almost_equal(w[:nconv], W[:nconv], decimal=10)


if __name__ == '__main__':
    test__main()
