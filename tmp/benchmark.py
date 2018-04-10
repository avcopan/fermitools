import fermitools
import drivers
import interfaces.psi4 as interface
import numpy


labels = ('C', 'C', 'H', 'H', 'H', 'H')
coords = ((0.,  0.000000000000,  1.257707399600),
          (0.,  0.000000000000, -1.257707399600),
          (0.,  1.733261359900,  2.319645032500),
          (0., -1.733261359900,  2.319645032500),
          (0.,  1.733261359900, -2.319645032500),
          (0., -1.733261359900, -2.319645032500))
charge = 0
spin = 0
basis = 'def2-sv(p)'
# basis = 'sto-3g'
nconv = 10
nroot = 2*nconv
nguess = 10*nconv
maxdim = 20*nconv
blsize = nroot // 2
maxiter = 100
rthresh = 1e-5
oo_maxiter = 200
oo_rthresh = 1e-10
diis_start = 3
diis_nvec = 20
disk = True

h_ao, r_ao, p_ao = drivers.integrals(
        basis=basis, labels=labels, coords=coords, angstrom=False,
        interface=interface)
co_guess, cv_guess, no, nv = drivers.hf_orbitals(
        labels=labels, coords=coords, charge=charge, spin=spin, basis=basis,
        angstrom=False, interface=interface)
t2_guess = numpy.zeros((no, no, nv, nv))

en_elec, co, cv, t2, info = fermitools.oo.odc12.solve(
            h_ao=h_ao, r_ao=r_ao, co_guess=co_guess, cv_guess=cv_guess,
            t2_guess=t2_guess, maxiter=oo_maxiter, rthresh=oo_rthresh,
            diis_start=diis_start, diis_nvec=diis_nvec, print_conv=True)

fermitools.lr.odc12.solve_spectrum(
        h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
        nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
        rthresh=rthresh, print_conv=True, disk=disk, blsize=blsize)
