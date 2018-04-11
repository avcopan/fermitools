import fermitools
import drivers
import interfaces.psi4 as interface
import numpy
import time


def run_test(labels, coords, outfile, exact_diagonal=False):
    charge = 0
    spin = 0
    basis = 'def2-sv(p)'
    # basis = 'sto-3g'
    nroot = 10
    nconv = 10
    maxiter = 100
    rthresh = 1e-5
    nguess = 10*nroot
    maxdim = 20*nroot
    oo_maxiter = 200
    oo_rthresh = 1e-10
    diis_start = 3
    diis_nvec = 20

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

    wref, _, _ = fermitools.lr.odc12.solve_spectrum(
            h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=4*nroot,
            nconv=nconv, nguess=4*nguess, maxdim=3*maxdim, maxiter=maxiter,
            rthresh=rthresh, print_conv=True, blsize=nroot, disk=True)
    W = wref[:nconv]

    t = time.time()
    w, z, info = fermitools.lr.odc12.solve_spectrum1(
            h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
            nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
            rthresh=rthresh, print_conv=True, exact_diagonal=exact_diagonal)
    w = w[:nconv]
    dt = time.time() - t
    niter = info['niter']
    nstates = sum(numpy.any(numpy.isclose(wi, W, atol=10*rthresh)) for wi in w)
    outfile.write('{:5d} {:5.2f} {:5d}'.format(niter, dt, nstates))
    outfile.flush()

    t = time.time()
    w, z, info = fermitools.lr.odc12.solve_spectrum2(
            h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
            nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
            rthresh=rthresh, print_conv=True, exact_diagonal=exact_diagonal)
    w = w[:nconv]
    dt = time.time() - t
    niter = info['niter']
    nstates = sum(numpy.any(numpy.isclose(wi, W, atol=10*rthresh)) for wi in w)
    outfile.write('{:5d} {:5.2f} {:5d}'.format(niter, dt, nstates))
    outfile.flush()

    t = time.time()
    w, z, info = fermitools.lr.odc12.solve_spectrum3(
            h_ao=h_ao, r_ao=r_ao, co=co, cv=cv, t2=t2, nroot=nroot,
            nconv=nconv, nguess=nguess, maxdim=maxdim, maxiter=maxiter,
            rthresh=rthresh, print_conv=True, exact_diagonal=exact_diagonal)
    w = w[:nconv]
    dt = time.time() - t
    niter = info['niter']
    nstates = sum(numpy.any(numpy.isclose(wi, W, atol=10*rthresh)) for wi in w)
    outfile.write('{:5d} {:5.2f} {:5d}'.format(niter, dt, nstates))
    outfile.flush()


if __name__ == '__main__':
    molecules = (
            [
                ('O', 'H', 'H'),
                ((0.,  0.000000000000, -0.123592575851),
                 (0.,  1.429453855079,  0.980751933841),
                 (0., -1.429453855079,  0.980751933841))
            ],
            [
                ('N', 'N'),
                ((0., 0., -1.036228392795),
                 (0., 0.,  1.036228392795))
            ],
            [
                ('H', 'C', 'N'),
                ((0., 0., -3.050529108951),
                 (0., 0., -1.050440604637),
                 (0., 0.,  1.119678949420))
            ],
            [
                ('C', 'O', 'H', 'H'),
                ((0.,  0.000000000000, -1.135781219104),
                 (0.,  0.000000000000,  1.132863002620),
                 (0., -1.761718374921, -2.227902775989),
                 (0.,  1.761718374921, -2.227902775989))
            ],
            [
                ('C', 'C', 'H', 'H', 'O', 'H', 'H'),
                ((0.,  2.308607877028, -0.693815416341),
                 (0., -0.217883674964, -0.800960002254),
                 (0.,  3.328678970321,  1.091346846579),
                 (0.,  3.397177089758, -2.429376646356),
                 (0., -1.858030821325,  1.195561345788),
                 (0., -1.239760617162, -2.580619135922),
                 (0., -0.891692867584,  2.742257610364))
            ]
    )

    f = open('benchmark.dat', 'w+')
    for d in [True, False]:
        for labels, coords in molecules[-1:]:
            f.write(str(labels))
            f.write('\n')
            f.flush()
            run_test(labels, coords, outfile=f, exact_diagonal=d)
            f.write('\n')
            f.flush()
    f.close()
