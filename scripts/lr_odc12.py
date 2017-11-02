import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface
from . import odc12


def main():
    CHARGE = +0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(norb=norb, nocc=nocc, h_aso=h_aso,
                                 g_aso=g_aso, c_guess=c,
                                 t2_guess=t2_guess, niter=200,
                                 e_thresh=1e-14, r_thresh=1e-12,
                                 print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Test the orbital and amplitude gradients
    import os

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    no = nocc
    nv = norb - nocc

    x = numpy.zeros(no * nv)
    t = numpy.ravel(fermitools.math.asym.compound_index(t2, {0: (0, 1),
                                                             1: (2, 3)}))

    en_dxdx_func = odc12.orbital_hessian_functional(norb=norb, nocc=nocc,
                                                    h_aso=h_aso, g_aso=g_aso,
                                                    c=c, npts=11)
    en_dtdx_func = odc12.mixed_hessian_functional(norb=norb, nocc=nocc,
                                                  h_aso=h_aso, g_aso=g_aso,
                                                  c=c, npts=11)
    en_dxdt_func = odc12.mixed_hessian_transp_functional(norb=norb,
                                                         nocc=nocc,
                                                         h_aso=h_aso,
                                                         g_aso=g_aso,
                                                         c=c, npts=11)
    en_dtdt_func = odc12.amplitude_hessian_functional(norb=norb, nocc=nocc,
                                                      h_aso=h_aso,
                                                      g_aso=g_aso,
                                                      c=c, npts=11)

    def generate_orbital_hessian():
        en_dxdx = en_dxdx_func(x, t)
        numpy.save(os.path.join(data_path, 'lr_odc12/neutral/en_dxdx.npy'),
                   en_dxdx)

    def generate_mixed_hessian():
        en_dtdx = en_dtdx_func(x, t)
        numpy.save(os.path.join(data_path, 'lr_odc12/neutral/en_dtdx.npy'),
                   en_dtdx)

    def generate_mixed_hessian_transp():
        en_dxdt = en_dxdt_func(x, t)
        numpy.save(os.path.join(data_path, 'lr_odc12/neutral/en_dxdt.npy'),
                   en_dxdt)

    def generate_amplitude_hessian():
        en_dtdt = en_dtdt_func(x, t)
        numpy.save(os.path.join(data_path, 'lr_odc12/neutral/en_dtdt.npy'),
                   en_dtdt)

    print("Numerical Hessian calculations ...")
    generate_orbital_hessian()
    print("... orbital Hessian finished")
    generate_mixed_hessian()
    print("... mixed Hessian finished")
    generate_mixed_hessian_transp()
    print("... transposed mixed Hessian finished")
    generate_amplitude_hessian()
    print("... amplitude Hessian finished")


if __name__ == '__main__':
    main()
