import scripts.ocepa0 as ocepa0

import numpy
import scipy.linalg as spla

import fermitools
import interfaces.psi4 as interface

import os

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
t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
en_elec, c, t2 = ocepa0.solve(norb=norb, nocc=nocc, h_aso=h_aso,
                              g_aso=g_aso, c_guess=c,
                              t2_guess=t2_guess, niter=200,
                              e_thresh=1e-14, r_thresh=1e-13,
                              print_conv=True)

en_dx_func = ocepa0.orbital_gradient_functional(norb=norb, nocc=nocc,
                                                h_aso=h_aso,
                                                g_aso=g_aso, c=c)
en_dt_func = ocepa0.amplitude_gradient_functional(norb=norb, nocc=nocc,
                                                  h_aso=h_aso,
                                                  g_aso=g_aso, c=c)
en_dxdx_func = ocepa0.orbital_hessian_functional(norb=norb, nocc=nocc,
                                                 h_aso=h_aso,
                                                 g_aso=g_aso, c=c)
en_dxdt_func = ocepa0.mixed_hessian_functional(norb=norb, nocc=nocc,
                                               h_aso=h_aso,
                                               g_aso=g_aso, c=c)
en_dtdx_func = ocepa0.mixed_hessian_transp_functional(norb=norb, nocc=nocc,
                                                      h_aso=h_aso,
                                                      g_aso=g_aso, c=c)
en_dtdt_func = ocepa0.amplitude_hessian_functional(norb=norb, nocc=nocc,
                                                   h_aso=h_aso,
                                                   g_aso=g_aso, c=c)

no = nocc
nv = norb - nocc
x = numpy.zeros(no * nv)
t = numpy.ravel(fermitools.math.asym.compound_index(t2, {0: (0, 1),
                                                         1: (2, 3)}))

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data')

print("Computing blocks of the electronic gradient...")
en_dx = en_dx_func(x, t)
print(en_dx.round(8))
en_dt = en_dt_func(x, t)
print(en_dt.round(8))

print("Computing blocks of the electronic Hessian...")
en_dxdt = en_dxdt_func(x, t)
numpy.save(os.path.join(data_path, 'lr_ocepa0/neutral/en_dxdt.npy'), en_dxdt)
print("... finished mixed block")
en_dtdx = en_dtdx_func(x, t)
numpy.save(os.path.join(data_path, 'lr_ocepa0/neutral/en_dtdx.npy'), en_dtdx)
print("... finished mixed block transpose")
en_dxdx = en_dxdx_func(x, t)
numpy.save(os.path.join(data_path, 'lr_ocepa0/neutral/en_dxdx.npy'), en_dxdx)
print("... finished orbital block")
en_dtdt = en_dtdt_func(x, t)
numpy.save(os.path.join(data_path, 'lr_ocepa0/neutral/en_dtdt.npy'), en_dtdt)
print("... finished amplitude block")
