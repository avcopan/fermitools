import simplehf
import simplehf.interface.pyscf as interface

import numpy
import scipy.linalg as spla

CHARGE = +1
SPIN = 1
BASIS = 'STO-3G'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

s_ao = interface.integrals.overlap(basis=BASIS, labels=LABELS,
                                   coords=COORDS)
h_ao = interface.integrals.core_hamiltonian(basis=BASIS, labels=LABELS,
                                            coords=COORDS)
g_ao = interface.integrals.repulsion(basis=BASIS, labels=LABELS,
                                     coords=COORDS)

na = simplehf.chem.elec.count_alpha(labels=LABELS, charge=CHARGE, spin=SPIN)
nb = simplehf.chem.elec.count_beta(labels=LABELS, charge=CHARGE, spin=SPIN)
ac, bc = interface.hf.unrestricted_orbitals(basis=BASIS, labels=LABELS,
                                            coords=COORDS, charge=CHARGE,
                                            spin=SPIN)
ad_ao = simplehf.hf.orb.density(na, ac)
bd_ao = simplehf.hf.orb.density(nb, bc)


af_ao, bf_ao = simplehf.hf.uhf.fock(h_ao, g_ao, ad_ao, bd_ao)

ae = simplehf.hf.orb.energies(s_ao, af_ao)
be = simplehf.hf.orb.energies(s_ao, bf_ao)

srt = simplehf.math.spinorb.mo.sort_order(7, na=na, nb=nb)
e = numpy.concatenate((ae, be))[srt, ]
c = spla.block_diag(ac, bc)[:, srt]

g_so = simplehf.math.spinorb.ao.expand(g_ao, ((0, 2), (1, 3)))
g = simplehf.math.trans.transform(g_so, {0: c, 1: c, 2: c, 3: c})

g = g - g.transpose((0, 1, 3, 2))

o = slice(None, na + nb)
v = slice(na + nb, None)
x = numpy.newaxis

t = (g[o, o, v, v]
     / (e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))

corr_energy = 1. / 4 * numpy.sum(g[o, o, v, v] * t)

print(corr_energy)
