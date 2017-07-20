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

ac, bc = interface.hf.unrestricted_orbitals(basis=BASIS, labels=LABELS,
                                            coords=COORDS, charge=CHARGE,
                                            spin=SPIN)

naocc = simplehf.chem.elec.count_alpha(labels=LABELS, charge=CHARGE, spin=SPIN)
nbocc = simplehf.chem.elec.count_beta(labels=LABELS, charge=CHARGE, spin=SPIN)
sort_order = simplehf.math.spinorb.mo.sort_order(dim=7, na=naocc, nb=nbocc)

c = spla.block_diag(ac, bc)[:, sort_order]

t_ao = interface.integrals.kinetic(basis=BASIS, labels=LABELS,
                                   coords=COORDS)
v_ao = interface.integrals.nuclear(basis=BASIS, labels=LABELS,
                                   coords=COORDS)
r_ao = interface.integrals.repulsion(basis=BASIS, labels=LABELS,
                                     coords=COORDS)

h_ao = t_ao + v_ao
h_so = simplehf.math.spinorb.ao.expand(h_ao, ((0, 1),))
r_so = simplehf.math.spinorb.ao.expand(r_ao, ((0, 2), (1, 3)))

h = simplehf.math.trans.transform(h_so, {0: c, 1: c})
r = simplehf.math.trans.transform(r_so, {0: c, 1: c, 2: c, 3: c})

g = r - r.transpose((0, 1, 3, 2))

nocc = naocc + nbocc

o = slice(None, nocc)
v = slice(nocc, None)
x = numpy.newaxis

f = h + numpy.einsum('piqi->pq', g[:, o, :, o])
e = numpy.diag(f)

t = (g[o, o, v, v]
     / (e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))

energy = 1. / 4 * numpy.sum(g[o, o, v, v] * t)

print(energy)
