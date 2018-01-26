import numpy

from ..math import expm
from ..math.spinorb import decompose_onebody


def orbital_rotation(co, cv, t1):
    no, nv = t1.shape
    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))
    a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
    aco, bco = co
    acv, bcv = cv
    na, nb = aco.shape[1], bco.shape[1]
    au, bu = decompose_onebody(expm(a), na=na, nb=nb)
    ac = numpy.dot(numpy.hstack((aco, acv)), au)
    bc = numpy.dot(numpy.hstack((bco, bcv)), bu)
    aco, acv = numpy.hsplit(ac, (na,))
    bco, bcv = numpy.hsplit(bc, (nb,))
    return (aco, bco), (acv, bcv)
