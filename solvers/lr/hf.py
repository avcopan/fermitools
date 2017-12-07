import numpy
import scipy


def solve_spectrum(nroots, a, b, pc, x_guess, y_guess, r_thresh=1e-10):
    x = x_guess
    y = y_guess
    xy = numpy.hstack((x, y))
    yx = numpy.hstack((y, x))

    for _ in range(200):
        print("ITERATION {:d}".format(_))

        ex = a(xy) + b(yx)
        ey = b(xy) + a(yx)
        mx = +xy
        my = -yx

        e = (+ numpy.dot(xy.T, ex)
             + numpy.dot(yx.T, ey))
        m = (+ numpy.dot(xy.T, mx)
             + numpy.dot(yx.T, my))

        print(numpy.linalg.eigvalsh(e)[:10])
        k, u = scipy.linalg.eigh(a=m, b=e)

        w = 1. / k

        rx = numpy.dot(ex, u) - numpy.dot(mx, u) * w
        ry = numpy.dot(ey, u) - numpy.dot(my, u) * w

        print('before:')
        print(x.shape)
        x_new = pc(+w)(rx)
        y_new = pc(-w)(ry)
        u = numpy.bmat([[x, y], [y, x]])
        u_new = numpy.bmat([[x_new, y_new], [y_new, x_new]])
        u_orth = u_new - numpy.linalg.multi_dot([u, u.T, u_new])
        u_norm = scipy.linalg.orth(u_orth)
        u = numpy.array(numpy.hstack((u, u_norm)))
        xy, yx = numpy.split(u, 2, axis=0)
        print('after:')
        print(x.shape)

        r_norm = scipy.linalg.norm((rx[:, -nroots:], ry[:, -nroots:]))
        print(r_norm)
        print(w[-nroots:][::-1])

        converged = r_norm < r_thresh

        if converged:
            break

    return w[-nroots:][::-1]
