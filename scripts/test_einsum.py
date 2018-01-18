import numpy
import time

numpy.random.seed(10)
a = numpy.random.random((350, 350, 80, 80))
b = numpy.random.random((10, 10, 80, 80))

print("Einsum")
t = time.time()
numpy.einsum("abcd,ijcd->ijab", a, b)
print(time.time() - t)

print("Tensordot")
t = time.time()
numpy.tensordot(a, b, axes=([2, 3], [2, 3]))
print(time.time() - t)
