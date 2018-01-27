import h5py
import tempfile


def callable_disk_array(a, name=None, fname=None, truncate=False):
    name = name if name is not None else 'a'
    fname = fname if fname is not None else tempfile.mkstemp(suffix='.hdf5')[1]
    mode = 'r+' if not truncate else 'w'

    # write array to disk and free its memory
    f = h5py.File(name=fname, mode=mode)
    f[name] = a
    f.close()
    a = None

    def _call():
        # read array from disk
        f = h5py.File(name=fname, mode='r')
        a = f[name].value
        f.close()
        return a

    return _call


if __name__ == '__main__':
    import time
    import numpy
    a_ = callable_disk_array(numpy.random.random((100, 100, 100, 100)))
    b_ = callable_disk_array(numpy.random.random((101, 101, 101, 101)))
    time.sleep(5)
    a = a_()
    print(a.shape)
    time.sleep(5)
    b = b_()
    print(b.shape)
    time.sleep(5)
