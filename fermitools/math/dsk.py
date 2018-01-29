import h5py
import tempfile


def callable_core_array(a):

    def _call():
        return a

    return _call


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
