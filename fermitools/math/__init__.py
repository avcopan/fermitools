from . import spinorb
from . import combinatorics
from . import asym
from . import sigma
from . import diis
from .ix import cast
from .ix import diagonal_indices
from .findif import central_difference
from .rav import ravel, raveler
from .urav import unravel, unraveler
from .trans import transform
from .ot import orth
from .ex import expm
from .dsk import callable_disk_array
from .dsk import callable_core_array

# Choose your einsum function:

from .tensoralg import einsum

# def einsum(*args, **kwargs):
#     """Call optimized einsum
#     """
#     import numpy
#     kwargs['optimize'] = 'optimal'
#     return numpy.einsum(*args, **kwargs)


__all__ = [
        'spinorb',
        'combinatorics',
        'asym',
        'sigma',
        'diis',
        'tensoralg',
        'transform',
        'cast',
        'diagonal_indices',
        'central_difference',
        'einsum',
        'ravel', 'raveler',
        'unravel', 'unraveler',
        'einsum',
        'orth',
        'expm',
        'callable_disk_array',
        'callable_core_array']
