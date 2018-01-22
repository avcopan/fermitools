from . import spinorb
from . import combinatorics
from . import asym
from . import sigma
from .bcast import broadcast_sum
from .findif import central_difference
from .rav import ravel, raveler
from .urav import unravel, unraveler
from .trans import transform
from .ot import orth
from .ex import expm

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
        'tensoralg',
        'transform',
        'broadcast_sum',
        'central_difference',
        'einsum',
        'ravel', 'raveler',
        'unravel', 'unraveler',
        'einsum',
        'orth',
        'expm']
