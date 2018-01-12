from . import linalg
from . import spinorb
from . import combinatorics
from . import asym
from .trans import transform, transformer
from .bcast import broadcast_sum
from .findif import central_difference
from .rav import ravel, raveler
from .urav import unravel, unraveler


# Choose your einsum function:

from .tensoralg import einsum

# def einsum(*args, **kwargs):
#     """Call optimized einsum
#     """
#     import numpy
#     kwargs['optimize'] = 'optimal'
#     return numpy.einsum(*args, **kwargs)


__all__ = [
        'linalg',
        'spinorb',
        'combinatorics',
        'asym',
        'tensoralg',
        'transform', 'transformer',
        'broadcast_sum',
        'central_difference',
        'einsum',
        'ravel', 'raveler',
        'unravel', 'unraveler',
        'einsum']
