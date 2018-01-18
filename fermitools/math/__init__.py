from . import linalg
from . import spinorb
from . import combinatorics
from . import asym
from .bcast import broadcast_sum
from .findif import central_difference
from .rav import ravel, raveler
from .urav import unravel, unraveler
from .trans import transform

# Choose your einsum function:

from .tensoralg import einsum

#def einsum(*args, **kwargs):
#    """Call optimized einsum
#    """
#    import numpy
#    kwargs['optimize'] = 'optimal'
#    return numpy.einsum(*args, **kwargs)


__all__ = [
        'linalg',
        'spinorb',
        'combinatorics',
        'asym',
        'tensoralg',
        'transform',
        'broadcast_sum',
        'central_difference',
        'einsum',
        'ravel', 'raveler',
        'unravel', 'unraveler',
        'einsum']
