from .math import broadcast_sum


def doubles_resolvent_denominator(eo1, eo2, ev1, ev2):
    return broadcast_sum({0: +eo1, 1: +eo2, 2: -ev1, 3: -ev2})
