def add(f, g):

    def _sum(*args, **kwargs):
        return f(*args, **kwargs) + g(*args, **kwargs)

    return _sum


def sub(f, g):

    def _diff(*args, **kwargs):
        return f(*args, **kwargs) - g(*args, **kwargs)

    return _diff
