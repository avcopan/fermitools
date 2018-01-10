def parse_einsum_subscripts(subscripts):
    csv, rsub = (subscripts.split('->') if '->' in subscripts
                 else (subscripts, None))
    subs = csv.split(',')
    return subs, rsub


def contraction_indices(sub1, sub2):
    c = ''.join(x for x in sub1 if x in sub2)
    return c


def free_indices(sub1, sub2):
    f1 = ''.join(x for x in sub1 if x not in sub2)
    f2 = ''.join(x for x in sub2 if x not in sub1)
    return f1 + f2
