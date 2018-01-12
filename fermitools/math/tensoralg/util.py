def parse_einsum_subscripts(subscripts):
    s = str.replace(subscripts, '...', '#')
    csv, rsub = str.split(s, '->') if '->' in s else (s, None)
    subs = csv.split(',')
    return subs, rsub
