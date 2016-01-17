# utils_seriesmodel.py

def print_dict_values(d):
    # NOTE: needs debugging as some classes still cause error when trying to
    # print
    output = ''
    keys = d.keys()
    keys.sort()
    for k in keys:
        v = d[k]
        if type(v) in [str, int, float, bool]:
            output += '\n%s: %s' % (k, v)
        else:
            output += '\n%s: %s' % (k, type(v))
    return output
