# utils_seriesmodel.py
import numpy as np


def print_dict_values(d, name='dict'):
    # NOTE: needs debugging as some classes still cause error when trying to
    # print
    output = name
    keys = d.keys()
    keys.sort()
    for k in keys:
        v = d[k]
        if type(v) in [str, int, float, bool]:
            output += '\n%s: %s' % (k, v)
        else:
            output += '\n%s: %s' % (k, type(v))
    return output


def extract_column_indexes(trigger_spots, column_headers):
    trigger_indexes = [0] # the time index is kept
    for spot in trigger_spots:
        trigger_indexes.append(np.argmax(column_headers == spot))
    # print trigger_indexes
    # print column_headers[trigger_indexes]
    return trigger_indexes


def array_slicer(x, trigger_indexes):
    return x[:, trigger_indexes].tolist()

def pd_slicer(X, trigger_indexes):
    Z = X.apply(lambda x: array_slicer(x, trigger_indexes))
    ZZ = Z.apply(lambda x: np.array(x))
    return ZZ
