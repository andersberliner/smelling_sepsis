# nonetype.py
# Anders Berliner
# 20160115
#
# Holds the nonetype class, which has the transform method that does nothing,
# so for no reducing, no scaling, no featurizing, can use all of the same
# calls

class NoneType(object):
    def __init__(self):
        pass
    def fit_transform(self, X=None, y=None):
        return X
    def transform(self, X=None):
        return X
    def predict(self, X=None):
        return X
    def fit(self, X=None, y=None):
        pass

class NoneTypeFeaturizer(NoneType):
    def __init__(self):
        pass
    def fit_transform(self, X=None, y=None):
        # return transformed X and score
        return X, 1.0


class NoneTypeScaler(NoneType):
    def __init__(self):
        pass


class NoneTypeReducer(NoneType):
    def __init__(self):
        pass
