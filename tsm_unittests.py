# tsm_unittests.py
import time
from output_capstone import print_to_file_and_terminal as ptf
from tsm import TriggeredSeriesModel


def run_tsm_unittests(X,y,column_headers,verbose=True,logfile=None):
    tsm = TriggeredSeriesModel(column_headers, logfile=logfile, verbose=verbose,
        on_disk=False, debug=True)

    ptf('Try set-up', logfile)
    tsm.setup(X,y)
    print X.head()
    print X.iloc[0][0:15,0:4]
    print X.iloc[0].shape
    tsm._check_trial_integrity()

    print 'Preprocess'
    Xpp = tsm.preprocess(X)
    # print Xpp.iloc[0][0:15,0:4]
    print Xpp.iloc[0].shape
    print 'Prune'
    Xp = tsm.prune_spots(Xpp, ['26B', '11R', '45B', '36B', '30R', '11B'], column_headers)
    # print Xp.iloc[0][0:15]
    print Xp.iloc[0].shape
    print tsm.trigger_indexes, tsm.trigger_spots

    print 'Featurize'
    Xtrig = tsm.featurize_triggers(Xp,30)
    print Xtrig.shape
    print tsm.trigger_feature_times
    print tsm.trigger_features
    print tsm.trigger_featurizers

    print 'Now try doing all of these steps with fit'

    return tsm
