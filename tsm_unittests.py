# tsm_unittests.py
import time
from output_capstone import print_to_file_and_terminal as ptf
from tsm import TriggeredSeriesModel


def run_tsm_unittests(X,y,verbose=True,logfile=None):
    tsm = TriggeredSeriesModel(logfile=logfile, verbose=verbose)

    ptf('Try set-up', logfile)
    tsm.setup(X,y)



    return tsm
