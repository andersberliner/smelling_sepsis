# utils_capstone.py
# Anders Berliner
# 20160111

def print_to_file_and_terminal(s, logfile=None, to_file=True, to_terminal=True):
    if to_file:
        if logfile:
            print >>logfile, s
    if to_terminal:
        print s
